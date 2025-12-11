#include "dd_vrp.h"

// OR-Tools headers
#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/base/logging.h"

#include <limits>
#include <cmath>

namespace ddvrp {

// ---------- OrToolsPrizeCollectingCvrpSolver ----------

using namespace operations_research;

OrToolsPrizeCollectingCvrpSolver::OrToolsPrizeCollectingCvrpSolver(
    int search_time_limit_ms
) : time_limit_ms_(search_time_limit_ms) {}

VRPSolution OrToolsPrizeCollectingCvrpSolver::solve(
    const VRPInstance&         instance,
    const std::vector<NodeId>& active_customers,
    const std::vector<double>& penalties
) {
    const int depot = instance.depot_id();

    // Build local node set: 0 = depot, 1..m = active customers
    const int m = static_cast<int>(active_customers.size());
    const int local_n = 1 + m;

    // Map local index -> global node id
    std::vector<NodeId> local_to_global(local_n);
    local_to_global[0] = depot;
    for (int i = 0; i < m; ++i) {
        local_to_global[1 + i] = active_customers[i];
    }

    const int num_vehicles = instance.max_vehicles;

    RoutingIndexManager manager(local_n, num_vehicles, 0);
    RoutingModel routing(manager);

    const auto& dist = instance.dist;

    // Cost callback: travel cost from i to j using global dist
    const int transit_cb_index =
        routing.RegisterTransitCallback(
            [&manager, &local_to_global, &dist](int64_t from_index,
                                                int64_t to_index) -> int64_t {
                int from_node = manager.IndexToNode(from_index).value();
                int to_node   = manager.IndexToNode(to_index).value();
                NodeId g_from = local_to_global[from_node];
                NodeId g_to   = local_to_global[to_node];
                double c      = dist[g_from][g_to];
                if (c < 0) c = 0; // just in case
                // OR-Tools expects integer costs; assume dist is scaled or integer.
                return static_cast<int64_t>(std::llround(c));
            }
        );

    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index);

    // Capacity dimension: demand at each node (0 at depot)
    const int demand_cb_index =
        routing.RegisterUnaryTransitCallback(
            [&manager, &local_to_global, &instance](int64_t from_index) -> int64_t {
                int from_node = manager.IndexToNode(from_index).value();
                NodeId g      = local_to_global[from_node];
                if (g == instance.depot_id()) return 0;
                double d = instance.customers[g].demand;
                if (d < 0) d = 0;
                return static_cast<int64_t>(std::llround(d));
            }
        );

    routing.AddDimension(
        demand_cb_index,
        0,  // no slack
        static_cast<int64_t>(std::llround(instance.vehicle_capacity)),
        true, // start cumul at zero
        "Capacity"
    );

    // Prize-collecting via disjunctions
    // Local nodes 1..m are customers, each with penalty penalties[global_id]
    for (int local = 1; local < local_n; ++local) {
        NodeId g = local_to_global[local];
        double pen = (g >= 0 && g < static_cast<NodeId>(penalties.size()))
                     ? penalties[g]
                     : std::numeric_limits<double>::infinity();

        int64_t pen_int;
        if (std::isinf(pen) || pen >= std::numeric_limits<int64_t>::max() / 4) {
            pen_int = std::numeric_limits<int64_t>::max() / 4;
        } else if (pen < 0) {
            pen_int = 0;
        } else {
            pen_int = static_cast<int64_t>(std::llround(pen));
        }
        routing.AddDisjunction({manager.NodeToIndex(local)}, pen_int);
    }

    // Search parameters
    RoutingSearchParameters search_parameters = DefaultRoutingSearchParameters();
    search_parameters.set_first_solution_strategy(
        FirstSolutionStrategy::PARALLEL_CHEAPEST_INSERTION);
    search_parameters.set_local_search_metaheuristic(
        LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
    search_parameters.mutable_time_limit()->set_milliseconds(time_limit_ms_);

    const Assignment* sol = routing.SolveWithParameters(search_parameters);

    VRPSolution result;
    if (!sol) {
        result.total_cost = std::numeric_limits<Cost>::infinity();
        return result;
    }

    // Extract routes
    result.total_cost = static_cast<Cost>(sol->ObjectiveValue());

    for (int v = 0; v < num_vehicles; ++v) {
        int64_t index = routing.Start(v);
        // Empty vehicle: next is end immediately
        if (routing.IsEnd(sol->Value(routing.NextVar(index)))) continue;

        Route route;
        route.vehicle_id = v;

        // Walk along the route
        while (!routing.IsEnd(index)) {
            int node = manager.IndexToNode(index).value();
            route.nodes.push_back(local_to_global[node]);
            index = sol->Value(routing.NextVar(index));
        }
        // Add final depot
        int node = manager.IndexToNode(index).value();
        route.nodes.push_back(local_to_global[node]);

        // Compute route cost explicitly (for clarity)
        Cost c = 0.0;
        for (size_t i = 1; i < route.nodes.size(); ++i) {
            c += instance.dist[route.nodes[i-1]][route.nodes[i]];
        }
        route.cost = c;
        result.routes.push_back(std::move(route));
    }

    return result;
}

// ---------- ScenarioGenerator ----------

ScenarioGenerator::ScenarioGenerator(std::uint64_t seed)
    : rng_(seed) {}

Scenario ScenarioGenerator::sample(const VRPInstance& instance) {
    Scenario s;
    s.active.assign(instance.num_nodes(), false);

    s.active[instance.depot_id()] = true;
    for (int v = 1; v < instance.num_nodes(); ++v) {
        double p = instance.customers[v].p;
        if (p <= 0.0) {
            s.active[v] = false;
        } else if (p >= 1.0) {
            s.active[v] = true;
        } else {
            std::bernoulli_distribution bern(p);
            s.active[v] = bern(rng_);
        }
    }
    return s;
}

// ---------- Model B (partition + recourse) ----------

namespace model_b {

Cost estimate_expected_cost(
    const VRPInstance&     instance,
    const Partition&       partition,
    DeterministicVRPSolver& solver,
    int                    num_samples,
    ScenarioGenerator&     generator
) {
    if (num_samples <= 0) return 0.0;

    std::vector<double> penalties(instance.num_nodes());
    for (int v = 0; v < instance.num_nodes(); ++v) {
        penalties[v] = instance.customers[v].penalty;
    }

    Cost sum_cost = 0.0;

    for (int s = 0; s < num_samples; ++s) {
        Scenario scen = generator.sample(instance);
        Cost scenario_cost = 0.0;

        for (const auto& district : partition.districts) {
            // Build list of realized customers in this district
            std::vector<NodeId> active;
            active.reserve(district.size());
            for (NodeId v : district) {
                if (v <= 0 || v >= instance.num_nodes()) continue;
                if (scen.active[v]) active.push_back(v);
            }
            if (active.empty()) continue;

            VRPSolution sol = solver.solve(instance, active, penalties);
            scenario_cost += sol.total_cost;
        }

        sum_cost += scenario_cost;
    }

    return sum_cost / static_cast<Cost>(num_samples);
}

} // namespace model_b

// ---------- Model A (a-priori tours + shortcut) ----------

namespace model_a {

APrioriTours build_apriori_tours(
    const VRPInstance&     instance,
    DeterministicVRPSolver& solver
) {
    // All customers are mandatory: penalties = +∞
    std::vector<NodeId> all_customers;
    all_customers.reserve(instance.num_customers());
    for (int v = 1; v < instance.num_nodes(); ++v) {
        all_customers.push_back(v);
    }

    std::vector<double> penalties(instance.num_nodes(),
                                  std::numeric_limits<double>::infinity());

    VRPSolution sol = solver.solve(instance, all_customers, penalties);

    APrioriTours apr;
    apr.tours.reserve(sol.routes.size());
    for (const auto& r : sol.routes) {
        apr.tours.push_back(r.nodes);
    }
    return apr;
}

// Evaluate one scenario by shortcutting and greedy capacity handling.
Cost evaluate_scenario(
    const VRPInstance& instance,
    const APrioriTours& apriori,
    const Scenario&    scen
) {
    const int depot = instance.depot_id();
    Cost total_cost = 0.0;

    for (const auto& tour : apriori.tours) {
        if (tour.size() < 2) continue; // ignore degenerate

        std::vector<NodeId> realized_path;
        realized_path.reserve(tour.size());
        realized_path.push_back(depot);

        double load = 0.0;
        Cost   penalty_cost = 0.0;

        // We assume tour = [depot, ..., depot].
        // Traverse interior nodes, apply probability and capacity.
        for (size_t i = 1; i + 1 < tour.size(); ++i) {
            NodeId v = tour[i];
            if (v == depot) continue;
            if (v <= 0 || v >= instance.num_nodes()) continue;
            if (!scen.active[v]) continue;

            double d = instance.customers[v].demand;
            if (load + d <= instance.vehicle_capacity) {
                load += d;
                realized_path.push_back(v);
            } else {
                // drop v, pay penalty, do not visit
                penalty_cost += instance.customers[v].penalty;
            }
        }

        realized_path.push_back(depot);

        // Travel cost for this vehicle
        Cost travel = 0.0;
        for (size_t i = 1; i < realized_path.size(); ++i) {
            NodeId u = realized_path[i-1];
            NodeId v = realized_path[i];
            travel += instance.dist[u][v];
        }

        total_cost += travel + penalty_cost;
    }

    return total_cost;
}

Cost estimate_expected_cost(
    const VRPInstance& instance,
    const APrioriTours& apriori,
    int                 num_samples,
    ScenarioGenerator&  generator
) {
    if (num_samples <= 0) return 0.0;

    Cost sum_cost = 0.0;
    for (int s = 0; s < num_samples; ++s) {
        Scenario scen = generator.sample(instance);
        sum_cost += evaluate_scenario(instance, apriori, scen);
    }
    return sum_cost / static_cast<Cost>(num_samples);
}

} // namespace model_a

} // namespace ddvrp

