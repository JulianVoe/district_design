#pragma once

#include <vector>
#include <cstdint>
#include <random>

namespace ddvrp {

// ---------- Basic types ----------

using NodeId    = int32_t;
using VehicleId = int32_t;
using Cost      = double;

// ---------- Input data ----------

struct Customer {
    NodeId id;         // Node id, 0 is reserved for depot
    double x = 0.0;    // Optional coordinates (not strictly used here)
    double y = 0.0;
    double demand = 0.0;   // w_v > 0 for v != 0, 0 for depot
    double p = 1.0;        // activation probability
    double penalty = 0.0;  // drop penalty π_v (can be large)
};

struct VRPInstance {
    // Node 0 is depot, nodes 1..n-1 are customers.
    std::vector<Customer> customers;

    // Metric closure distance matrix: dist[i][j] is travel cost i->j.
    std::vector<std::vector<Cost>> dist;

    double vehicle_capacity = 0.0;
    int    max_vehicles     = 1;   // upper bound used by deterministic solver

    NodeId depot_id() const { return 0; }
    int num_nodes() const   { return static_cast<int>(customers.size()); }
    int num_customers() const { return num_nodes() - 1; }
};

// ---------- VRP solution representation ----------

struct Route {
    VehicleId          vehicle_id;
    std::vector<NodeId> nodes;   // includes depot at start and end
    Cost               cost = 0.0;
};

struct VRPSolution {
    std::vector<Route> routes;
    Cost               total_cost = 0.0;
};

// ---------- Deterministic solver interface ----------

class DeterministicVRPSolver {
public:
    virtual ~DeterministicVRPSolver() = default;

    // Solve a prize-collecting CVRP on a subset of customers.
    //
    // - instance: full instance
    // - active_customers: list of customer node ids (1..n-1) that are eligible
    //   in this subproblem (district ∩ scenario)
    // - penalties: penalty for dropping each *global* node id; typically
    //   penalties[0] is ignored (depot), penalties[v] for v>=1 is π_v,
    //   or +∞ if mandatory
    virtual VRPSolution solve(
        const VRPInstance&           instance,
        const std::vector<NodeId>&   active_customers,
        const std::vector<double>&   penalties
    ) = 0;
};

// ---------- OR-Tools-based deterministic solver ----------

class OrToolsPrizeCollectingCvrpSolver final : public DeterministicVRPSolver {
public:
    explicit OrToolsPrizeCollectingCvrpSolver(int search_time_limit_ms = 10000);

    VRPSolution solve(
        const VRPInstance&           instance,
        const std::vector<NodeId>&   active_customers,
        const std::vector<double>&   penalties
    ) override;

private:
    int time_limit_ms_;
};

// ---------- Stochastic layer (scenario) ----------

struct Scenario {
    // active[v] == true if node v is realized; v=0 is depot (always true)
    std::vector<bool> active;
};

class ScenarioGenerator {
public:
    explicit ScenarioGenerator(std::uint64_t seed = 42);

    Scenario sample(const VRPInstance& instance);

private:
    std::mt19937_64 rng_;
};

// ---------- District partition (Model B) ----------

struct Partition {
    // districts[i] is the list of customer node ids assigned to district i
    std::vector<std::vector<NodeId>> districts;
};

namespace model_b {

// Estimate expected cost for a given partition under Model B
// (partition + recourse via deterministic prize-collecting CVRP).
Cost estimate_expected_cost(
    const VRPInstance&     instance,
    const Partition&       partition,
    DeterministicVRPSolver& solver,
    int                    num_samples,
    ScenarioGenerator&     generator
);

} // namespace model_b

// ---------- A-priori tours (Model A) ----------

struct APrioriTours {
    // tours[i] is a full cycle for vehicle i: depot ... depot
    std::vector<std::vector<NodeId>> tours;
};

namespace model_a {

// Build deterministic a-priori tours using the deterministic solver
// on the full instance (all customers mandatory).
APrioriTours build_apriori_tours(
    const VRPInstance&     instance,
    DeterministicVRPSolver& solver
);

// Evaluate cost of a-priori tours under a single scenario:
// shortcut inactive customers and greedily drop customers
// that would violate capacity, paying penalties.
Cost evaluate_scenario(
    const VRPInstance& instance,
    const APrioriTours& apriori,
    const Scenario&    scen
);

// Monte Carlo estimate of expected cost under Model A.
Cost estimate_expected_cost(
    const VRPInstance& instance,
    const APrioriTours& apriori,
    int                 num_samples,
    ScenarioGenerator&  generator
);

} // namespace model_a

} // namespace ddvrp
