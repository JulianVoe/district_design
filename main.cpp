#include "dd_vrp.h"
#include <iostream>

int main() {
    using namespace ddvrp;

    VRPInstance inst;
    // Fill inst.customers, inst.dist, inst.vehicle_capacity, inst.max_vehicles
    // Node 0 must be depot.

    OrToolsPrizeCollectingCvrpSolver solver(5000);
    ScenarioGenerator gen(1234);

    // ---- Model B: partition + recourse ----
    Partition P;
    // e.g., simple partition: each customer its own district
    P.districts.resize(inst.num_customers());
    for (int v = 1; v < inst.num_nodes(); ++v) {
        P.districts[v-1].push_back(v);
    }

    Cost exp_cost_B = model_b::estimate_expected_cost(
        inst, P, solver, /*num_samples=*/1000, gen
    );
    std::cout << "Model B expected cost: " << exp_cost_B << "\n";

    // ---- Model A: a-priori tours + shortcut ----
    APrioriTours apriori = model_a::build_apriori_tours(inst, solver);

    Cost exp_cost_A = model_a::estimate_expected_cost(
        inst, apriori, /*num_samples=*/1000, gen
    );
    std::cout << "Model A expected cost: " << exp_cost_A << "\n";

    return 0;
}
