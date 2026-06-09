#include "agent/planner.hpp"
#include <limits>

namespace uik::agent {

Planner::Planner() : Planner(Config{}) {}

Planner::Planner(Config config) : config_(config), rng_(config.seed) {}

std::vector<Action> Planner::plan(const State& current,
                                    const State& goal,
                                    IWorldModel& world_model,
                                    int horizon) {
    if (horizon <= 0) return {};

    std::vector<Action> best_plan;
    Real best_dist = std::numeric_limits<Real>::max();

    std::uniform_int_distribution<int> action_dist(0, config_.action_space - 1);

    for (std::size_t sample = 0; sample < config_.num_samples; ++sample) {
        std::vector<Action> candidate_plan;
        candidate_plan.reserve(static_cast<std::size_t>(horizon));
        State sim_state = current;

        for (int t = 0; t < horizon; ++t) {
            Action a{action_dist(rng_)};
            candidate_plan.push_back(a);
            try {
                sim_state = world_model.predict_next(sim_state, a);
            } catch (...) {
                break;
            }
        }

        Real dist = state_distance(sim_state, goal);
        if (dist < best_dist) {
            best_dist = dist;
            best_plan = std::move(candidate_plan);
        }
    }

    return best_plan;
}

Real Planner::state_distance(const State& a, const State& b) const {
    if (a.latent.empty() || b.latent.empty()) return 0.0;
    Tensor diff = a.latent - b.latent;
    return diff.l2_norm();
}

} // namespace uik::agent
