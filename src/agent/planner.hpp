#pragma once

#include "common/interfaces.hpp"
#include <random>

namespace uik::agent {

// Plans action sequences using the world model for look-ahead.
// Implements a simple random-shooting planner: sample N action sequences,
// simulate each through the world model, pick the one closest to the goal.
class Planner final : public IPlanner {
public:
    struct Config {
        std::size_t num_samples     = 64;  // number of random trajectories
        int         action_space    = 4;   // number of discrete actions
        unsigned    seed            = 42;
    };

    Planner();
    explicit Planner(Config config);

    std::vector<Action> plan(const State& current,
                              const State& goal,
                              IWorldModel& world_model,
                              int horizon) override;

private:
    Config config_;
    mutable std::mt19937 rng_;

    Real state_distance(const State& a, const State& b) const;
};

} // namespace uik::agent
