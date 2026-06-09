#pragma once

#include "common/interfaces.hpp"
#include <random>
#include <memory>
#include <cmath>

namespace uik::agent {

// Plans action sequences using the world model for look-ahead.
// Implements a simple random-shooting planner: sample N action sequences,
// simulate each through the world model, pick the one closest to the goal.
class Planner final : public IPlanner {
public:
    struct Config {
        std::size_t num_samples     = 64;  // random-shooting fallback trajectories
        int         action_space    = 4;   // number of discrete actions
        unsigned    seed            = 42;
        // MCTS parameters
        std::size_t mcts_simulations = 128;  // total MCTS simulations
        Real exploration_constant    = 1.4;  // UCB1 exploration constant
        bool use_mcts                = true;
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

    // MCTS tree node
    struct MCTSNode {
        Action action{0};
        std::size_t visits = 0;
        Real total_value = 0.0;
        std::vector<std::unique_ptr<MCTSNode>> children;
        MCTSNode* parent = nullptr;
        bool expanded = false;
    };

    std::vector<Action> mcts_plan(const State& current,
                                   const State& goal,
                                   IWorldModel& world_model,
                                   int horizon);
    Real mcts_simulate(MCTSNode* node, const State& state,
                        const State& goal, IWorldModel& world_model,
                        int depth, int max_depth);
    MCTSNode* mcts_select(MCTSNode* node) const;
    void mcts_expand(MCTSNode* node);
    Real ucb1(const MCTSNode* node) const;
};

} // namespace uik::agent
