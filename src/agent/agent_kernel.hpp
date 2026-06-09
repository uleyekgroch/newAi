#pragma once

#include "common/interfaces.hpp"
#include "world_model/world_model.hpp"
#include "symbolic_descent/search_engine.hpp"
#include "meta_evolution/evolutionary_selector.hpp"
#include "agent/goal_setter.hpp"
#include "agent/planner.hpp"
#include <vector>
#include <functional>

namespace uik::agent {

// The Universal Intelligence Kernel: orchestrates all layers.
// perceive → model → induce rules → set goal → plan → act → evolve
class AgentKernel {
public:
    struct Config {
        world_model::WorldModel::Config wm_config;
        symbolic_descent::SearchEngine::Config search_config;
        meta_evolution::EvolutionarySelector::Config evo_config;
        GoalSetter::Config goal_config;
        Planner::Config planner_config;
        Real novelty_threshold = 0.3;
        std::size_t evolve_interval = 10;  // evolve every N steps
        int planning_horizon = 5;
    };

    explicit AgentKernel(Config config);

    // Run the agent for N steps in an environment
    void run(IEnvironment& env, std::size_t max_steps);

    // Single step
    Action step(const Observation& obs, Real external_reward);

    // Access internal state for inspection
    [[nodiscard]] std::size_t step_count() const { return step_count_; }
    [[nodiscard]] Real total_reward() const { return total_reward_; }
    [[nodiscard]] const world_model::WorldModel& world_model() const {
        return world_model_;
    }
    [[nodiscard]] const meta_evolution::EvolutionarySelector& evolution() const {
        return evolution_;
    }

    struct StepLog {
        std::size_t step       = 0;
        Real novelty           = 0.0;
        Real compression_prog  = 0.0;
        Real external_reward   = 0.0;
        Real intrinsic_reward  = 0.0;
        int action_id          = 0;
    };
    [[nodiscard]] const std::vector<StepLog>& logs() const { return logs_; }

private:
    Config config_;
    world_model::WorldModel world_model_;
    symbolic_descent::SearchEngine search_engine_;
    meta_evolution::EvolutionarySelector evolution_;
    GoalSetter goal_setter_;
    Planner planner_;

    std::size_t step_count_ = 0;
    Real total_reward_ = 0.0;
    State current_state_;
    std::vector<StepLog> logs_;
};

} // namespace uik::agent
