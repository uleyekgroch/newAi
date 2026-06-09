#pragma once

#include "common/interfaces.hpp"
#include "common/logger.hpp"
#include "world_model/world_model.hpp"
#include "symbolic_descent/search_engine.hpp"
#include "symbolic_descent/rule_library.hpp"
#include "meta_evolution/evolutionary_selector.hpp"
#include "meta_evolution/parameter_adapter.hpp"
#include "meta_evolution/self_modifier.hpp"
#include "symbolic_descent/dsl.hpp"
#include "agent/goal_setter.hpp"
#include "agent/planner.hpp"
#include <vector>
#include <functional>
#include <deque>

namespace uik::agent {

// The Universal Intelligence Kernel: orchestrates all layers.
// perceive → model → induce rules → set goal → plan → act → evolve → self-modify
class AgentKernel {
public:
    struct Config {
        world_model::WorldModel::Config wm_config;
        symbolic_descent::SearchEngine::Config search_config;
        meta_evolution::EvolutionarySelector::Config evo_config;
        meta_evolution::ParameterAdapter::Config adapter_config;
        meta_evolution::SelfModifier::Config self_mod_config;
        GoalSetter::Config goal_config;
        Planner::Config planner_config;
        Real novelty_threshold      = 0.3;
        std::size_t evolve_interval = 10;    // evolve every N steps
        std::size_t induce_interval = 20;    // induce rules every N steps
        std::size_t adapt_interval  = 25;    // self-modify every N steps
        std::size_t induce_max_iter = 50;    // max search iterations for induction
        std::size_t obs_buffer_size = 10;    // observation buffer for induction
        int planning_horizon        = 5;
        bool enable_self_modification = true;
    };

    explicit AgentKernel(Config config);

    void run(IEnvironment& env, std::size_t max_steps);
    Action step(const Observation& obs, Real external_reward);

    [[nodiscard]] std::size_t step_count() const { return step_count_; }
    [[nodiscard]] Real total_reward() const { return total_reward_; }
    [[nodiscard]] const world_model::WorldModel& world_model() const {
        return world_model_;
    }
    [[nodiscard]] const meta_evolution::EvolutionarySelector& evolution() const {
        return evolution_;
    }
    [[nodiscard]] const symbolic_descent::RuleLibrary& rule_library() const {
        return rule_library_;
    }
    [[nodiscard]] const meta_evolution::ParameterAdapter& adapter() const {
        return adapter_;
    }
    [[nodiscard]] const StructuredLogger& logger() const { return logger_; }
    [[nodiscard]] StructuredLogger& logger() { return logger_; }

    struct StepLog {
        std::size_t step       = 0;
        Real novelty           = 0.0;
        Real compression_prog  = 0.0;
        Real external_reward   = 0.0;
        Real intrinsic_reward  = 0.0;
        int action_id          = 0;
        std::size_t rules_count = 0;
        Real learning_rate     = 0.0;
        Real exploration_bonus = 0.0;
    };
    [[nodiscard]] const std::vector<StepLog>& logs() const { return logs_; }

private:
    Config config_;
    world_model::WorldModel world_model_;
    symbolic_descent::SearchEngine search_engine_;
    symbolic_descent::RuleLibrary rule_library_;
    meta_evolution::EvolutionarySelector evolution_;
    meta_evolution::ParameterAdapter adapter_;
    meta_evolution::SelfModifier self_modifier_;
    symbolic_descent::DSL dsl_;
    GoalSetter goal_setter_;
    Planner planner_;
    StructuredLogger logger_;

    std::size_t step_count_ = 0;
    Real total_reward_ = 0.0;
    State current_state_;
    std::vector<StepLog> logs_;
    std::deque<Tensor> obs_buffer_;

    void try_induce_rules();
    void try_evolve();
    void try_self_modify(Real novelty, Real comp_progress,
                          Real external_reward);
    void try_strategy_evolution(Real novelty, Real comp_progress,
                                Real external_reward);
    Action select_action_from_evolved(const State& state);
};

} // namespace uik::agent
