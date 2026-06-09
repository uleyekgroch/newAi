#include "agent/agent_kernel.hpp"
#include "symbolic_descent/dsl.hpp"
#include <sstream>

namespace uik::agent {

AgentKernel::AgentKernel(Config config)
    : config_(config)
    , world_model_(config.wm_config)
    , search_engine_(config.search_config)
    , rule_library_(100)
    , evolution_(config.evo_config)
    , adapter_(config.adapter_config)
    , goal_setter_(config.goal_config)
    , planner_(config.planner_config)
{
    logger_.info("kernel_init", {
        {"latent_dim", std::to_string(config.wm_config.latent_dim)},
        {"action_space", std::to_string(config.wm_config.action_space)},
        {"self_modify", config.enable_self_modification ? "true" : "false"}
    });
}

void AgentKernel::run(IEnvironment& env, std::size_t max_steps) {
    Observation obs = env.reset();
    Real ext_reward = 0.0;

    logger_.info("run_start", {{"max_steps", std::to_string(max_steps)}});

    for (std::size_t s = 0; s < max_steps; ++s) {
        Action action = step(obs, ext_reward);
        StepResult result = env.step(action);
        obs = result.observation;
        ext_reward = result.reward;

        if (result.done) {
            logger_.info("env_done", {{"step", std::to_string(s + 1)}});
            break;
        }
    }

    logger_.info("run_end", {
        {"steps", std::to_string(step_count_)},
        {"total_reward", std::to_string(total_reward_)},
        {"rules", std::to_string(rule_library_.size())},
        {"adaptations", std::to_string(adapter_.adaptations_count())}
    });
}

Action AgentKernel::step(const Observation& obs, Real external_reward) {
    ++step_count_;

    // 1. Perceive
    State state = world_model_.encode(obs);

    // 2. Detect novelty
    Real novelty = world_model_.compute_novelty(state);

    // 3. Update world model
    world_model_.update(obs);

    // 4. Buffer observations
    obs_buffer_.push_back(obs.data);
    if (obs_buffer_.size() > config_.obs_buffer_size) {
        obs_buffer_.pop_front();
    }

    // 5. Compression progress
    Real comp_progress = world_model_.compression_progress();

    // 6. Compute reward (use possibly-adapted curiosity weight)
    auto adapted = adapter_.current_params();
    Reward reward = goal_setter_.compute_reward(
        comp_progress, external_reward, novelty);
    total_reward_ += reward.total(adapted.curiosity_weight);

    // 7. Set goal
    State goal = goal_setter_.set_goal(state, comp_progress, external_reward);

    // 8. Plan
    auto plan = planner_.plan(state, goal, world_model_,
                               config_.planning_horizon);

    // 9. Rule induction
    if (step_count_ % config_.induce_interval == 0) {
        try_induce_rules();
    }

    // 10. Meta-evolution
    if (step_count_ % config_.evolve_interval == 0) {
        try_evolve();
    }

    // 11. Self-modification: adapt parameters based on performance
    if (config_.enable_self_modification &&
        step_count_ % config_.adapt_interval == 0) {
        try_self_modify(novelty, comp_progress, external_reward);
    }

    // 12. Select action
    Action action{0};
    if (!plan.empty()) {
        action = plan.front();
    } else {
        action = select_action_from_evolved(state);
    }

    world_model_.record_action(action);

    // Log
    current_state_ = state;
    logs_.push_back({step_count_, novelty, comp_progress,
                     external_reward, reward.intrinsic, action.id,
                     rule_library_.size(),
                     adapted.learning_rate, adapted.exploration_bonus});

    return action;
}

void AgentKernel::try_induce_rules() {
    if (obs_buffer_.size() < 2) return;

    ISearchEngine::Dataset data;
    for (std::size_t i = 0; i + 1 < obs_buffer_.size(); ++i) {
        data.emplace_back(obs_buffer_[i], obs_buffer_[i + 1]);
    }

    auto result = search_engine_.search(data, config_.induce_max_iter);
    if (result.has_value()) {
        Real score = search_engine_.last_best_score();
        rule_library_.add_rule("transition", *result, score);

        logger_.info("rule_induced", {
            {"score", std::to_string(score)},
            {"total_rules", std::to_string(rule_library_.size())}
        });

        auto programs = rule_library_.extract_programs();
        if (!programs.empty()) {
            auto fitness = [&data](const ProgramPtr& prog) -> Real {
                symbolic_descent::DSL dsl;
                Real total = 0.0;
                for (const auto& [input, expected] : data) {
                    try {
                        Tensor output = dsl.execute(prog, input);
                        Tensor diff = output - expected;
                        total -= diff.l2_norm();
                    } catch (...) {
                        total -= 1000.0;
                    }
                }
                total -= static_cast<Real>(prog->description_length());
                return total;
            };
            evolution_.seed(programs, fitness);
        }
    }
}

void AgentKernel::try_evolve() {
    ISearchEngine::Dataset data;
    for (std::size_t i = 0; i + 1 < obs_buffer_.size(); ++i) {
        data.emplace_back(obs_buffer_[i], obs_buffer_[i + 1]);
    }

    auto fitness = [&data](const ProgramPtr& prog) -> Real {
        if (!prog) return -1e9;
        symbolic_descent::DSL dsl;
        Real total = 0.0;
        for (const auto& [input, expected] : data) {
            try {
                Tensor output = dsl.execute(prog, input);
                Tensor diff = output - expected;
                total -= diff.l2_norm();
            } catch (...) {
                total -= 1000.0;
            }
        }
        total -= static_cast<Real>(prog->description_length());
        return total;
    };

    evolution_.evolve_once(fitness);

    auto best = evolution_.archive().best();
    if (best.has_value() && best->fitness > -100.0) {
        rule_library_.add_rule("evolved", best->program, best->fitness);
    }
}

void AgentKernel::try_self_modify(Real novelty, Real comp_progress,
                                    Real external_reward) {
    meta_evolution::ParameterAdapter::PerformanceSnapshot snap;
    snap.reward = external_reward;
    snap.compression_prog = comp_progress;
    snap.novelty = novelty;
    snap.prediction_error = world_model_.prediction_error_rate();

    auto new_params = adapter_.adapt(snap);

    // Apply adapted parameters to subsystems
    config_.goal_config.curiosity_weight = new_params.curiosity_weight;
    config_.goal_config.exploration_bonus = new_params.exploration_bonus;

    // Apply adapted learning rate to world model
    world_model_.set_learning_rate(new_params.learning_rate);

    // Recreate goal setter with adapted config
    goal_setter_ = GoalSetter(config_.goal_config);

    logger_.info("self_modify", {
        {"lr", std::to_string(new_params.learning_rate)},
        {"explore", std::to_string(new_params.exploration_bonus)},
        {"curiosity", std::to_string(new_params.curiosity_weight)},
        {"adaptations", std::to_string(adapter_.adaptations_count())}
    });
}

Action AgentKernel::select_action_from_evolved(const State& state) {
    Real best_score = -std::numeric_limits<Real>::max();
    int best_action = 0;

    int n_actions = config_.planner_config.action_space;
    for (int a = 0; a < n_actions; ++a) {
        try {
            State next = world_model_.predict_next(state, Action{a});
            Real nov = world_model_.compute_novelty(next);
            if (nov > best_score) {
                best_score = nov;
                best_action = a;
            }
        } catch (...) {
            continue;
        }
    }
    return Action{best_action};
}

} // namespace uik::agent
