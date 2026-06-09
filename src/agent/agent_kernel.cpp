#include "agent/agent_kernel.hpp"
#include "symbolic_descent/dsl.hpp"

namespace uik::agent {

AgentKernel::AgentKernel(Config config)
    : config_(config)
    , world_model_(config.wm_config)
    , search_engine_(config.search_config)
    , rule_library_(100)
    , evolution_(config.evo_config)
    , goal_setter_(config.goal_config)
    , planner_(config.planner_config)
{}

void AgentKernel::run(IEnvironment& env, std::size_t max_steps) {
    Observation obs = env.reset();
    Real ext_reward = 0.0;

    for (std::size_t s = 0; s < max_steps; ++s) {
        Action action = step(obs, ext_reward);
        StepResult result = env.step(action);
        obs = result.observation;
        ext_reward = result.reward;

        if (result.done) break;
    }
}

Action AgentKernel::step(const Observation& obs, Real external_reward) {
    ++step_count_;

    // 1. Perceive: encode observation into latent state
    State state = world_model_.encode(obs);

    // 2. Detect novelty
    Real novelty = world_model_.compute_novelty(state);

    // 3. Update world model (learns dynamics from previous action)
    world_model_.update(obs);

    // 4. Buffer observations for rule induction
    obs_buffer_.push_back(obs.data);
    if (obs_buffer_.size() > config_.obs_buffer_size) {
        obs_buffer_.pop_front();
    }

    // 5. Compute compression progress (intrinsic motivation)
    Real comp_progress = world_model_.compression_progress();

    // 6. Compute reward
    Reward reward = goal_setter_.compute_reward(
        comp_progress, external_reward, novelty);
    total_reward_ += reward.total(config_.goal_config.curiosity_weight);

    // 7. Set goal based on intrinsic + external reward
    State goal = goal_setter_.set_goal(state, comp_progress, external_reward);

    // 8. Plan action sequence toward goal using world model
    auto plan = planner_.plan(state, goal, world_model_,
                               config_.planning_horizon);

    // 9. Rule induction: periodically search for patterns in observations
    if (step_count_ % config_.induce_interval == 0) {
        try_induce_rules();
    }

    // 10. Meta-evolution: periodically evolve program archive
    if (step_count_ % config_.evolve_interval == 0) {
        try_evolve();
    }

    // 11. Select action: prefer plan, fallback to evolved program insight
    Action action{0};
    if (!plan.empty()) {
        action = plan.front();
    } else {
        action = select_action_from_evolved(state);
    }

    // Record action for world model dynamics learning
    world_model_.record_action(action);

    // Log
    current_state_ = state;
    logs_.push_back({step_count_, novelty, comp_progress,
                     external_reward, reward.intrinsic, action.id,
                     rule_library_.size()});

    return action;
}

void AgentKernel::try_induce_rules() {
    if (obs_buffer_.size() < 2) return;

    // Build dataset: consecutive observation pairs
    ISearchEngine::Dataset data;
    for (std::size_t i = 0; i + 1 < obs_buffer_.size(); ++i) {
        data.emplace_back(obs_buffer_[i], obs_buffer_[i + 1]);
    }

    auto result = search_engine_.search(data, config_.induce_max_iter);
    if (result.has_value()) {
        Real score = search_engine_.last_best_score();
        rule_library_.add_rule("transition", *result, score);

        // Seed evolution with discovered rules
        auto programs = rule_library_.extract_programs();
        if (!programs.empty()) {
            auto fitness = [&data, this](const ProgramPtr& prog) -> Real {
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
    // Fitness = MDL score on recent observations
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

    // Feed best evolved programs back into rule library
    auto best = evolution_.archive().best();
    if (best.has_value() && best->fitness > -100.0) {
        rule_library_.add_rule("evolved", best->program, best->fitness);
    }
}

Action AgentKernel::select_action_from_evolved(const State& state) {
    // Use best rule to decide action: try each action in world model,
    // pick the one whose predicted next state is most "interesting"
    // (highest novelty × compression progress)
    Real best_score = -std::numeric_limits<Real>::max();
    int best_action = 0;

    int n_actions = config_.planner_config.action_space;
    for (int a = 0; a < n_actions; ++a) {
        try {
            State next = world_model_.predict_next(state, Action{a});
            Real nov = world_model_.compute_novelty(next);
            Real score = nov;
            if (score > best_score) {
                best_score = score;
                best_action = a;
            }
        } catch (...) {
            continue;
        }
    }
    return Action{best_action};
}

} // namespace uik::agent
