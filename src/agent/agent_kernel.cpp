#include "agent/agent_kernel.hpp"

namespace uik::agent {

AgentKernel::AgentKernel(Config config)
    : config_(config)
    , world_model_(config.wm_config)
    , search_engine_(config.search_config)
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

    // 3. Update world model
    world_model_.update(obs);

    // 4. Compute compression progress (intrinsic motivation)
    Real comp_progress = world_model_.compression_progress();

    // 5. Compute reward
    Reward reward = goal_setter_.compute_reward(
        comp_progress, external_reward, novelty);
    total_reward_ += reward.total(config_.goal_config.curiosity_weight);

    // 6. Set goal based on intrinsic + external reward
    State goal = goal_setter_.set_goal(state, comp_progress, external_reward);

    // 7. Plan action sequence toward goal using world model
    auto plan = planner_.plan(state, goal, world_model_,
                               config_.planning_horizon);

    // 8. Meta-evolution: periodically evolve program archive
    if (step_count_ % config_.evolve_interval == 0) {
        auto fitness_fn = [this](const ProgramPtr& prog) -> Real {
            return -static_cast<Real>(prog->description_length());
        };
        evolution_.evolve_once(fitness_fn);
    }

    // 9. Select action (first action from plan, or action 0 if no plan)
    Action action{0};
    if (!plan.empty()) {
        action = plan.front();
    }

    // Log
    current_state_ = state;
    logs_.push_back({step_count_, novelty, comp_progress,
                     external_reward, reward.intrinsic, action.id});

    return action;
}

} // namespace uik::agent
