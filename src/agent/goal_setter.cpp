#include "agent/goal_setter.hpp"
#include <cmath>

namespace uik::agent {

GoalSetter::GoalSetter() : GoalSetter(Config{}) {}

GoalSetter::GoalSetter(Config config) : config_(config) {}

State GoalSetter::set_goal(const State& current,
                            Real compression_progress,
                            Real external_reward) {
    // Goal = current state biased toward direction of maximum expected reward.
    // The goal is a latent state that the planner will try to reach.
    // In absence of a learned goal generator, we perturb the current state
    // in the direction that maximizes combined intrinsic + external reward.
    Real total_drive = external_reward +
                        config_.curiosity_weight * compression_progress;
    Real scale = std::tanh(total_drive) * config_.exploration_bonus;

    // Perturb latent state: goal = current + scale * direction
    // Direction: uniform perturbation toward "more interesting" states
    std::vector<Real> goal_latent(current.latent.flat_size());
    for (Dim i = 0; i < current.latent.flat_size(); ++i) {
        goal_latent[i] = current.latent.at(i) + scale;
    }
    return State{Tensor(current.latent.shape(), std::move(goal_latent))};
}

Reward GoalSetter::compute_reward(Real compression_progress,
                                    Real external_reward,
                                    Real novelty) const {
    Real intrinsic = compression_progress;
    if (novelty > config_.novelty_threshold) {
        intrinsic += config_.exploration_bonus;
    }
    return Reward{external_reward, intrinsic};
}

} // namespace uik::agent
