#include "agent/goal_setter.hpp"
#include <cmath>

namespace uik::agent {

GoalSetter::GoalSetter() : GoalSetter(Config{}) {}

GoalSetter::GoalSetter(Config config) : config_(config) {}

State GoalSetter::set_goal(const State& current,
                            Real compression_progress,
                            Real external_reward) {
    // Goal = current state biased toward direction of maximum expected reward.
    // Apply time discount: recent rewards matter more
    Real discount = 1.0;
    if (!step_rewards_.empty()) {
        // Decay influence of old rewards
        discount = std::pow(config_.gamma, static_cast<Real>(step_rewards_.size()));
    }
    Real total_drive = external_reward +
                        config_.curiosity_weight * compression_progress;
    Real scale = std::tanh(total_drive * discount) * config_.exploration_bonus;

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

Real GoalSetter::discounted_return() const {
    // J = Σ γ^(τ-t0) * R(τ)
    Real total = 0.0;
    Real gamma_power = 1.0;
    for (const auto& r : step_rewards_) {
        total += gamma_power * r;
        gamma_power *= config_.gamma;
    }
    return total;
}

void GoalSetter::record_step_reward(Real reward) {
    step_rewards_.push_back(reward);
}

void GoalSetter::reset_episode() {
    step_rewards_.clear();
}

} // namespace uik::agent
