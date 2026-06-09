#include "agent/goal_setter.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace uik::agent {

GoalSetter::GoalSetter() : GoalSetter(Config{}) {}

GoalSetter::GoalSetter(Config config) : config_(config) {}

State GoalSetter::set_goal(const State& current,
                            Real compression_progress,
                            Real external_reward) {
    // Track compression history for learning progress
    compression_history_.push_back(compression_progress);
    if (compression_history_.size() > config_.progress_window) {
        compression_history_.pop_front();
    }

    // Autonomous goal discovery:
    // Instead of fixed scale, discover which latent dimensions to push
    // by tracking which directions correlate with learning progress
    Dim latent_size = current.latent.flat_size();
    if (goal_direction_scores_.empty()) {
        goal_direction_scores_.resize(latent_size, 0.0);
    }

    // Update goal direction scores based on learning progress
    Real lp = learning_progress();
    if (lp > 0.0 && goal_direction_count_ > 0 && latent_size > 0) {
        // Reinforce the current state direction proportional to progress
        for (Dim i = 0; i < latent_size && i < goal_direction_scores_.size(); ++i) {
            Real state_val = current.latent.at(i);
            goal_direction_scores_[i] =
                0.95 * goal_direction_scores_[i] + 0.05 * lp * state_val;
        }
    }
    ++goal_direction_count_;

    // Apply time discount
    Real discount = 1.0;
    if (!step_rewards_.empty()) {
        discount = std::pow(config_.gamma, static_cast<Real>(step_rewards_.size()));
    }

    // Combine: external drive + curiosity drive + learning progress
    Real total_drive = external_reward +
                        config_.curiosity_weight * compression_progress +
                        config_.competence_weight * lp;
    Real base_scale = std::tanh(total_drive * discount) * config_.exploration_bonus;

    std::vector<Real> goal_latent(latent_size);
    for (Dim i = 0; i < latent_size; ++i) {
        Real direction_bias = 0.0;
        if (i < goal_direction_scores_.size() && goal_direction_count_ > 5) {
            // Use discovered goal directions
            direction_bias = 0.3 * std::tanh(goal_direction_scores_[i]);
        }
        goal_latent[i] = current.latent.at(i) + base_scale + direction_bias;
    }
    return State{Tensor(current.latent.shape(), std::move(goal_latent))};
}

Reward GoalSetter::compute_reward(Real compression_progress,
                                    Real external_reward,
                                    Real novelty) const {
    // Update novelty statistics (Welford's online algorithm)
    ++novelty_count_;
    Real delta = novelty - running_novelty_mean_;
    running_novelty_mean_ += delta / static_cast<Real>(novelty_count_);
    Real delta2 = novelty - running_novelty_mean_;
    running_novelty_var_ += delta * delta2;

    // Information-gain based exploration (not simple threshold)
    Real info_reward = information_gain_reward(novelty, compression_progress);

    // Learning progress signal
    compression_history_.push_back(compression_progress);
    if (compression_history_.size() > config_.progress_window) {
        compression_history_.pop_front();
    }
    Real lp = learning_progress();

    // Composite intrinsic reward:
    // R_intrinsic = α₁·compression_progress + α₂·info_gain + α₃·learning_progress
    Real intrinsic = config_.curiosity_weight * compression_progress
                   + config_.info_gain_weight * info_reward
                   + config_.competence_weight * lp;

    return Reward{external_reward, intrinsic};
}

Real GoalSetter::information_gain_reward(Real novelty,
                                           Real compression_progress) const {
    // Information gain ≈ how much this observation would reduce uncertainty
    // Proxy: normalized novelty × compression progress improvement
    // High novelty + high compression progress = high information gain
    // (states that are both novel AND learnable)
    Real normalized_novelty = 0.0;
    if (novelty_count_ > 1) {
        Real variance = running_novelty_var_ / static_cast<Real>(novelty_count_ - 1);
        Real std_dev = std::sqrt(std::max(variance, 1e-10));
        // Z-score: how many std devs above mean
        normalized_novelty = (novelty - running_novelty_mean_) / std_dev;
    } else {
        normalized_novelty = novelty;
    }

    // Learnability gate: very high novelty with zero compression = noise, not info
    // Sweet spot: moderate novelty + positive compression
    Real learnability = std::max(0.0, compression_progress);
    Real novelty_signal = std::max(0.0, normalized_novelty);

    // Information gain = novelty * learnability (states that are both novel AND learnable)
    Real info_gain = novelty_signal * (1.0 + learnability);

    // Also reward moderate novelty (zone of proximal development)
    // Peak reward at ~1 standard deviation above mean
    Real zpd_bonus = std::exp(-0.5 * (normalized_novelty - 1.0) * (normalized_novelty - 1.0));

    return info_gain + 0.5 * zpd_bonus;
}

Real GoalSetter::learning_progress() const {
    if (compression_history_.size() < 2) return 0.0;

    // Learning progress = improvement in compression over recent window
    // Compare first half vs second half of window
    std::size_t mid = compression_history_.size() / 2;
    Real first_half = 0.0, second_half = 0.0;
    std::size_t count1 = 0, count2 = 0;

    for (std::size_t i = 0; i < compression_history_.size(); ++i) {
        if (i < mid) {
            first_half += compression_history_[i];
            ++count1;
        } else {
            second_half += compression_history_[i];
            ++count2;
        }
    }

    if (count1 == 0 || count2 == 0) return 0.0;
    Real avg1 = first_half / static_cast<Real>(count1);
    Real avg2 = second_half / static_cast<Real>(count2);

    // Positive = getting better at compressing (learning)
    return std::max(0.0, avg2 - avg1);
}

Real GoalSetter::discounted_return() const {
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
