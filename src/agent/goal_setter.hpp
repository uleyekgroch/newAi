#pragma once

#include "common/interfaces.hpp"
#include <deque>

namespace uik::agent {

// Sets goals based on intrinsic motivation (compression progress)
// and external rewards. Implements Schmidhuber's curiosity drive
// with information-gain based exploration and autonomous goal discovery.
class GoalSetter final : public IGoalSetter {
public:
    struct Config {
        Real curiosity_weight      = 0.5;  // alpha: balance intrinsic vs external
        Real exploration_bonus      = 0.1;  // base bonus for novel states
        Real novelty_threshold      = 0.3;  // below this, prefer exploitation
        Real gamma                  = 0.99; // time discount factor γ
        Real info_gain_weight       = 0.3;  // weight for information gain component
        Real competence_weight      = 0.2;  // weight for learning progress
        std::size_t progress_window = 10;   // window for measuring learning progress
    };

    GoalSetter();
    explicit GoalSetter(Config config);

    State set_goal(const State& current,
                   Real compression_progress,
                   Real external_reward) override;

    [[nodiscard]] Reward compute_reward(Real compression_progress,
                                         Real external_reward,
                                         Real novelty) const;

    // Information-gain based exploration signal
    [[nodiscard]] Real information_gain_reward(Real novelty,
                                                Real compression_progress) const;

    // Learning progress: improvement in compression over recent window
    [[nodiscard]] Real learning_progress() const;

    // Discounted cumulative reward: J = Σ γ^(t-t0) * R(t)
    [[nodiscard]] Real discounted_return() const;
    void record_step_reward(Real reward);
    void reset_episode();
    [[nodiscard]] Real gamma() const { return config_.gamma; }
    [[nodiscard]] std::size_t episode_steps() const { return step_rewards_.size(); }

private:
    Config config_;
    std::vector<Real> step_rewards_;            // per-step rewards for discounting
    mutable std::deque<Real> compression_history_;  // recent compression progress values
    mutable Real running_novelty_mean_  = 0.0;  // online mean of novelty
    mutable Real running_novelty_var_   = 1.0;  // online variance of novelty
    mutable std::size_t novelty_count_  = 0;

    // Autonomous goal discovery: track which goal directions yield progress
    mutable std::vector<Real> goal_direction_scores_;
    mutable std::size_t goal_direction_count_ = 0;
};

} // namespace uik::agent
