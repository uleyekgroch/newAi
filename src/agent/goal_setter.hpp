#pragma once

#include "common/interfaces.hpp"

namespace uik::agent {

// Sets goals based on intrinsic motivation (compression progress)
// and external rewards. Implements Schmidhuber's curiosity drive.
class GoalSetter final : public IGoalSetter {
public:
    struct Config {
        Real curiosity_weight   = 0.5; // alpha: balance intrinsic vs external
        Real exploration_bonus  = 0.1; // bonus for novel states
        Real novelty_threshold  = 0.3; // below this, prefer exploitation
        Real gamma              = 0.99; // time discount factor γ
    };

    GoalSetter();
    explicit GoalSetter(Config config);

    State set_goal(const State& current,
                   Real compression_progress,
                   Real external_reward) override;

    [[nodiscard]] Reward compute_reward(Real compression_progress,
                                         Real external_reward,
                                         Real novelty) const;

    // Discounted cumulative reward: J = Σ γ^(t-t0) * R(t)
    [[nodiscard]] Real discounted_return() const;
    void record_step_reward(Real reward);
    void reset_episode();
    [[nodiscard]] Real gamma() const { return config_.gamma; }
    [[nodiscard]] std::size_t episode_steps() const { return step_rewards_.size(); }

private:
    Config config_;
    std::vector<Real> step_rewards_;  // per-step rewards for discounting
};

} // namespace uik::agent
