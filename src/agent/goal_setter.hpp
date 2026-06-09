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
    };

    GoalSetter();
    explicit GoalSetter(Config config);

    State set_goal(const State& current,
                   Real compression_progress,
                   Real external_reward) override;

    [[nodiscard]] Reward compute_reward(Real compression_progress,
                                         Real external_reward,
                                         Real novelty) const;

private:
    Config config_;
};

} // namespace uik::agent
