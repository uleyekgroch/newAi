#include <gtest/gtest.h>
#include "agent/arc_env.hpp"

using namespace uik;
using namespace uik::agent;

class ArcEnvTest : public ::testing::Test {
protected:
    ArcEnvironment env_;
};

TEST_F(ArcEnvTest, ResetReturnsValidObservation) {
    auto obs = env_.reset();
    EXPECT_GT(obs.data.flat_size(), 0u);
}

TEST_F(ArcEnvTest, ActionSpaceSizePositive) {
    EXPECT_GT(env_.action_space_size(), 0);
}

TEST_F(ArcEnvTest, StepReturnsResult) {
    env_.reset();
    auto result = env_.step(Action{0});
    EXPECT_GT(result.observation.data.flat_size(), 0u);
}

TEST_F(ArcEnvTest, SkipActionAdvancesPuzzle) {
    env_.reset();
    int skip_action = env_.action_space_size() - 1;
    auto result = env_.step(Action{skip_action});
    EXPECT_GT(result.observation.data.flat_size(), 0u);
}

TEST_F(ArcEnvTest, SubmitActionGivesReward) {
    env_.reset();
    int submit_action = env_.action_space_size() - 2;
    auto result = env_.step(Action{submit_action});
    // Reward should be between 0 and 1 for similarity
    EXPECT_GE(result.reward, 0.0);
    EXPECT_LE(result.reward, 1.0);
}

TEST_F(ArcEnvTest, ModifyCellAction) {
    env_.reset();
    // Modify first cell
    auto result = env_.step(Action{0});
    EXPECT_GT(result.observation.data.flat_size(), 0u);
}

TEST_F(ArcEnvTest, MultipleResets) {
    for (int i = 0; i < 5; ++i) {
        auto obs = env_.reset();
        EXPECT_GT(obs.data.flat_size(), 0u);
    }
}

TEST_F(ArcEnvTest, EpisodeTerminates) {
    env_.reset();
    bool done = false;
    for (int i = 0; i < 1000 && !done; ++i) {
        int skip = env_.action_space_size() - 1;
        auto result = env_.step(Action{skip});
        done = result.done;
    }
    EXPECT_TRUE(done);
}

TEST_F(ArcEnvTest, CustomConfig) {
    ArcEnvironment::Config cfg;
    cfg.grid_rows = 3;
    cfg.grid_cols = 3;
    cfg.num_colors = 5;
    cfg.puzzles_per_episode = 2;
    cfg.seed = 123;
    ArcEnvironment custom_env(cfg);
    auto obs = custom_env.reset();
    EXPECT_GT(obs.data.flat_size(), 0u);
}
