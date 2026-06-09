#include <gtest/gtest.h>
#include "agent/curriculum.hpp"

using namespace uik;
using namespace uik::agent;

class CurriculumTest : public ::testing::Test {
protected:
    CurriculumManager manager_;
};

TEST_F(CurriculumTest, StartsAtFirstStage) {
    EXPECT_EQ(manager_.current_stage(), 0u);
    EXPECT_FALSE(manager_.complete());
}

TEST_F(CurriculumTest, HasMultipleStages) {
    EXPECT_GE(manager_.total_stages(), 3u);
}

TEST_F(CurriculumTest, CurrentEnvIsValid) {
    auto& env = manager_.current_env();
    auto obs = env.reset();
    EXPECT_GT(obs.data.flat_size(), 0u);
}

TEST_F(CurriculumTest, AdvancesOnGoodPerformance) {
    std::size_t initial_stage = manager_.current_stage();
    // Report good episodes
    for (int i = 0; i < 10; ++i) {
        manager_.report_episode(10.0, 50);
    }
    EXPECT_GT(manager_.current_stage(), initial_stage);
}

TEST_F(CurriculumTest, DoesNotAdvanceOnBadPerformance) {
    std::size_t initial_stage = manager_.current_stage();
    for (int i = 0; i < 5; ++i) {
        manager_.report_episode(-10.0, 100);
    }
    EXPECT_EQ(manager_.current_stage(), initial_stage);
}

TEST_F(CurriculumTest, TracksAverageReward) {
    manager_.report_episode(1.0, 50);
    manager_.report_episode(3.0, 50);
    EXPECT_DOUBLE_EQ(manager_.average_reward(), 2.0);
}

TEST_F(CurriculumTest, TracksTotalEpisodes) {
    manager_.report_episode(1.0, 50);
    manager_.report_episode(2.0, 50);
    manager_.report_episode(3.0, 50);
    EXPECT_EQ(manager_.total_episodes(), 3u);
}

TEST_F(CurriculumTest, StageNameAvailable) {
    EXPECT_FALSE(manager_.stage_name().empty());
}

TEST_F(CurriculumTest, CustomConfig) {
    CurriculumManager::Config cfg;
    cfg.stages = {
        {"grid_world_3x3", 50, 0.1, 1},
        {"grid_world_5x5", 100, 0.1, 1},
    };
    cfg.seed = 99;
    CurriculumManager custom(cfg);
    EXPECT_EQ(custom.total_stages(), 2u);
    custom.report_episode(10.0, 10);
    EXPECT_EQ(custom.current_stage(), 1u);
}

TEST_F(CurriculumTest, CanReachCompletion) {
    // Force quick completion with custom config
    CurriculumManager::Config cfg;
    cfg.stages = {
        {"grid_world_3x3", 50, -100.0, 1},
    };
    CurriculumManager quick(cfg);
    quick.report_episode(-50.0, 10);
    EXPECT_TRUE(quick.complete());
}
