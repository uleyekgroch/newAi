#include <gtest/gtest.h>
#include "agent/grid_world.hpp"
#include <cmath>

using namespace uik;
using namespace uik::agent;

TEST(GridWorld, reset_returns_observation_of_correct_size) {
    GridWorld::Config cfg;
    cfg.rows = 3;
    cfg.cols = 4;
    GridWorld env(cfg);
    auto obs = env.reset();
    // observation = grid (3*4) + target (3*4) = 24
    EXPECT_EQ(obs.data.flat_size(), 24u);
}

TEST(GridWorld, action_space_is_six) {
    GridWorld env;
    EXPECT_EQ(env.action_space_size(), 6);
}

TEST(GridWorld, noop_does_not_change_grid) {
    GridWorld env;
    env.reset();
    auto grid_before = env.grid();
    env.step(Action{GridWorld::kNoop});
    EXPECT_EQ(env.grid(), grid_before);
}

TEST(GridWorld, flipH_reverses_rows) {
    GridWorld::Config cfg;
    cfg.rows = 2;
    cfg.cols = 3;
    cfg.num_colors = 100;
    GridWorld env(cfg);
    env.reset();

    auto before = env.grid();
    env.step(Action{GridWorld::kFlipH});
    auto after = env.grid();

    // Each row should be reversed
    for (Dim r = 0; r < 2; ++r) {
        for (Dim c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(after[r * 3 + c], before[r * 3 + (2 - c)]);
        }
    }
}

TEST(GridWorld, flipV_reverses_cols) {
    GridWorld::Config cfg;
    cfg.rows = 3;
    cfg.cols = 2;
    cfg.num_colors = 100;
    GridWorld env(cfg);
    env.reset();

    auto before = env.grid();
    env.step(Action{GridWorld::kFlipV});
    auto after = env.grid();

    // Rows should be vertically flipped
    for (Dim r = 0; r < 3; ++r) {
        for (Dim c = 0; c < 2; ++c) {
            EXPECT_DOUBLE_EQ(after[r * 2 + c], before[(2 - r) * 2 + c]);
        }
    }
}

TEST(GridWorld, increment_adds_one_mod_colors) {
    GridWorld::Config cfg;
    cfg.rows = 2;
    cfg.cols = 2;
    cfg.num_colors = 10;
    GridWorld env(cfg);
    env.reset();

    auto before = env.grid();
    env.step(Action{GridWorld::kIncrement});
    auto after = env.grid();

    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_DOUBLE_EQ(after[i], std::fmod(before[i] + 1.0, 10.0));
    }
}

TEST(GridWorld, shift_right_circular) {
    GridWorld::Config cfg;
    cfg.rows = 1;
    cfg.cols = 4;
    cfg.num_colors = 100;
    GridWorld env(cfg);
    env.reset();

    auto before = env.grid();
    env.step(Action{GridWorld::kShiftRight});
    auto after = env.grid();

    // Last element wraps to first
    EXPECT_DOUBLE_EQ(after[0], before[3]);
    EXPECT_DOUBLE_EQ(after[1], before[0]);
    EXPECT_DOUBLE_EQ(after[2], before[1]);
    EXPECT_DOUBLE_EQ(after[3], before[2]);
}

TEST(GridWorld, terminates_at_max_steps) {
    GridWorld::Config cfg;
    cfg.max_steps = 5;
    GridWorld env(cfg);
    env.reset();

    for (int i = 0; i < 4; ++i) {
        auto result = env.step(Action{GridWorld::kNoop});
        EXPECT_FALSE(result.done) << "step " << (i + 1);
    }
    auto result = env.step(Action{GridWorld::kNoop});
    EXPECT_TRUE(result.done);
}

TEST(GridWorld, reward_is_negative) {
    GridWorld env;
    env.reset();
    auto result = env.step(Action{GridWorld::kNoop});
    EXPECT_LT(result.reward, 0.0);
}
