#include <gtest/gtest.h>
#include "meta_evolution/parameter_adapter.hpp"

using namespace uik;
using namespace uik::meta_evolution;

TEST(ParameterAdapter, default_params) {
    ParameterAdapter adapter;
    auto params = adapter.current_params();
    EXPECT_DOUBLE_EQ(params.learning_rate, 0.01);
    EXPECT_DOUBLE_EQ(params.exploration_bonus, 0.1);
    EXPECT_DOUBLE_EQ(params.curiosity_weight, 0.5);
}

TEST(ParameterAdapter, no_adaptation_with_insufficient_history) {
    ParameterAdapter adapter;
    ParameterAdapter::PerformanceSnapshot snap{0.1, 0.05, 0.5, 0.3};
    auto p1 = adapter.adapt(snap);
    auto p2 = adapter.adapt(snap);
    EXPECT_DOUBLE_EQ(p1.learning_rate, 0.01);
    EXPECT_DOUBLE_EQ(p2.learning_rate, 0.01);
    EXPECT_EQ(adapter.adaptations_count(), 0u);
}

TEST(ParameterAdapter, adapts_after_sufficient_history) {
    ParameterAdapter adapter;
    for (int i = 0; i < 5; ++i) {
        adapter.adapt({-0.1 * static_cast<Real>(i), 0.01, 0.5, 0.5});
    }
    EXPECT_GT(adapter.adaptations_count(), 0u);
}

TEST(ParameterAdapter, learning_rate_clamped) {
    ParameterAdapter::Config cfg;
    cfg.min_learning_rate = 0.001;
    cfg.max_learning_rate = 0.1;
    cfg.window_size = 5;
    cfg.adaptation_rate = 0.5;  // aggressive
    ParameterAdapter adapter(cfg);

    // Feed increasing errors to push LR up
    for (int i = 0; i < 20; ++i) {
        adapter.adapt({-1.0, 0.0, 0.5, static_cast<Real>(i) * 0.1});
    }
    auto params = adapter.current_params();
    EXPECT_LE(params.learning_rate, cfg.max_learning_rate);
    EXPECT_GE(params.learning_rate, cfg.min_learning_rate);
}

TEST(ParameterAdapter, exploration_clamped) {
    ParameterAdapter::Config cfg;
    cfg.min_exploration = 0.01;
    cfg.max_exploration = 0.5;
    cfg.window_size = 5;
    cfg.adaptation_rate = 0.5;
    ParameterAdapter adapter(cfg);

    // Feed decreasing novelty to push exploration up
    for (int i = 0; i < 20; ++i) {
        adapter.adapt({-1.0, 0.0, 1.0 / (1.0 + static_cast<Real>(i)), 0.1});
    }
    auto params = adapter.current_params();
    EXPECT_LE(params.exploration_bonus, cfg.max_exploration);
    EXPECT_GE(params.exploration_bonus, cfg.min_exploration);
}

TEST(ParameterAdapter, reward_trend_computed) {
    ParameterAdapter adapter;
    for (int i = 0; i < 10; ++i) {
        adapter.adapt({static_cast<Real>(i) * 0.1, 0.01, 0.5, 0.1});
    }
    // Positive reward trend (increasing rewards)
    EXPECT_GT(adapter.reward_trend(), 0.0);
}

TEST(ParameterAdapter, compression_trend_computed) {
    ParameterAdapter adapter;
    for (int i = 0; i < 10; ++i) {
        adapter.adapt({0.0, static_cast<Real>(i) * 0.01, 0.5, 0.1});
    }
    EXPECT_GT(adapter.compression_trend(), 0.0);
}
