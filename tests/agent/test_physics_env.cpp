#include <gtest/gtest.h>
#include "agent/physics_env.hpp"
#include <cmath>

using namespace uik;
using namespace uik::agent;

class PhysicsEnvTest : public ::testing::Test {
protected:
    PhysicsEnvironment env_;
};

TEST_F(PhysicsEnvTest, ResetReturnsValidObservation) {
    auto obs = env_.reset();
    EXPECT_EQ(obs.data.flat_size(), 64u);
}

TEST_F(PhysicsEnvTest, ActionSpaceSizeCorrect) {
    // 5 max objects * 4 directions + 2 global = 22
    EXPECT_EQ(env_.action_space_size(), 22);
}

TEST_F(PhysicsEnvTest, StepAdvancesSimulation) {
    env_.reset();
    EXPECT_EQ(env_.step_count(), 0u);
    env_.step(Action{0});
    EXPECT_EQ(env_.step_count(), 1u);
}

TEST_F(PhysicsEnvTest, ObjectsExistAfterReset) {
    env_.reset();
    EXPECT_GE(env_.objects().size(), 3u); // 2-3 movable + 1 ground
}

TEST_F(PhysicsEnvTest, GravityAffectsObjects) {
    env_.reset();
    Real initial_y = -1.0;
    for (const auto& obj : env_.objects()) {
        if (!obj.fixed) { initial_y = obj.y; break; }
    }
    // Step without action (noop)
    int noop = env_.action_space_size() - 2;
    for (int i = 0; i < 10; ++i) {
        env_.step(Action{noop});
    }
    Real final_y = -1.0;
    for (const auto& obj : env_.objects()) {
        if (!obj.fixed) { final_y = obj.y; break; }
    }
    // Object should have moved due to gravity
    EXPECT_NE(initial_y, final_y);
}

TEST_F(PhysicsEnvTest, PushAffectsVelocity) {
    env_.reset();
    // Push first object right (action = 0*4 + 1 = 1)
    env_.step(Action{1}); // push_right on obj 0
    bool has_velocity = false;
    for (const auto& obj : env_.objects()) {
        if (!obj.fixed && std::abs(obj.vx) > 0.01) {
            has_velocity = true;
            break;
        }
    }
    EXPECT_TRUE(has_velocity);
}

TEST_F(PhysicsEnvTest, EpisodeTerminatesAtMaxSteps) {
    env_.reset();
    bool done = false;
    for (int i = 0; i < 300 && !done; ++i) {
        auto result = env_.step(Action{0});
        done = result.done;
    }
    EXPECT_TRUE(done);
}

TEST_F(PhysicsEnvTest, RewardIsNegativeDistance) {
    env_.reset();
    auto result = env_.step(Action{0});
    // Reward should be negative (distance-based)
    EXPECT_LE(result.reward, 0.0);
}

TEST_F(PhysicsEnvTest, ObjectsStayInBounds) {
    env_.reset();
    for (int i = 0; i < 100; ++i) {
        env_.step(Action{1}); // push right repeatedly
    }
    for (const auto& obj : env_.objects()) {
        EXPECT_GE(obj.x, 0.0);
        EXPECT_LE(obj.x, 10.0);
        EXPECT_GE(obj.y, 0.0);
        EXPECT_LE(obj.y, 10.0);
    }
}

TEST_F(PhysicsEnvTest, DropObjectAction) {
    env_.reset();
    std::size_t initial_count = env_.objects().size();
    int drop_action = static_cast<int>(env_.objects().size()) * 4 + 1;
    env_.step(Action{drop_action});
    EXPECT_GE(env_.objects().size(), initial_count);
}

TEST_F(PhysicsEnvTest, CustomConfig) {
    PhysicsEnvironment::Config cfg;
    cfg.max_objects = 3;
    cfg.gravity = -0.5;
    cfg.max_steps = 50;
    cfg.seed = 99;
    PhysicsEnvironment custom(cfg);
    auto obs = custom.reset();
    EXPECT_EQ(obs.data.flat_size(), 64u);
}
