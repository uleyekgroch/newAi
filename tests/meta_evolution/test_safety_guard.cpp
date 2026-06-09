#include <gtest/gtest.h>
#include "meta_evolution/safety_guard.hpp"

using namespace uik;
using namespace uik::meta_evolution;

class SafetyGuardTest : public ::testing::Test {
protected:
    SafetyGuard guard_;
};

TEST_F(SafetyGuardTest, AcceptsSimpleProgram) {
    auto prog = make_program(OpKind::Identity);
    auto result = guard_.validate_program(prog);
    EXPECT_TRUE(result.safe);
}

TEST_F(SafetyGuardTest, RejectsNullProgram) {
    auto result = guard_.validate_program(nullptr);
    EXPECT_FALSE(result.safe);
    EXPECT_EQ(result.reason, "null program");
}

TEST_F(SafetyGuardTest, AcceptsValidParams) {
    auto result = guard_.validate_params(0.01, 1.0, 0.5);
    EXPECT_TRUE(result.safe);
}

TEST_F(SafetyGuardTest, RejectsExcessiveLearningRate) {
    auto result = guard_.validate_params(5.0, 1.0, 0.5);
    EXPECT_FALSE(result.safe);
}

TEST_F(SafetyGuardTest, RejectsNegativeLearningRate) {
    auto result = guard_.validate_params(-0.1, 1.0, 0.5);
    EXPECT_FALSE(result.safe);
}

TEST_F(SafetyGuardTest, ClampsParams) {
    Real lr = 10.0, eb = -5.0, cw = 100.0;
    guard_.clamp_params(lr, eb, cw);
    EXPECT_LE(lr, 1.0);
    EXPECT_GE(eb, 0.0);
    EXPECT_LE(cw, 2.0);
}

TEST_F(SafetyGuardTest, RejectsTooDeepProgram) {
    SafetyGuard::Config cfg;
    cfg.max_program_depth = 3;
    SafetyGuard strict(cfg);

    // Build a 5-deep program
    auto p = make_program(OpKind::Identity);
    for (int i = 0; i < 5; ++i) {
        p = compose(p, make_program(OpKind::Identity));
    }
    auto result = strict.validate_program(p);
    EXPECT_FALSE(result.safe);
}

TEST_F(SafetyGuardTest, AcceptsModificationWithSufficientRewards) {
    std::vector<Real> rewards(20, 1.0);
    auto result = guard_.validate_modification(rewards);
    EXPECT_TRUE(result.safe);
}

TEST_F(SafetyGuardTest, RejectsModificationWithRegression) {
    std::vector<Real> rewards;
    for (int i = 0; i < 5; ++i) rewards.push_back(1.0);
    for (int i = 0; i < 5; ++i) rewards.push_back(-1.0);
    auto result = guard_.validate_modification(rewards);
    EXPECT_FALSE(result.safe);
}

TEST_F(SafetyGuardTest, TracksValidationsAndRejections) {
    (void)guard_.validate_program(nullptr);
    (void)guard_.validate_params(0.01, 1.0, 0.5);
    EXPECT_EQ(guard_.validations(), 2u);
    EXPECT_EQ(guard_.rejections(), 1u);
}

TEST_F(SafetyGuardTest, RecordsPerformance) {
    for (int i = 0; i < 10; ++i) {
        guard_.record_performance(1.0);
    }
    EXPECT_FALSE(guard_.should_rollback());
}

TEST_F(SafetyGuardTest, DetectsRollbackNeeded) {
    for (int i = 0; i < 20; ++i) {
        guard_.record_performance(1.0);
    }
    for (int i = 0; i < 10; ++i) {
        guard_.record_performance(-2.0);
    }
    EXPECT_TRUE(guard_.should_rollback());
}

TEST_F(SafetyGuardTest, RejectsUnboundedLoop) {
    auto prog = make_program(OpKind::Repeat, 999, 0,
                              {make_program(OpKind::Identity)});
    SafetyGuard::Config cfg;
    cfg.max_loop_iterations = 100;
    SafetyGuard strict(cfg);
    auto result = strict.validate_program(prog);
    EXPECT_FALSE(result.safe);
}

TEST_F(SafetyGuardTest, AcceptsBoundedLoop) {
    auto prog = make_program(OpKind::Repeat, 5, 0,
                              {make_program(OpKind::Identity)});
    auto result = guard_.validate_program(prog);
    EXPECT_TRUE(result.safe);
}
