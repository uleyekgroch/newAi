#include <gtest/gtest.h>
#include "meta_evolution/interpreter.hpp"

using namespace uik;
using namespace uik::meta_evolution;

class InterpreterTest : public ::testing::Test {
protected:
    Interpreter interp_;
};

TEST_F(InterpreterTest, ExecuteIdentity) {
    auto prog = make_program(OpKind::Identity);
    Tensor input({4}, std::vector<Real>{1, 2, 3, 4});
    auto result = interp_.execute(prog, input);
    EXPECT_TRUE(result.halted_safely);
    EXPECT_EQ(result.output.flat_size(), 4u);
    EXPECT_DOUBLE_EQ(result.output.at(0), 1.0);
}

TEST_F(InterpreterTest, ExecuteSafeMode) {
    auto prog = make_program(OpKind::FlipH);
    Tensor input({4}, std::vector<Real>{1, 2, 3, 4});
    auto result = interp_.execute_safe(prog, input);
    EXPECT_TRUE(result.halted_safely);
    EXPECT_DOUBLE_EQ(result.output.at(0), 4.0);
}

TEST_F(InterpreterTest, RejectsNullProgram) {
    Tensor input({4}, std::vector<Real>{1, 2, 3, 4});
    auto result = interp_.execute(nullptr, input);
    EXPECT_FALSE(result.halted_safely);
}

TEST_F(InterpreterTest, RewriteProgram) {
    auto prog = compose(make_program(OpKind::Identity),
                         make_program(OpKind::Identity));
    auto replacement = make_program(OpKind::FlipH);
    auto rewritten = interp_.rewrite(prog, {1}, replacement);
    EXPECT_NE(rewritten, nullptr);
    EXPECT_EQ(rewritten->children[1]->kind, OpKind::FlipH);
}

TEST_F(InterpreterTest, RewriteInvalidPath) {
    auto prog = make_program(OpKind::Identity);
    auto replacement = make_program(OpKind::FlipH);
    auto result = interp_.rewrite(prog, {5}, replacement);
    EXPECT_EQ(result->kind, OpKind::Identity); // unchanged
}

TEST_F(InterpreterTest, ExecuteCompose) {
    auto prog = compose(make_program(OpKind::Add, 1),
                         make_program(OpKind::Add, 2));
    Tensor input({3}, std::vector<Real>{0, 0, 0});
    auto result = interp_.execute(prog, input);
    EXPECT_TRUE(result.halted_safely);
    EXPECT_DOUBLE_EQ(result.output.at(0), 3.0);
}

TEST_F(InterpreterTest, ExecuteRepeat) {
    auto prog = make_program(OpKind::Repeat, 3, 0,
                              {make_program(OpKind::Add, 1)});
    Tensor input({2}, std::vector<Real>{0, 0});
    auto result = interp_.execute(prog, input);
    EXPECT_TRUE(result.halted_safely);
    EXPECT_DOUBLE_EQ(result.output.at(0), 3.0);
}

TEST_F(InterpreterTest, TracksStatistics) {
    auto prog = make_program(OpKind::Identity);
    Tensor input({2}, std::vector<Real>{1, 2});
    (void)interp_.execute(prog, input);
    (void)interp_.execute(prog, input);
    EXPECT_EQ(interp_.total_executions(), 2u);
}

TEST_F(InterpreterTest, StepLimitEnforced) {
    Interpreter::Config cfg;
    cfg.max_execution_steps = 5;
    Interpreter limited(cfg);

    // Repeat 100 times to exceed step limit
    auto prog = make_program(OpKind::Repeat, 100, 0,
                              {make_program(OpKind::Add, 1)});
    Tensor input({2}, std::vector<Real>{0, 0});
    auto result = limited.execute(prog, input);
    EXPECT_FALSE(result.halted_safely);
}
