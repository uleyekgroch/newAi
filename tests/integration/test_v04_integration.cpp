#include <gtest/gtest.h>
#include "agent/agent_kernel.hpp"
#include "agent/arc_env.hpp"
#include "agent/physics_env.hpp"
#include "agent/curriculum.hpp"
#include "meta_evolution/self_modifier.hpp"
#include "meta_evolution/safety_guard.hpp"
#include "meta_evolution/interpreter.hpp"
#include "symbolic_descent/dsl.hpp"

using namespace uik;
using namespace uik::agent;
using namespace uik::meta_evolution;
using namespace uik::symbolic_descent;

// v0.4 integration: verify all Phase 3 components work together

TEST(V04Integration, DSLExtendedOpsComposeWithGridOps) {
    DSL dsl;
    // Add(1) -> FlipH
    auto prog = compose(make_program(OpKind::Add, 1),
                         make_program(OpKind::FlipH));
    Tensor input({4}, std::vector<Real>{1, 2, 3, 4});
    auto result = dsl.execute(prog, input);
    // Add: [2,3,4,5] -> FlipH: [5,4,3,2]
    EXPECT_DOUBLE_EQ(result.at(0), 5.0);
    EXPECT_DOUBLE_EQ(result.at(3), 2.0);
}

TEST(V04Integration, SafetyGuardProtectsInterpreter) {
    Interpreter::Config cfg;
    cfg.safety_config.max_program_depth = 5;
    Interpreter interp(cfg);

    // Build a very deep program
    auto deep = make_program(OpKind::Identity);
    for (int i = 0; i < 10; ++i) {
        deep = compose(deep, make_program(OpKind::Identity));
    }

    Tensor input({2}, std::vector<Real>{1, 2});
    auto result = interp.execute(deep, input);
    EXPECT_FALSE(result.halted_safely);
}

TEST(V04Integration, SelfModifierWithInterpreter) {
    SelfModifier modifier;
    Interpreter interp;

    auto eval_fn = [&interp](const ProgramPtr& prog) -> Real {
        Tensor input({4}, std::vector<Real>{1, 2, 3, 4});
        auto result = interp.execute_safe(prog, input);
        if (!result.halted_safely) return -1.0;
        // Fitness: how different from input
        Real diff = 0.0;
        for (Dim i = 0; i < input.flat_size(); ++i) {
            Real d = result.output.at(i) - input.at(i);
            diff += d * d;
        }
        return diff;
    };

    modifier.try_modify(SelfModifier::StrategyKind::FitnessWeighting, eval_fn);
    EXPECT_GE(modifier.strategy_archive().size(), 3u);
}

TEST(V04Integration, ArcEnvironmentWithDSL) {
    ArcEnvironment env;
    DSL dsl;
    auto obs = env.reset();

    // Use a DSL program on the observation
    auto prog = make_program(OpKind::Threshold, 3, 1);
    auto transformed = dsl.execute(prog, obs.data);
    EXPECT_EQ(transformed.flat_size(), obs.data.flat_size());
}

TEST(V04Integration, PhysicsEnvironmentRunsEpisode) {
    PhysicsEnvironment env;
    env.reset();

    Real total_reward = 0.0;
    bool done = false;
    for (int i = 0; i < 200 && !done; ++i) {
        auto result = env.step(Action{1}); // push right
        total_reward += result.reward;
        done = result.done;
    }
    // Should accumulate some reward
    EXPECT_TRUE(done);
}

TEST(V04Integration, CurriculumProgressesThroughStages) {
    CurriculumManager::Config cfg;
    cfg.stages = {
        {"grid_world_3x3", 50, -100.0, 1},
        {"grid_world_5x5", 100, -100.0, 1},
    };
    CurriculumManager manager(cfg);

    EXPECT_EQ(manager.current_stage(), 0u);
    manager.report_episode(0.0, 10); // threshold is -100, so any reward advances
    EXPECT_EQ(manager.current_stage(), 1u);
    manager.report_episode(0.0, 10);
    EXPECT_TRUE(manager.complete());
}

TEST(V04Integration, RepeatLoopWithSafetyBound) {
    Interpreter::Config cfg;
    cfg.max_execution_steps = 50;
    Interpreter interp(cfg);

    // Repeat(Add(1), 1000) should hit step limit
    auto prog = make_program(OpKind::Repeat, 1000, 0,
                              {make_program(OpKind::Add, 1)});
    Tensor input({2}, std::vector<Real>{0, 0});
    auto result = interp.execute(prog, input);
    EXPECT_FALSE(result.halted_safely);
}

TEST(V04Integration, StoreLoadAcrossCompose) {
    DSL dsl;
    // Store(0) -> Multiply(2) -> Load(0)
    // Should recall the original values, not the multiplied ones
    auto prog = compose(
        make_program(OpKind::Store, 0),
        compose(
            make_program(OpKind::Multiply, 2),
            make_program(OpKind::Load, 0)
        )
    );
    Tensor data({3}, std::vector<Real>{1, 2, 3});
    auto result = dsl.execute(prog, data);
    EXPECT_DOUBLE_EQ(result.at(0), 1.0);
    EXPECT_DOUBLE_EQ(result.at(1), 2.0);
    EXPECT_DOUBLE_EQ(result.at(2), 3.0);
}
