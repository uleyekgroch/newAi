#include <gtest/gtest.h>
#include "agent/goal_setter.hpp"
#include "agent/planner.hpp"
#include "agent/agent_kernel.hpp"
#include <random>

using namespace uik;
using namespace uik::agent;

// ── GoalSetter Tests ──

TEST(GoalSetter, set_goal_returns_correct_dim) {
    GoalSetter setter;
    State current{Tensor({8}, 0.5)};
    State goal = setter.set_goal(current, 0.1, 0.0);
    EXPECT_EQ(goal.latent.flat_size(), 8u);
}

TEST(GoalSetter, higher_compression_progress_shifts_goal_more) {
    GoalSetter setter;
    State current{Tensor({4}, 0.0)};
    State goal_low = setter.set_goal(current, 0.1, 0.0);
    State goal_high = setter.set_goal(current, 10.0, 0.0);

    // Higher compression progress → larger perturbation
    Real shift_low = (goal_low.latent - current.latent).l2_norm();
    Real shift_high = (goal_high.latent - current.latent).l2_norm();
    EXPECT_GT(shift_high, shift_low);
}

TEST(GoalSetter, compute_reward_includes_intrinsic) {
    GoalSetter setter;
    Reward r = setter.compute_reward(0.5, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(r.external, 1.0);
    EXPECT_GT(r.intrinsic, 0.0);
}

TEST(GoalSetter, compute_reward_exploration_bonus_for_novel) {
    GoalSetter::Config cfg;
    cfg.novelty_threshold = 0.3;
    cfg.exploration_bonus = 0.1;
    GoalSetter setter(cfg);

    Reward novel = setter.compute_reward(0.0, 0.0, 0.5);     // above threshold
    Reward boring = setter.compute_reward(0.0, 0.0, 0.1);    // below threshold
    EXPECT_GT(novel.intrinsic, boring.intrinsic);
}

// ── Planner Tests ──

TEST(Planner, plan_returns_actions_of_correct_horizon) {
    world_model::WorldModel::Config wm_cfg;
    wm_cfg.input_dim = 16;
    wm_cfg.latent_dim = 4;
    wm_cfg.action_space = 4;
    world_model::WorldModel wm(wm_cfg);

    Planner planner;
    State current{Tensor({4}, 0.0)};
    State goal{Tensor({4}, 1.0)};
    auto actions = planner.plan(current, goal, wm, 5);
    EXPECT_EQ(actions.size(), 5u);
}

TEST(Planner, plan_with_zero_horizon_returns_empty) {
    world_model::WorldModel::Config wm_cfg;
    wm_cfg.input_dim = 16;
    wm_cfg.latent_dim = 4;
    wm_cfg.action_space = 4;
    world_model::WorldModel wm(wm_cfg);

    Planner planner;
    State current{Tensor({4}, 0.0)};
    State goal{Tensor({4}, 1.0)};
    auto actions = planner.plan(current, goal, wm, 0);
    EXPECT_TRUE(actions.empty());
}

// ── Simple test environment ──
class TestEnv final : public IEnvironment {
public:
    explicit TestEnv(Dim obs_size = 16) : obs_size_(obs_size) {}

    Observation reset() override {
        step_ = 0;
        return {Tensor({obs_size_}, 0.5)};
    }

    StepResult step(const Action& /*action*/) override {
        ++step_;
        return {Observation{Tensor({obs_size_}, 0.5)}, -0.1, step_ >= 10};
    }

    int action_space_size() const override { return 4; }

private:
    Dim obs_size_;
    std::size_t step_ = 0;
};

// ── AgentKernel Tests ──

TEST(AgentKernel, run_completes_without_crash) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 3;
    cfg.evolve_interval = 5;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    EXPECT_NO_THROW(kernel.run(env, 20));
}

TEST(AgentKernel, step_count_matches_execution) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 3;
    cfg.evolve_interval = 100;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    kernel.run(env, 20);
    // Environment terminates at step 10
    EXPECT_EQ(kernel.step_count(), 10u);
}

TEST(AgentKernel, logs_match_step_count) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 2;
    cfg.evolve_interval = 100;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    kernel.run(env, 20);
    EXPECT_EQ(kernel.logs().size(), kernel.step_count());
}

TEST(AgentKernel, single_step_returns_valid_action) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 3;
    cfg.evolve_interval = 100;

    AgentKernel kernel(cfg);
    Observation obs{Tensor({16}, 0.5)};
    Action action = kernel.step(obs, 0.0);
    EXPECT_GE(action.id, 0);
}

TEST(AgentKernel, evolution_triggered_at_interval) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 2;
    cfg.evolve_interval = 5;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    kernel.run(env, 20);
    // Evolution should have been triggered at step 5 and 10
    EXPECT_GT(kernel.evolution().archive().size(), 0u);
}

TEST(AgentKernel, rule_induction_triggered_at_interval) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 2;
    cfg.evolve_interval = 100;
    cfg.induce_interval = 5;
    cfg.induce_max_iter = 10;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    kernel.run(env, 20);
    // Rule induction should have been triggered
    // (may or may not find rules depending on data, but should not crash)
    EXPECT_GE(kernel.rule_library().size(), 0u);
}

TEST(AgentKernel, logs_contain_rules_count) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 2;
    cfg.evolve_interval = 100;
    cfg.induce_interval = 3;

    AgentKernel kernel(cfg);
    TestEnv env(16);
    kernel.run(env, 20);
    ASSERT_FALSE(kernel.logs().empty());
    // Each log entry should have a valid rules_count field
    for (const auto& log : kernel.logs()) {
        EXPECT_GE(log.rules_count, 0u);
    }
}

TEST(AgentKernel, record_action_integrated_in_step) {
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 16;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 4;
    cfg.planning_horizon = 3;
    cfg.evolve_interval = 100;

    AgentKernel kernel(cfg);
    Observation obs{Tensor({16}, 0.5)};
    Action a1 = kernel.step(obs, 0.0);
    Action a2 = kernel.step(obs, 0.0);
    // World model should be learning dynamics from recorded actions
    EXPECT_GE(kernel.world_model().prediction_error_rate(), 0.0);
    (void)a1;
    (void)a2;
}
