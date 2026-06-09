#include <gtest/gtest.h>
#include "agent/agent_kernel.hpp"
#include "agent/grid_world.hpp"
#include "common/program.hpp"

using namespace uik;
using namespace uik::agent;

// End-to-end test: kernel runs in GridWorld and exhibits learning behavior
TEST(Integration, gridworld_agent_completes_without_crash) {
    GridWorld::Config env_cfg;
    env_cfg.rows = 3;
    env_cfg.cols = 3;
    env_cfg.max_steps = 50;

    AgentKernel::Config cfg;
    cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2;
    cfg.wm_config.latent_dim   = 8;
    cfg.wm_config.action_space = 6;
    cfg.planning_horizon       = 3;
    cfg.evolve_interval        = 5;
    cfg.induce_interval        = 8;
    cfg.adapt_interval         = 10;
    cfg.obs_buffer_size        = 6;
    cfg.planner_config.action_space = 6;
    cfg.enable_self_modification = true;

    AgentKernel kernel(cfg);
    GridWorld env(env_cfg);
    EXPECT_NO_THROW(kernel.run(env, 50));
    EXPECT_GT(kernel.step_count(), 0u);
}

TEST(Integration, self_modification_activates) {
    GridWorld::Config env_cfg;
    env_cfg.rows = 3;
    env_cfg.cols = 3;
    env_cfg.max_steps = 100;

    AgentKernel::Config cfg;
    cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2;
    cfg.wm_config.latent_dim   = 8;
    cfg.wm_config.action_space = 6;
    cfg.planning_horizon       = 3;
    cfg.evolve_interval        = 5;
    cfg.induce_interval        = 8;
    cfg.adapt_interval         = 10;
    cfg.obs_buffer_size        = 6;
    cfg.planner_config.action_space = 6;
    cfg.enable_self_modification = true;

    AgentKernel kernel(cfg);
    GridWorld env(env_cfg);
    kernel.run(env, 100);

    // Self-modification should have triggered at steps 10, 20, 30, ...
    EXPECT_GT(kernel.adapter().adaptations_count(), 0u);
}

TEST(Integration, logger_captures_events) {
    GridWorld::Config env_cfg;
    env_cfg.rows = 3;
    env_cfg.cols = 3;
    env_cfg.max_steps = 30;

    AgentKernel::Config cfg;
    cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2;
    cfg.wm_config.latent_dim   = 8;
    cfg.wm_config.action_space = 6;
    cfg.planning_horizon       = 2;
    cfg.evolve_interval        = 10;
    cfg.induce_interval        = 10;
    cfg.adapt_interval         = 10;
    cfg.obs_buffer_size        = 5;
    cfg.planner_config.action_space = 6;

    AgentKernel kernel(cfg);
    GridWorld env(env_cfg);
    kernel.run(env, 30);

    // Logger should have kernel_init, run_start, run_end at minimum
    EXPECT_GE(kernel.logger().count(), 3u);
}

TEST(Integration, logs_contain_adapted_params) {
    GridWorld::Config env_cfg;
    env_cfg.rows = 3;
    env_cfg.cols = 3;
    env_cfg.max_steps = 50;

    AgentKernel::Config cfg;
    cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2;
    cfg.wm_config.latent_dim   = 8;
    cfg.wm_config.action_space = 6;
    cfg.planning_horizon       = 2;
    cfg.evolve_interval        = 10;
    cfg.induce_interval        = 10;
    cfg.adapt_interval         = 10;
    cfg.obs_buffer_size        = 5;
    cfg.planner_config.action_space = 6;
    cfg.enable_self_modification = true;

    AgentKernel kernel(cfg);
    GridWorld env(env_cfg);
    kernel.run(env, 50);

    // Step logs should have LR and exploration bonus
    const auto& logs = kernel.logs();
    ASSERT_FALSE(logs.empty());
    for (const auto& log : logs) {
        EXPECT_GE(log.learning_rate, 0.0);
        EXPECT_GE(log.exploration_bonus, 0.0);
    }
}

TEST(Integration, program_serialization_roundtrip) {
    auto prog = compose(
        make_program(OpKind::FlipH),
        compose(
            make_program(OpKind::Rotate90),
            make_program(OpKind::MapColor, 2, 7)
        )
    );

    std::string serialized = serialize(prog);
    auto restored = deserialize(serialized);

    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(restored->kind, OpKind::Compose);
    EXPECT_EQ(restored->description_length(), prog->description_length());
    EXPECT_EQ(restored->depth(), prog->depth());
    EXPECT_EQ(restored->node_count(), prog->node_count());

    // Re-serialize should produce identical string
    EXPECT_EQ(serialize(restored), serialized);
}

TEST(Integration, disabled_self_modification) {
    GridWorld::Config env_cfg;
    env_cfg.rows = 3;
    env_cfg.cols = 3;
    env_cfg.max_steps = 50;

    AgentKernel::Config cfg;
    cfg.wm_config.input_dim    = env_cfg.rows * env_cfg.cols * 2;
    cfg.wm_config.latent_dim   = 8;
    cfg.wm_config.action_space = 6;
    cfg.planning_horizon       = 2;
    cfg.evolve_interval        = 10;
    cfg.induce_interval        = 10;
    cfg.adapt_interval         = 10;
    cfg.obs_buffer_size        = 5;
    cfg.planner_config.action_space = 6;
    cfg.enable_self_modification = false;

    AgentKernel kernel(cfg);
    GridWorld env(env_cfg);
    kernel.run(env, 50);

    EXPECT_EQ(kernel.adapter().adaptations_count(), 0u);
}
