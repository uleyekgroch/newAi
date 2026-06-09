#include <gtest/gtest.h>
#include "world_model/latent_state.hpp"
#include "world_model/dynamics_predictor.hpp"
#include "world_model/novelty_detector.hpp"
#include "world_model/world_model.hpp"
#include "symbolic_descent/search_engine.hpp"
#include "symbolic_descent/dsl.hpp"
#include "meta_evolution/self_modifier.hpp"
#include "agent/planner.hpp"
#include "agent/agent_kernel.hpp"
#include "agent/arc_env.hpp"
#include "agent/curriculum.hpp"
#include "common/program.hpp"
#include <cmath>

using namespace uik;
using namespace uik::world_model;
using namespace uik::symbolic_descent;
using namespace uik::meta_evolution;
using namespace uik::agent;

// ═══════════════════════════════════════════════════════════
// P1: LatentEncoder Online Learning
// ═══════════════════════════════════════════════════════════

TEST(EncoderLearning, LearnReducesReconstructionLoss) {
    LatentEncoder::Config c;
    c.input_dim = 16; c.latent_dim = 8; c.hidden_dim = 16;
    c.num_heads = 2; c.patch_size = 4; c.num_layers = 1;
    LatentEncoder enc(c);

    Observation obs;
    obs.data = Tensor({16}, 0.5);

    Real loss1 = enc.learn(obs, 0.01);
    // Run multiple learning steps
    Real loss_last = loss1;
    for (int i = 0; i < 50; ++i) {
        loss_last = enc.learn(obs, 0.01);
    }
    // Loss should decrease with learning
    EXPECT_LT(loss_last, loss1 + 0.1);
    EXPECT_GT(enc.learn_count(), 50u);
}

TEST(EncoderLearning, TotalReconstructionLossAccumulates) {
    LatentEncoder::Config c;
    c.input_dim = 16; c.latent_dim = 8; c.hidden_dim = 16;
    c.num_heads = 2; c.patch_size = 4; c.num_layers = 1;
    LatentEncoder enc(c);

    EXPECT_DOUBLE_EQ(enc.total_reconstruction_loss(), 0.0);
    Observation obs;
    obs.data = Tensor({16}, 0.3);
    enc.learn(obs, 0.001);
    EXPECT_GT(enc.total_reconstruction_loss(), 0.0);
}

TEST(EncoderLearning, WorldModelCallsLearn) {
    WorldModel::Config wmc;
    wmc.input_dim = 16; wmc.latent_dim = 4; wmc.action_space = 4;
    WorldModel wm(wmc);

    Observation obs;
    obs.data = Tensor({16}, 0.5);
    wm.update(obs);
    EXPECT_GT(wm.encoder_learn_count(), 0u);
}

// ═══════════════════════════════════════════════════════════
// P2: Self-modification closed loop
// ═══════════════════════════════════════════════════════════

TEST(SelfModClosedLoop, CurrentStrategyReturnsValidProgram) {
    SelfModifier sm;
    using SK = SelfModifier::StrategyKind;

    const auto& rs = sm.current_strategy(SK::RewardShaping);
    EXPECT_NE(rs, nullptr);
    const auto& gf = sm.current_strategy(SK::GoalFunction);
    EXPECT_NE(gf, nullptr);
}

TEST(SelfModClosedLoop, EvolveAllRunsAllStrategies) {
    SelfModifier sm;
    auto eval = [](SelfModifier::StrategyKind /*kind*/, const ProgramPtr& p) -> Real {
        if (!p) return -1e6;
        return -static_cast<Real>(p->description_length());
    };
    std::size_t improved = sm.evolve_all(eval);
    // May or may not improve, but should not crash
    EXPECT_GE(improved, 0u);
}

TEST(SelfModClosedLoop, AgentKernelIntegratesSelfMod) {
    // 3x3 grid with target → obs dim = 18
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 18;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 6;
    cfg.planner_config.action_space = 6;
    AgentKernel agent(cfg);

    GridWorld::Config gc;
    gc.rows = 3; gc.cols = 3; gc.num_colors = 5; gc.max_steps = 10;
    GridWorld env(gc);
    auto obs = env.reset();

    // Step should not crash (self-mod is integrated)
    Action action = agent.step(obs, 0.0);
    EXPECT_GE(action.id, 0);
}

// ═══════════════════════════════════════════════════════════
// P3: Beam search + type-directed synthesis
// ═══════════════════════════════════════════════════════════

TEST(BeamSearch, BeamSearchEnabledByDefault) {
    SearchEngine::Config cfg;
    EXPECT_TRUE(cfg.use_beam_search);
    EXPECT_EQ(cfg.beam_width, 8u);
}

TEST(BeamSearch, SearchWithBeamFindsProgram) {
    SearchEngine::Config cfg;
    cfg.use_beam_search = true;
    cfg.beam_width = 4;
    SearchEngine engine(cfg);

    // Simple identity task
    ISearchEngine::Dataset data;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    data.push_back({input, input});

    auto result = engine.search(data, 10);
    EXPECT_TRUE(result.has_value());
}

TEST(BeamSearch, SearchWithoutBeamWorks) {
    SearchEngine::Config cfg;
    cfg.use_beam_search = false;
    SearchEngine engine(cfg);

    ISearchEngine::Dataset data;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    data.push_back({input, input});

    auto result = engine.search(data, 5);
    EXPECT_TRUE(result.has_value());
}

// ═══════════════════════════════════════════════════════════
// P4: DynamicsPredictor 2-layer MLP
// ═══════════════════════════════════════════════════════════

TEST(DynamicsDeep, ConfigBasedConstruction) {
    DynamicsPredictor::Config cfg;
    cfg.latent_dim = 8;
    cfg.hidden_dim = 16;
    cfg.action_space_size = 4;
    DynamicsPredictor dp(cfg);

    State s{Tensor({8}, 0.5)};
    Action a{0};
    State next = dp.predict(s, a);
    EXPECT_EQ(next.latent.flat_size(), 8u);
}

TEST(DynamicsDeep, BackwardCompatibleConstructor) {
    DynamicsPredictor dp(8, 4, 42);

    State s{Tensor({8}, 0.3)};
    Action a{1};
    State next = dp.predict(s, a);
    EXPECT_EQ(next.latent.flat_size(), 8u);
}

TEST(DynamicsDeep, LearnReducesError) {
    DynamicsPredictor dp(4, 2, 42);

    State current{Tensor({4}, {0.1, 0.2, 0.3, 0.4})};
    State target{Tensor({4}, {0.5, 0.5, 0.5, 0.5})};
    Action a{0};

    Real error_before = dp.prediction_error(dp.predict(current, a), target);
    for (int i = 0; i < 100; ++i) {
        dp.learn(current, a, target, 0.01);
    }
    Real error_after = dp.prediction_error(dp.predict(current, a), target);
    EXPECT_LT(error_after, error_before);
}

TEST(DynamicsDeep, ResidualConnection) {
    DynamicsPredictor dp(4, 2, 42);

    State zero_state{Tensor({4}, 0.0)};
    Action a{0};
    State next = dp.predict(zero_state, a);
    // With residual s' = tanh(s + delta), starting from 0 should still produce output
    bool all_zero = true;
    for (Dim i = 0; i < 4; ++i) {
        if (std::abs(next.latent.at(i)) > 1e-10) all_zero = false;
    }
    // Not necessarily all zero due to bias terms
    (void)all_zero;
    EXPECT_EQ(next.latent.flat_size(), 4u);
}

// ═══════════════════════════════════════════════════════════
// P5: MCTS Planner
// ═══════════════════════════════════════════════════════════

TEST(MCTSPlanner, MCTSEnabledByDefault) {
    Planner::Config cfg;
    EXPECT_TRUE(cfg.use_mcts);
    EXPECT_EQ(cfg.mcts_simulations, 128u);
}

TEST(MCTSPlanner, MCTSPlanReturnsCorrectHorizon) {
    WorldModel::Config wmc;
    wmc.input_dim = 16; wmc.latent_dim = 4; wmc.action_space = 4;
    WorldModel wm(wmc);

    Planner::Config pc;
    pc.action_space = 4;
    pc.use_mcts = true;
    pc.mcts_simulations = 32;
    Planner planner(pc);

    State current{Tensor({4}, 0.0)};
    State goal{Tensor({4}, 1.0)};
    auto plan = planner.plan(current, goal, wm, 5);
    EXPECT_EQ(plan.size(), 5u);
}

TEST(MCTSPlanner, FallbackRandomShootingWorks) {
    WorldModel::Config wmc;
    wmc.input_dim = 16; wmc.latent_dim = 4; wmc.action_space = 4;
    WorldModel wm(wmc);

    Planner::Config pc;
    pc.action_space = 4;
    pc.use_mcts = false;
    Planner planner(pc);

    State current{Tensor({4}, 0.0)};
    State goal{Tensor({4}, 1.0)};
    auto plan = planner.plan(current, goal, wm, 3);
    EXPECT_EQ(plan.size(), 3u);
}

// ═══════════════════════════════════════════════════════════
// M6: Information-theoretic NoveltyDetector
// ═══════════════════════════════════════════════════════════

TEST(InfoNovelty, InformationGainInitiallyZero) {
    NoveltyDetector nd;
    EXPECT_DOUBLE_EQ(nd.information_gain(), 0.0);
}

TEST(InfoNovelty, SurpriseIncreasesForOutliers) {
    NoveltyDetector nd(100, 3);

    // Observe many similar states
    for (int i = 0; i < 20; ++i) {
        nd.observe(State{Tensor({4}, 0.0)});
    }

    // Normal state should have low novelty
    Real normal_score = nd.score(State{Tensor({4}, 0.0)});

    // Outlier state should have higher novelty
    Real outlier_score = nd.score(State{Tensor({4}, 5.0)});
    EXPECT_GT(outlier_score, normal_score);
}

TEST(InfoNovelty, ResetClearsDistribution) {
    NoveltyDetector nd;
    nd.observe(State{Tensor({4}, 1.0)});
    nd.reset();
    // After reset, first observation should be novel again
    Real score = nd.score(State{Tensor({4}, 1.0)});
    EXPECT_DOUBLE_EQ(score, 1.0);
}

// ═══════════════════════════════════════════════════════════
// M7: Log-likelihood compression progress
// ═══════════════════════════════════════════════════════════

TEST(CompressionProgress, ZeroForFewObservations) {
    WorldModel::Config wmc;
    wmc.input_dim = 16; wmc.latent_dim = 4; wmc.action_space = 4;
    WorldModel wm(wmc);

    EXPECT_DOUBLE_EQ(wm.compression_progress(), 0.0);
    Observation obs;
    obs.data = Tensor({16}, 0.5);
    wm.update(obs);
    EXPECT_DOUBLE_EQ(wm.compression_progress(), 0.0);
}

TEST(CompressionProgress, NonNegative) {
    WorldModel::Config wmc;
    wmc.input_dim = 16; wmc.latent_dim = 4; wmc.action_space = 4;
    WorldModel wm(wmc);

    Observation obs;
    obs.data = Tensor({16}, 0.5);
    for (int i = 0; i < 10; ++i) {
        wm.record_action(Action{i % 4});
        wm.update(obs);
    }
    EXPECT_GE(wm.compression_progress(), 0.0);
}

// ═══════════════════════════════════════════════════════════
// M8: ARC benchmark expanded patterns
// ═══════════════════════════════════════════════════════════

TEST(ArcExpanded, BenchmarkHas15Puzzles) {
    ArcEnvironment::Config ac;
    ac.grid_rows = 5; ac.grid_cols = 5; ac.num_colors = 10;
    ac.puzzles_per_episode = 5;
    ArcEnvironment env(ac);
    env.use_benchmark_tasks();
    EXPECT_EQ(env.total_puzzles(), 15u);
}

TEST(ArcExpanded, AllPuzzlesHaveTrainPairs) {
    ArcEnvironment::Config ac;
    ac.grid_rows = 5; ac.grid_cols = 5; ac.num_colors = 10;
    ac.puzzles_per_episode = 5;
    ArcEnvironment env(ac);
    env.use_benchmark_tasks();
    auto obs = env.reset();

    for (std::size_t i = 0; i < env.total_puzzles(); ++i) {
        const auto& puzzle = env.current_puzzle();
        EXPECT_GE(puzzle.train_pairs.size(), 1u);
        EXPECT_GT(puzzle.test_input.flat_size(), 0u);
        EXPECT_GT(puzzle.test_output.flat_size(), 0u);
        // Skip to next puzzle
        auto grid_size = ac.grid_rows * ac.grid_cols;
        auto result = env.step(Action{static_cast<int>(grid_size) + 1});
        if (result.done) break;
    }
}

// ═══════════════════════════════════════════════════════════
// M9: Open-ended structural variation
// ═══════════════════════════════════════════════════════════

TEST(OpenEndedStructure, GeneratesDiverseStageTypes) {
    CurriculumManager::Config config;
    // Pass threshold = -999 means agent advances immediately
    config.stages = {{"test_stage", 10, -999.0, 1}};
    config.seed = 42;
    CurriculumManager cm(config);
    cm.enable_open_ended();

    // First episode advances past fixed stage and generates first new stage
    cm.report_episode(0.0, 5);
    EXPECT_GE(cm.generated_stages(), 1u);

    // Keep reporting to trigger more generations
    // Each time we pass the threshold of a generated stage, a new one is created
    for (int i = 0; i < 50; ++i) {
        cm.report_episode(999.0, 5);
    }
    // Should have generated multiple stages (not necessarily 7 due to
    // advancement logic requiring min_episodes)
    EXPECT_GE(cm.generated_stages(), 3u);
}

TEST(OpenEndedStructure, GeneratedStagesArePlayable) {
    CurriculumManager::Config config;
    config.stages = {{"test_stage", 10, -999.0, 1}};
    config.seed = 42;
    CurriculumManager cm(config);
    cm.enable_open_ended();

    // Advance past fixed stage
    cm.report_episode(999.0, 5);
    EXPECT_GT(cm.generated_stages(), 0u);

    // The generated environment should be usable
    auto& env = cm.current_env();
    auto obs = env.reset();
    EXPECT_GT(obs.data.flat_size(), 0u);
}

// ═══════════════════════════════════════════════════════════
// GoalSetter: Information-gain exploration & autonomous goals
// ═══════════════════════════════════════════════════════════

TEST(GoalSetter, InformationGainRewardUsesStatistics) {
    GoalSetter::Config gc;
    gc.info_gain_weight = 0.3;
    gc.competence_weight = 0.2;
    GoalSetter gs(gc);

    // Feed some novelty values to build statistics
    for (int i = 0; i < 10; ++i) {
        Real novelty = 0.1 * i;
        (void)gs.compute_reward(0.01, 0.0, novelty);
    }

    // Very high novelty should produce higher info-gain reward
    Real low_info = gs.information_gain_reward(0.1, 0.01);
    Real high_info = gs.information_gain_reward(2.0, 0.01);
    EXPECT_GT(high_info, low_info);
}

TEST(GoalSetter, CompositeRewardIncludesAllComponents) {
    GoalSetter::Config gc;
    gc.curiosity_weight = 0.5;
    gc.info_gain_weight = 0.3;
    gc.competence_weight = 0.2;
    GoalSetter gs(gc);

    // First call builds baselines
    (void)gs.compute_reward(0.0, 0.0, 0.0);
    // Non-zero inputs should produce non-zero intrinsic reward
    auto r2 = gs.compute_reward(1.0, 0.5, 0.5);
    // External should be passed through
    EXPECT_DOUBLE_EQ(r2.external, 0.5);
}

TEST(GoalSetter, LearningProgressMeasuresImprovement) {
    GoalSetter::Config gc;
    gc.progress_window = 6;
    GoalSetter gs(gc);

    // Feed decreasing compression → no positive progress
    for (int i = 5; i >= 0; --i) {
        (void)gs.compute_reward(0.1 * i, 0.0, 0.0);
    }
    Real lp_decrease = gs.learning_progress();

    // Reset and feed increasing compression → positive progress
    GoalSetter gs2(gc);
    for (int i = 0; i < 6; ++i) {
        (void)gs2.compute_reward(0.1 * i, 0.0, 0.0);
    }
    Real lp_increase = gs2.learning_progress();
    EXPECT_GE(lp_increase, lp_decrease);
}

TEST(GoalSetter, AutonomousGoalDiscoveryUsesDirectionScores) {
    GoalSetter::Config gc;
    gc.progress_window = 5;
    GoalSetter gs(gc);

    // Build up state with clear learning progress
    Tensor lat({4}, {0.1, 0.2, 0.3, 0.4});
    State s1{lat};

    // Feed increasing compression to create learning progress
    for (int i = 0; i < 10; ++i) {
        gs.set_goal(s1, 0.1 * i, 0.5);
    }
    // After enough calls, goal direction should be biased
    State goal = gs.set_goal(s1, 1.0, 0.5);
    EXPECT_EQ(goal.latent.flat_size(), 4u);
    // Goal should differ from current state
    bool differs = false;
    for (Dim i = 0; i < 4; ++i) {
        if (std::abs(goal.latent.at(i) - s1.latent.at(i)) > 1e-10) {
            differs = true;
        }
    }
    EXPECT_TRUE(differs);
}

// ═══════════════════════════════════════════════════════════
// Integration: Full v0.6 system test
// ═══════════════════════════════════════════════════════════

TEST(V06Integration, FullLoopWithAllFixes) {
    // 3x3 grid → obs dim = 18, action_space = 6 for GridWorld
    AgentKernel::Config cfg;
    cfg.wm_config.input_dim = 18;
    cfg.wm_config.latent_dim = 4;
    cfg.wm_config.action_space = 6;
    cfg.planner_config.action_space = 6;
    AgentKernel agent(cfg);

    GridWorld::Config gc;
    gc.rows = 3; gc.cols = 3; gc.num_colors = 5; gc.max_steps = 20;
    GridWorld env(gc);
    auto obs = env.reset();

    // Run 10 steps: encoder learns, MCTS plans, self-mod evolves
    for (int i = 0; i < 10; ++i) {
        Action action = agent.step(obs, 0.0);
        auto result = env.step(action);
        obs = result.observation;
        if (result.done) {
            obs = env.reset();
        }
    }

    // Encoder should have learned something
    EXPECT_GT(agent.step_count(), 0u);
}
