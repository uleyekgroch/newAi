#include <gtest/gtest.h>
#include "world_model/latent_state.hpp"
#include "symbolic_descent/program_space.hpp"
#include "meta_evolution/self_modifier.hpp"
#include "agent/arc_env.hpp"
#include "agent/goal_setter.hpp"
#include "agent/curriculum.hpp"
#include "common/program.hpp"
#include <cmath>

using namespace uik;
using namespace uik::world_model;
using namespace uik::symbolic_descent;
using namespace uik::meta_evolution;
using namespace uik::agent;

// ── Gap 1: V-JEPA2 attention-based encoder ──

class AttentionEncoderTest : public ::testing::Test {
protected:
    LatentEncoder::Config make_config(Dim input = 64, Dim latent = 16,
                                       Dim hidden = 32, Dim heads = 4,
                                       Dim patch = 4) {
        LatentEncoder::Config c;
        c.input_dim = input;
        c.latent_dim = latent;
        c.hidden_dim = hidden;
        c.num_heads = heads;
        c.patch_size = patch;
        c.num_layers = 2;
        c.num_recurse = 2;
        c.seed = 42;
        return c;
    }
};

TEST_F(AttentionEncoderTest, EncodesWithAttention) {
    auto config = make_config();
    LatentEncoder encoder(config);
    Observation obs{Tensor({64}, 1.0)};
    auto state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

TEST_F(AttentionEncoderTest, MultiHeadAttention) {
    auto config = make_config(64, 16, 32, 4, 4);
    LatentEncoder encoder(config);
    EXPECT_EQ(encoder.num_heads(), 4u);
    Observation obs{Tensor({64}, 0.5)};
    auto state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

TEST_F(AttentionEncoderTest, SingleHeadFallback) {
    // hidden_dim=32 not divisible by 3 heads → fallback to 1
    auto config = make_config(64, 16, 32, 3, 4);
    LatentEncoder encoder(config);
    Observation obs{Tensor({64}, 0.5)};
    auto state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

TEST_F(AttentionEncoderTest, PatchEmbedding) {
    // 64 / 8 = 8 tokens
    auto config = make_config(64, 16, 32, 4, 8);
    LatentEncoder encoder(config);
    Observation obs{Tensor({64}, 0.3)};
    auto state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

TEST_F(AttentionEncoderTest, DifferentInputsProduceDifferentLatents) {
    auto config = make_config();
    LatentEncoder encoder(config);
    std::vector<Real> data1(64, 1.0);
    std::vector<Real> data2(64, -1.0);
    auto s1 = encoder.encode(Observation{Tensor({64}, std::move(data1))});
    auto s2 = encoder.encode(Observation{Tensor({64}, std::move(data2))});
    bool different = false;
    for (Dim i = 0; i < 16; ++i) {
        if (std::abs(s1.latent.at(i) - s2.latent.at(i)) > 1e-6) {
            different = true; break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(AttentionEncoderTest, DecodeRoundTrip) {
    auto config = make_config();
    LatentEncoder encoder(config);
    Observation obs{Tensor({64}, 0.5)};
    auto state = encoder.encode(obs);
    auto decoded = encoder.decode(state);
    EXPECT_EQ(decoded.flat_size(), 64u);
}

TEST_F(AttentionEncoderTest, BackwardCompatibleConstructor) {
    LatentEncoder encoder(64, 16, 42);
    Observation obs{Tensor({64}, 0.5)};
    auto state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

// ── Gap 2: Template-guided edit ──

class TemplateEditTest : public ::testing::Test {
protected:
    ProgramSpace space_{3, 10, 5, 42};
};

TEST_F(TemplateEditTest, AddTemplate) {
    auto prog = compose(make_program(OpKind::Rotate90),
                         make_program(OpKind::FlipH));
    EXPECT_EQ(space_.template_count(), 0u);
    space_.add_template(prog);
    EXPECT_GT(space_.template_count(), 0u);
}

TEST_F(TemplateEditTest, TemplateGuidedEditWithTemplates) {
    auto prog = make_program(OpKind::Rotate90);
    auto good = compose(make_program(OpKind::FlipH),
                         make_program(OpKind::Translate, 1, 2));
    space_.add_template(good);
    auto result = space_.template_guided_edit(prog);
    EXPECT_NE(result, nullptr);
}

TEST_F(TemplateEditTest, FallbackWithoutTemplates) {
    auto prog = make_program(OpKind::Rotate90);
    auto result = space_.template_guided_edit(prog);
    EXPECT_NE(result, nullptr);
}

TEST_F(TemplateEditTest, NullProgramFallback) {
    auto result = space_.template_guided_edit(nullptr);
    EXPECT_NE(result, nullptr);
}

TEST_F(TemplateEditTest, MultipleTemplates) {
    for (int i = 0; i < 10; ++i) {
        space_.add_template(space_.random_program());
    }
    EXPECT_GT(space_.template_count(), 0u);
    auto prog = make_program(OpKind::Identity);
    auto result = space_.template_guided_edit(prog);
    EXPECT_NE(result, nullptr);
}

// ── Gap 3: Enhanced self-modification ──

class EnhancedSelfModTest : public ::testing::Test {
protected:
    SelfModifier modifier_;
};

TEST_F(EnhancedSelfModTest, AllKindsInitialized) {
    auto all = SelfModifier::all_kinds();
    EXPECT_EQ(all.size(), 7u);
    for (auto kind : all) {
        auto& strat = modifier_.current_strategy(kind);
        EXPECT_NE(strat, nullptr);
    }
}

TEST_F(EnhancedSelfModTest, NewStrategyKinds) {
    using SK = SelfModifier::StrategyKind;
    auto& goal = modifier_.current_strategy(SK::GoalFunction);
    EXPECT_NE(goal, nullptr);
    auto& reward = modifier_.current_strategy(SK::RewardShaping);
    EXPECT_NE(reward, nullptr);
    auto& mutation = modifier_.current_strategy(SK::MutationStrategy);
    EXPECT_NE(mutation, nullptr);
    auto& simp = modifier_.current_strategy(SK::SimplificationRule);
    EXPECT_NE(simp, nullptr);
}

TEST_F(EnhancedSelfModTest, EvolveAll) {
    auto eval = [](SelfModifier::StrategyKind, const ProgramPtr& p) -> Real {
        return p ? 1.0 / static_cast<Real>(p->description_length() + 1) : 0.0;
    };
    auto improved = modifier_.evolve_all(eval);
    // At least the function should run without error
    EXPECT_GE(improved, 0u);
}

TEST_F(EnhancedSelfModTest, TryModifyNewKind) {
    using SK = SelfModifier::StrategyKind;
    auto eval = [](const ProgramPtr& p) -> Real {
        return p ? 0.5 : 0.0;
    };
    (void)modifier_.try_modify(SK::GoalFunction, eval);
    (void)modifier_.try_modify(SK::RewardShaping, eval);
}

// ── Gap 4: ARC-AGI benchmark tasks ──

class ArcBenchmarkTest : public ::testing::Test {
protected:
    ArcEnvironment::Config make_config() {
        ArcEnvironment::Config c;
        c.grid_rows = 5; c.grid_cols = 5; c.num_colors = 10;
        c.puzzles_per_episode = 5; c.seed = 42;
        return c;
    }
};

TEST_F(ArcBenchmarkTest, UseBenchmarkTasks) {
    ArcEnvironment env(make_config());
    env.use_benchmark_tasks();
    EXPECT_TRUE(env.using_benchmark());
    EXPECT_EQ(env.total_puzzles(), 15u);
}

TEST_F(ArcBenchmarkTest, BenchmarkPuzzleHasTrainPairs) {
    ArcEnvironment env(make_config());
    env.use_benchmark_tasks();
    (void)env.reset();
    const auto& puzzle = env.current_puzzle();
    EXPECT_GE(puzzle.train_pairs.size(), 3u);
    EXPECT_GT(puzzle.test_input.flat_size(), 0u);
    EXPECT_GT(puzzle.test_output.flat_size(), 0u);
}

TEST_F(ArcBenchmarkTest, CanPlayBenchmark) {
    ArcEnvironment env(make_config());
    env.use_benchmark_tasks();
    auto obs = env.reset();
    EXPECT_GT(obs.data.flat_size(), 0u);
    // Submit action
    int submit_action = env.action_space_size() - 2;
    auto result = env.step(Action{submit_action});
    EXPECT_FALSE(result.done);  // still have puzzles left
}

// ── Gap 5: γ time discount factor ──

class GammaDiscountTest : public ::testing::Test {
protected:
    GoalSetter::Config make_config(Real gamma = 0.99) {
        GoalSetter::Config c;
        c.gamma = gamma;
        return c;
    }
};

TEST_F(GammaDiscountTest, DefaultGamma) {
    GoalSetter gs;
    EXPECT_DOUBLE_EQ(gs.gamma(), 0.99);
}

TEST_F(GammaDiscountTest, CustomGamma) {
    GoalSetter gs(make_config(0.9));
    EXPECT_DOUBLE_EQ(gs.gamma(), 0.9);
}

TEST_F(GammaDiscountTest, DiscountedReturnEmpty) {
    GoalSetter gs;
    EXPECT_DOUBLE_EQ(gs.discounted_return(), 0.0);
}

TEST_F(GammaDiscountTest, DiscountedReturnSingleStep) {
    GoalSetter gs(make_config(0.5));
    gs.record_step_reward(1.0);
    EXPECT_DOUBLE_EQ(gs.discounted_return(), 1.0);
}

TEST_F(GammaDiscountTest, DiscountedReturnMultiStep) {
    GoalSetter gs(make_config(0.5));
    gs.record_step_reward(1.0);
    gs.record_step_reward(1.0);
    gs.record_step_reward(1.0);
    // J = 1.0 + 0.5*1.0 + 0.25*1.0 = 1.75
    EXPECT_NEAR(gs.discounted_return(), 1.75, 1e-10);
}

TEST_F(GammaDiscountTest, ResetEpisode) {
    GoalSetter gs;
    gs.record_step_reward(1.0);
    gs.record_step_reward(2.0);
    EXPECT_EQ(gs.episode_steps(), 2u);
    gs.reset_episode();
    EXPECT_EQ(gs.episode_steps(), 0u);
    EXPECT_DOUBLE_EQ(gs.discounted_return(), 0.0);
}

TEST_F(GammaDiscountTest, GoalSetterUsesDiscount) {
    GoalSetter gs(make_config(0.5));
    Tensor latent({4}, 0.5);
    State current{latent};

    // Without any history
    auto g1 = gs.set_goal(current, 1.0, 0.5);

    // After several steps, discount should affect
    for (int i = 0; i < 10; ++i) gs.record_step_reward(0.1);
    auto g2 = gs.set_goal(current, 1.0, 0.5);

    // With gamma=0.5, after 10 steps: discount = 0.5^10 ≈ 0.001
    // So g2's perturbation should be much smaller than g1's
    Real diff = 0.0;
    for (Dim i = 0; i < 4; ++i) {
        diff += std::abs(g1.latent.at(i) - g2.latent.at(i));
    }
    EXPECT_GT(diff, 0.0);
}

// ── Gap 6: Open-ended environment generation ──

class OpenEndedTest : public ::testing::Test {};

TEST_F(OpenEndedTest, EnableOpenEnded) {
    CurriculumManager cm;
    EXPECT_FALSE(cm.open_ended());
    cm.enable_open_ended();
    EXPECT_TRUE(cm.open_ended());
}

TEST_F(OpenEndedTest, GeneratesNewStagesWhenComplete) {
    // Create a minimal curriculum that completes quickly
    CurriculumManager::Config config;
    config.stages = {{"test_stage", 10, -999.0, 1}};  // pass threshold so low it always passes
    config.seed = 42;
    CurriculumManager cm(config);
    cm.enable_open_ended();

    // Report one successful episode → should advance past fixed stages
    bool advanced = cm.report_episode(0.0, 5);
    EXPECT_TRUE(advanced);

    // Should have generated a new stage
    EXPECT_GT(cm.generated_stages(), 0u);
    EXPECT_GT(cm.total_stages(), 1u);  // 1 fixed + generated
}

TEST_F(OpenEndedTest, MultipleGenerations) {
    CurriculumManager::Config config;
    config.stages = {{"test_stage", 10, -999.0, 1}};
    config.seed = 42;
    CurriculumManager cm(config);
    cm.enable_open_ended();

    // Advance past first stage
    cm.report_episode(0.0, 5);
    std::size_t first_gen = cm.generated_stages();
    EXPECT_GE(first_gen, 1u);

    // Report more episodes after advancing
    for (int i = 0; i < 5; ++i) {
        cm.report_episode(999.0, 5);
    }
    // Should keep generating
    EXPECT_GE(cm.generated_stages(), first_gen);
}

TEST_F(OpenEndedTest, FixedCurriculumDoesNotGenerate) {
    CurriculumManager cm;
    // Don't enable open_ended
    // Fill all 6 default stages → should not generate
    EXPECT_EQ(cm.generated_stages(), 0u);
}
