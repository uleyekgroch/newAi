#include <gtest/gtest.h>
#include "symbolic_descent/dsl.hpp"
#include "symbolic_descent/program_space.hpp"
#include "symbolic_descent/mdl_evaluator.hpp"
#include "symbolic_descent/search_engine.hpp"

using namespace uik;
using namespace uik::symbolic_descent;

// ── DSL Tests ──

TEST(DSL, identity_returns_input_unchanged) {
    DSL dsl;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    auto prog = identity();
    Tensor output = dsl.execute(prog, input);
    EXPECT_EQ(output, input);
}

TEST(DSL, flip_h_reverses_1d) {
    DSL dsl;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    auto prog = make_program(OpKind::FlipH);
    Tensor output = dsl.execute(prog, input);
    Tensor expected({4}, {4.0, 3.0, 2.0, 1.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, flip_h_2d_reverses_columns) {
    DSL dsl;
    Tensor input({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto prog = make_program(OpKind::FlipH);
    Tensor output = dsl.execute(prog, input);
    Tensor expected({2, 3}, {3.0, 2.0, 1.0, 6.0, 5.0, 4.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, flip_v_2d_reverses_rows) {
    DSL dsl;
    Tensor input({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto prog = make_program(OpKind::FlipV);
    Tensor output = dsl.execute(prog, input);
    Tensor expected({2, 3}, {4.0, 5.0, 6.0, 1.0, 2.0, 3.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, map_color_replaces_values) {
    DSL dsl;
    Tensor input({4}, {1.0, 2.0, 1.0, 3.0});
    auto prog = make_program(OpKind::MapColor, 1, 9);
    Tensor output = dsl.execute(prog, input);
    Tensor expected({4}, {9.0, 2.0, 9.0, 3.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, constant_fills_all) {
    DSL dsl;
    Tensor input({3}, {1.0, 2.0, 3.0});
    auto prog = make_program(OpKind::Constant, 7);
    Tensor output = dsl.execute(prog, input);
    Tensor expected({3}, {7.0, 7.0, 7.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, compose_chains_operations) {
    DSL dsl;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    auto prog = compose(
        make_program(OpKind::FlipH),
        make_program(OpKind::MapColor, 4, 0)
    );
    Tensor output = dsl.execute(prog, input);
    // FlipH: [4,3,2,1] → MapColor(4→0): [0,3,2,1]
    Tensor expected({4}, {0.0, 3.0, 2.0, 1.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, translate_2d_shifts_right) {
    DSL dsl;
    Tensor input({2, 3}, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    auto prog = make_program(OpKind::Translate, 1, 0); // dx=1, dy=0
    Tensor output = dsl.execute(prog, input);
    // [1,0,0; 0,0,0] → shift right → [0,1,0; 0,0,0]
    Tensor expected({2, 3}, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    EXPECT_EQ(output, expected);
}

TEST(DSL, null_program_throws) {
    DSL dsl;
    Tensor input({4}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_THROW(dsl.execute(nullptr, input), std::invalid_argument);
}

// ── ProgramSpace Tests ──

TEST(ProgramSpace, random_program_not_null) {
    ProgramSpace space;
    auto prog = space.random_program();
    EXPECT_NE(prog, nullptr);
}

TEST(ProgramSpace, mutate_produces_different_or_same) {
    ProgramSpace space;
    auto original = make_program(OpKind::Identity);
    auto mutated = space.mutate(original);
    EXPECT_NE(mutated, nullptr);
}

TEST(ProgramSpace, crossover_produces_valid) {
    ProgramSpace space;
    auto a = make_program(OpKind::FlipH);
    auto b = make_program(OpKind::FlipV);
    auto child = space.crossover(a, b);
    EXPECT_NE(child, nullptr);
}

TEST(ProgramSpace, neighborhood_returns_requested_count) {
    ProgramSpace space;
    auto prog = make_program(OpKind::Identity);
    auto neighbors = space.neighborhood(prog, 15);
    EXPECT_EQ(neighbors.size(), 15u);
}

// ── MdlEvaluator Tests ──

TEST(MdlEvaluator, identity_scores_well_on_identity_data) {
    MdlEvaluator eval(1.0);
    MdlEvaluator::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {1.0, 2.0, 3.0, 4.0})}
    };
    auto id = identity();
    Real score = eval.score(id, data);
    // Identity has description_length=1, loss=0 → score = -(1 + 0) = -1
    EXPECT_NEAR(score, -1.0, 1e-6);
}

TEST(MdlEvaluator, shorter_perfect_program_scores_better) {
    MdlEvaluator eval(1.0);
    MdlEvaluator::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {1.0, 2.0, 3.0, 4.0})}
    };
    auto simple = identity(); // dl=1
    auto complex = compose(identity(), identity()); // dl=3
    EXPECT_GT(eval.score(simple, data), eval.score(complex, data));
}

TEST(MdlEvaluator, is_perfect_fit_true_for_identity) {
    MdlEvaluator eval(1.0);
    MdlEvaluator::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {1.0, 2.0, 3.0, 4.0})}
    };
    EXPECT_TRUE(eval.is_perfect_fit(identity(), data));
}

TEST(MdlEvaluator, is_perfect_fit_false_for_wrong_program) {
    MdlEvaluator eval(1.0);
    MdlEvaluator::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {5.0, 6.0, 7.0, 8.0})}
    };
    EXPECT_FALSE(eval.is_perfect_fit(identity(), data));
}

TEST(MdlEvaluator, null_program_returns_neg_inf) {
    MdlEvaluator eval(1.0);
    MdlEvaluator::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {1.0, 2.0, 3.0, 4.0})}
    };
    EXPECT_EQ(eval.score(nullptr, data), -std::numeric_limits<Real>::infinity());
}

// ── SearchEngine Tests ──

TEST(SearchEngine, finds_identity_for_identity_task) {
    SearchEngine::Config cfg;
    cfg.population_size = 30;
    cfg.seed = 42;
    SearchEngine engine(cfg);

    ISearchEngine::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {1.0, 2.0, 3.0, 4.0})},
        {Tensor({4}, {5.0, 6.0, 7.0, 8.0}), Tensor({4}, {5.0, 6.0, 7.0, 8.0})}
    };

    auto result = engine.search(data, 10);
    EXPECT_TRUE(result.has_value());
}

TEST(SearchEngine, finds_flip_h_for_reverse_task) {
    SearchEngine::Config cfg;
    cfg.population_size = 50;
    cfg.seed = 42;
    SearchEngine engine(cfg);

    ISearchEngine::Dataset data = {
        {Tensor({4}, {1.0, 2.0, 3.0, 4.0}), Tensor({4}, {4.0, 3.0, 2.0, 1.0})},
        {Tensor({4}, {5.0, 6.0, 7.0, 8.0}), Tensor({4}, {8.0, 7.0, 6.0, 5.0})}
    };

    auto result = engine.search(data, 50);
    EXPECT_TRUE(result.has_value());

    // Verify the found program actually works
    DSL dsl;
    Tensor test_input({4}, {9.0, 8.0, 7.0, 6.0});
    Tensor output = dsl.execute(*result, test_input);
    Tensor expected({4}, {6.0, 7.0, 8.0, 9.0});
    EXPECT_EQ(output, expected);
}

TEST(SearchEngine, empty_data_returns_nullopt) {
    SearchEngine engine;
    auto result = engine.search({}, 10);
    EXPECT_FALSE(result.has_value());
}
