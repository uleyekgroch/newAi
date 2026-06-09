#include <gtest/gtest.h>
#include "symbolic_descent/program_space.hpp"

using namespace uik;
using namespace uik::symbolic_descent;

class GuidedSearchTest : public ::testing::Test {
protected:
    ProgramSpace space_{4, 10, 5, 42};
};

TEST_F(GuidedSearchTest, TypeAwareMutatePreservesStructure) {
    auto prog = make_program(OpKind::Rotate90);
    auto mutated = space_.type_aware_mutate(prog);
    EXPECT_NE(mutated, nullptr);
    // Should mutate to a compatible geometric op
    auto kind = mutated->kind;
    EXPECT_TRUE(kind == OpKind::Rotate90 || kind == OpKind::FlipH ||
                kind == OpKind::FlipV);
}

TEST_F(GuidedSearchTest, TypeAwareMutateColorOps) {
    auto prog = make_program(OpKind::Fill, 1, 2);
    auto mutated = space_.type_aware_mutate(prog);
    auto kind = mutated->kind;
    EXPECT_TRUE(kind == OpKind::Fill || kind == OpKind::MapColor ||
                kind == OpKind::Constant);
}

TEST_F(GuidedSearchTest, AnalogySwapsStructurallySimilar) {
    auto a = compose(make_program(OpKind::Rotate90),
                      make_program(OpKind::FlipH));
    auto b = compose(make_program(OpKind::FlipV),
                      make_program(OpKind::Translate, 1, 2));
    auto result = space_.analogy_crossover(a, b);
    EXPECT_NE(result, nullptr);
    EXPECT_EQ(result->kind, OpKind::Compose);
}

TEST_F(GuidedSearchTest, GuidedNeighborhoodRespectsWeights) {
    auto prog = make_program(OpKind::Identity);
    // Heavy weight on mutation (index 0)
    std::vector<Real> weights = {10.0, 0.0, 0.0, 0.0, 0.0};
    auto neighbors = space_.guided_neighborhood(prog, weights, 5);
    EXPECT_EQ(neighbors.size(), 5u);
}

TEST_F(GuidedSearchTest, GuidedNeighborhoodHandlesZeroWeights) {
    auto prog = make_program(OpKind::Identity);
    std::vector<Real> weights = {0.0, 0.0, 0.0, 0.0, 0.0};
    auto neighbors = space_.guided_neighborhood(prog, weights, 5);
    EXPECT_EQ(neighbors.size(), 5u);
}

TEST_F(GuidedSearchTest, ParamPerturbationKeepsStructure) {
    auto prog = make_program(OpKind::Translate, 3, 3);
    auto perturbed = space_.param_perturbation(prog);
    EXPECT_EQ(perturbed->kind, OpKind::Translate);
    // Params should be slightly different
    EXPECT_NE(perturbed->param1, prog->param1);  // unlikely to be same
}

TEST_F(GuidedSearchTest, SimplifyRemovesIdentity) {
    auto prog = compose(make_program(OpKind::Identity),
                         make_program(OpKind::Rotate90));
    auto simplified = space_.simplify(prog);
    EXPECT_EQ(simplified->kind, OpKind::Rotate90);
}

TEST_F(GuidedSearchTest, SimplifyDoubleFlipH) {
    auto inner = make_program(OpKind::FlipH, 0, 0, {make_program(OpKind::Identity)});
    auto prog = make_program(OpKind::FlipH, 0, 0, {inner});
    auto simplified = space_.simplify(prog);
    EXPECT_EQ(simplified->kind, OpKind::Identity);
}

TEST_F(GuidedSearchTest, NeighborhoodIncludesGuidedOps) {
    auto prog = make_program(OpKind::Rotate90);
    auto neighbors = space_.neighborhood(prog, 20);
    EXPECT_EQ(neighbors.size(), 20u);
    for (const auto& n : neighbors) {
        EXPECT_NE(n, nullptr);
    }
}

TEST_F(GuidedSearchTest, AnalogyCrossoverWithLeafs) {
    auto a = make_program(OpKind::Identity);
    auto b = make_program(OpKind::Rotate90);
    auto result = space_.analogy_crossover(a, b);
    EXPECT_NE(result, nullptr);
}
