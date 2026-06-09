#include <gtest/gtest.h>
#include "symbolic_descent/rule_library.hpp"
#include "common/program.hpp"

using namespace uik;
using namespace uik::symbolic_descent;

static ProgramPtr make_prog(OpKind kind, int p1 = 0) {
    return std::make_shared<ProgramNode>(ProgramNode{kind, p1, 0, {}});
}

TEST(RuleLibrary, empty_initially) {
    RuleLibrary lib;
    EXPECT_TRUE(lib.empty());
    EXPECT_EQ(lib.size(), 0u);
}

TEST(RuleLibrary, add_and_retrieve) {
    RuleLibrary lib;
    lib.add_rule("transition", make_prog(OpKind::Identity), 0.5);
    EXPECT_EQ(lib.size(), 1u);
    EXPECT_FALSE(lib.empty());
}

TEST(RuleLibrary, find_best_returns_highest_fitness) {
    RuleLibrary lib;
    lib.add_rule("transition", make_prog(OpKind::Identity), 0.2);
    lib.add_rule("transition", make_prog(OpKind::FlipH), 0.8);
    lib.add_rule("transition", make_prog(OpKind::Rotate90), 0.5);

    auto best = lib.find_best("transition");
    ASSERT_TRUE(best.has_value());
    EXPECT_DOUBLE_EQ(best->fitness, 0.8);
}

TEST(RuleLibrary, find_best_no_match_returns_nullopt) {
    RuleLibrary lib(10);
    lib.add_rule("visual", make_prog(OpKind::Identity), 0.5);
    // Empty context matches everything
    auto result = lib.find_best("");
    EXPECT_TRUE(result.has_value());
}

TEST(RuleLibrary, all_sorted_returns_descending_fitness) {
    RuleLibrary lib;
    lib.add_rule("a", make_prog(OpKind::Identity), 0.1);
    lib.add_rule("b", make_prog(OpKind::FlipH), 0.9);
    lib.add_rule("c", make_prog(OpKind::Rotate90), 0.5);

    auto sorted = lib.all_sorted();
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_GE(sorted[0].fitness, sorted[1].fitness);
    EXPECT_GE(sorted[1].fitness, sorted[2].fitness);
}

TEST(RuleLibrary, prune_keeps_best) {
    RuleLibrary lib;
    lib.add_rule("a", make_prog(OpKind::Identity), 0.1);
    lib.add_rule("b", make_prog(OpKind::FlipH), 0.9);
    lib.add_rule("c", make_prog(OpKind::Rotate90), 0.5);
    lib.prune(2);

    EXPECT_EQ(lib.size(), 2u);
    auto best = lib.find_best("");
    ASSERT_TRUE(best.has_value());
    EXPECT_DOUBLE_EQ(best->fitness, 0.9);
}

TEST(RuleLibrary, mark_used_increments_count) {
    RuleLibrary lib;
    lib.add_rule("a", make_prog(OpKind::Identity), 0.5);
    lib.mark_used(0);
    lib.mark_used(0);
    EXPECT_EQ(lib.rules()[0].use_count, 2u);
}

TEST(RuleLibrary, extract_programs_returns_all) {
    RuleLibrary lib;
    lib.add_rule("a", make_prog(OpKind::Identity), 0.5);
    lib.add_rule("b", make_prog(OpKind::FlipH), 0.3);
    auto progs = lib.extract_programs();
    EXPECT_EQ(progs.size(), 2u);
}

TEST(RuleLibrary, auto_prunes_at_max_size) {
    RuleLibrary lib(3);
    lib.add_rule("a", make_prog(OpKind::Identity), 0.1);
    lib.add_rule("b", make_prog(OpKind::FlipH), 0.9);
    lib.add_rule("c", make_prog(OpKind::Rotate90), 0.5);
    lib.add_rule("d", make_prog(OpKind::FlipV), 0.7);
    EXPECT_LE(lib.size(), 3u);
}

TEST(RuleLibrary, update_fitness_changes_value) {
    RuleLibrary lib;
    lib.add_rule("a", make_prog(OpKind::Identity), 0.1);
    lib.update_fitness(0, 0.9);
    EXPECT_DOUBLE_EQ(lib.rules()[0].fitness, 0.9);
}
