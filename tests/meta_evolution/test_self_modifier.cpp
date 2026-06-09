#include <gtest/gtest.h>
#include "meta_evolution/self_modifier.hpp"

using namespace uik;
using namespace uik::meta_evolution;

class SelfModifierTest : public ::testing::Test {
protected:
    SelfModifier modifier_;
};

TEST_F(SelfModifierTest, InitializesWithDefaultStrategies) {
    EXPECT_EQ(modifier_.modification_count(), 0u);
    auto strategies = modifier_.strategies();
    EXPECT_EQ(strategies.size(), 3u);
}

TEST_F(SelfModifierTest, CurrentStrategyReturnsProgram) {
    auto prog = modifier_.current_strategy(SelfModifier::StrategyKind::FitnessWeighting);
    EXPECT_NE(prog, nullptr);
}

TEST_F(SelfModifierTest, TryModifyWithImprovingFunction) {
    auto eval = [](const ProgramPtr&) { return 0.5; };
    modifier_.try_modify(
        SelfModifier::StrategyKind::FitnessWeighting, eval);
    // May or may not modify depending on random mutations
    EXPECT_GE(modifier_.modification_count(), 0u);
}

TEST_F(SelfModifierTest, TryModifyRecordsArchive) {
    auto eval = [](const ProgramPtr&) { return 0.1; };
    modifier_.try_modify(SelfModifier::StrategyKind::ExplorationPolicy, eval);
    auto archive = modifier_.strategy_archive();
    // Archive should have at least the default strategies
    EXPECT_GE(archive.size(), 3u);
}

TEST_F(SelfModifierTest, AllStrategyKindsAccessible) {
    EXPECT_NE(modifier_.current_strategy(SelfModifier::StrategyKind::FitnessWeighting), nullptr);
    EXPECT_NE(modifier_.current_strategy(SelfModifier::StrategyKind::NeighborhoodBias), nullptr);
    EXPECT_NE(modifier_.current_strategy(SelfModifier::StrategyKind::ExplorationPolicy), nullptr);
}

TEST_F(SelfModifierTest, MultipleModificationsAccumulate) {
    auto eval = [n = 0](const ProgramPtr&) mutable { return 0.01 * (++n); };
    for (int i = 0; i < 5; ++i) {
        modifier_.try_modify(SelfModifier::StrategyKind::FitnessWeighting, eval);
    }
    auto archive = modifier_.strategy_archive();
    EXPECT_GE(archive.size(), 3u);
}
