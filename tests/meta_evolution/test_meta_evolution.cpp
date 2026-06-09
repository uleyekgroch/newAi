#include <gtest/gtest.h>
#include "meta_evolution/archive.hpp"
#include "meta_evolution/mutator.hpp"
#include "meta_evolution/evolutionary_selector.hpp"

using namespace uik;
using namespace uik::meta_evolution;

// ── Archive Tests ──

TEST(Archive, initially_empty) {
    Archive archive;
    EXPECT_TRUE(archive.empty());
    EXPECT_EQ(archive.size(), 0u);
}

TEST(Archive, add_increases_size) {
    Archive archive;
    archive.add(identity(), 1.0, 0.5, 0);
    EXPECT_EQ(archive.size(), 1u);
    EXPECT_FALSE(archive.empty());
}

TEST(Archive, sample_returns_value_when_nonempty) {
    Archive archive;
    archive.add(identity(), 1.0, 0.5, 0);
    auto entry = archive.sample();
    EXPECT_TRUE(entry.has_value());
    EXPECT_NE(entry->program, nullptr);
}

TEST(Archive, sample_returns_nullopt_when_empty) {
    Archive archive;
    EXPECT_FALSE(archive.sample().has_value());
}

TEST(Archive, best_returns_highest_fitness) {
    Archive archive;
    archive.add(identity(), 1.0, 0.0, 0);
    archive.add(make_program(OpKind::FlipH), 5.0, 0.0, 0);
    archive.add(make_program(OpKind::FlipV), 3.0, 0.0, 0);

    auto best = archive.best();
    EXPECT_TRUE(best.has_value());
    EXPECT_DOUBLE_EQ(best->fitness, 5.0);
}

TEST(Archive, top_k_returns_sorted) {
    Archive archive;
    archive.add(identity(), 1.0, 0.0, 0);
    archive.add(make_program(OpKind::FlipH), 5.0, 0.0, 0);
    archive.add(make_program(OpKind::FlipV), 3.0, 0.0, 0);

    auto top = archive.top_k(2);
    EXPECT_EQ(top.size(), 2u);
    EXPECT_GE(top[0].fitness, top[1].fitness);
}

TEST(Archive, prunes_when_exceeding_max_size) {
    Archive archive(5);
    for (int i = 0; i < 10; ++i) {
        archive.add(identity(), static_cast<Real>(i), 0.0, 0);
    }
    EXPECT_LE(archive.size(), 5u);
}

TEST(Archive, diversity_zero_for_identical_programs) {
    Archive archive;
    archive.add(identity(), 1.0, 0.0, 0);
    archive.add(identity(), 2.0, 0.0, 0);
    EXPECT_DOUBLE_EQ(archive.diversity(), 0.0);
}

TEST(Archive, diversity_positive_for_different_programs) {
    Archive archive;
    archive.add(identity(), 1.0, 0.0, 0); // dl=1
    auto complex = compose(identity(), make_program(OpKind::FlipH)); // dl=3
    archive.add(complex, 2.0, 0.0, 0);
    EXPECT_GT(archive.diversity(), 0.0);
}

// ── Mutator Tests ──

TEST(Mutator, mutate_returns_non_null) {
    Mutator mutator;
    auto original = identity();
    auto mutated = mutator.mutate(original);
    EXPECT_NE(mutated, nullptr);
}

TEST(Mutator, crossover_returns_non_null) {
    Mutator mutator;
    auto a = identity();
    auto b = make_program(OpKind::FlipH);
    auto child = mutator.crossover(a, b);
    EXPECT_NE(child, nullptr);
}

TEST(Mutator, mutate_null_returns_valid) {
    Mutator mutator;
    auto result = mutator.mutate(nullptr);
    EXPECT_NE(result, nullptr);
}

// ── EvolutionarySelector Tests ──

TEST(EvolutionarySelector, evolve_once_adds_to_archive) {
    EvolutionarySelector::Config cfg;
    cfg.offspring_per_gen = 5;
    cfg.archive_size = 50;
    EvolutionarySelector selector(cfg);

    auto fitness = [](const ProgramPtr& p) -> Real {
        return -static_cast<Real>(p->description_length());
    };
    selector.evolve_once(fitness);
    EXPECT_GT(selector.archive().size(), 0u);
    EXPECT_EQ(selector.generation(), 1u);
}

TEST(EvolutionarySelector, evolve_multiple_generations) {
    EvolutionarySelector::Config cfg;
    cfg.offspring_per_gen = 10;
    cfg.archive_size = 100;
    EvolutionarySelector selector(cfg);

    auto fitness = [](const ProgramPtr& p) -> Real {
        return -static_cast<Real>(p->description_length());
    };
    selector.evolve(fitness, 5);
    EXPECT_EQ(selector.generation(), 5u);
    EXPECT_GT(selector.archive().size(), 0u);
}

TEST(EvolutionarySelector, seed_populates_archive) {
    EvolutionarySelector selector;
    std::vector<ProgramPtr> seeds = {
        identity(),
        make_program(OpKind::FlipH),
        make_program(OpKind::FlipV)
    };
    auto fitness = [](const ProgramPtr& p) -> Real {
        return -static_cast<Real>(p->description_length());
    };
    selector.seed(seeds, fitness);
    EXPECT_EQ(selector.archive().size(), 3u);
}

TEST(EvolutionarySelector, best_fitness_improves_or_stable) {
    EvolutionarySelector::Config cfg;
    cfg.offspring_per_gen = 20;
    cfg.archive_size = 100;
    EvolutionarySelector selector(cfg);

    auto fitness = [](const ProgramPtr& p) -> Real {
        // Prefer shorter programs
        return -static_cast<Real>(p->description_length());
    };

    selector.evolve(fitness, 1);
    auto best_early = selector.archive().best();
    EXPECT_TRUE(best_early.has_value());

    selector.evolve(fitness, 20);
    auto best_late = selector.archive().best();
    EXPECT_TRUE(best_late.has_value());
    EXPECT_GE(best_late->fitness, best_early->fitness);
}
