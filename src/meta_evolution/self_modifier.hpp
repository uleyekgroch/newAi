#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include "symbolic_descent/program_space.hpp"
#include "meta_evolution/archive.hpp"
#include <functional>
#include <vector>
#include <random>

namespace uik::meta_evolution {

// Darwin Gödel Machine-style code-level self-modification.
// Instead of only adapting numeric parameters, SelfModifier rewrites
// the kernel's DSL programs: search heuristics, fitness functions,
// and neighborhood strategies are represented as ProgramNodes that
// can be mutated and evolved.
class SelfModifier {
public:
    struct Config {
        std::size_t candidates_per_round = 10;
        std::size_t max_eval_steps       = 50;
        Real improvement_threshold       = 0.01;
        unsigned seed                    = 42;
    };

    enum class StrategyKind {
        FitnessWeighting,
        NeighborhoodBias,
        ExplorationPolicy,
        // Enhanced self-referential: modify more kernel aspects
        GoalFunction,         // how goals are selected
        RewardShaping,        // intrinsic reward weighting
        MutationStrategy,     // which mutation operators to prefer
        SimplificationRule,   // program simplification heuristics
    };

    struct Strategy {
        StrategyKind kind;
        ProgramPtr program;
        Real fitness = 0.0;
    };

    SelfModifier();
    explicit SelfModifier(Config config);

    // Attempt to improve a strategy by evolving its program representation
    // Returns true if an improvement was found and applied
    bool try_modify(StrategyKind kind,
                    std::function<Real(const ProgramPtr&)> eval_fn);

    // Get current strategy program for a given kind
    [[nodiscard]] const ProgramPtr& current_strategy(StrategyKind kind) const;

    // Get modification count
    [[nodiscard]] std::size_t modification_count() const { return modifications_; }

    // Get all strategies
    [[nodiscard]] const std::vector<Strategy>& strategies() const {
        return strategies_;
    }

    // Record a strategy and its fitness for archive
    void record_strategy(StrategyKind kind, ProgramPtr prog, Real fitness);

    // Get the archive of all tried strategies
    [[nodiscard]] const Archive& strategy_archive() const { return archive_; }

    // Batch modification: try to improve all strategies in one round
    // Returns number of strategies improved
    std::size_t evolve_all(std::function<Real(StrategyKind, const ProgramPtr&)> eval_fn);

    // Get all strategy kinds
    static std::vector<StrategyKind> all_kinds();

private:
    Config config_;
    std::vector<Strategy> strategies_;
    Archive archive_;
    symbolic_descent::ProgramSpace space_;
    std::mt19937 rng_;
    std::size_t modifications_ = 0;

    std::size_t find_strategy_idx(StrategyKind kind) const;
    void ensure_strategy_exists(StrategyKind kind);
};

} // namespace uik::meta_evolution
