#pragma once

#include "meta_evolution/archive.hpp"
#include "meta_evolution/mutator.hpp"
#include "symbolic_descent/mdl_evaluator.hpp"
#include <functional>

namespace uik::meta_evolution {

// Orchestrates the evolutionary loop: select → mutate → evaluate → archive.
// This is the "self-improvement engine" of the kernel.
class EvolutionarySelector {
public:
    using FitnessFunc = std::function<Real(const ProgramPtr&)>;

    struct Config {
        std::size_t offspring_per_gen = 20;
        std::size_t archive_size     = 200;
        unsigned seed                = 42;
    };

    EvolutionarySelector();
    explicit EvolutionarySelector(Config config);

    // Run one generation of evolution
    void evolve_once(FitnessFunc fitness_fn);

    // Run multiple generations
    void evolve(FitnessFunc fitness_fn, std::size_t generations);

    // Seed the archive with initial programs
    void seed(const std::vector<ProgramPtr>& programs, FitnessFunc fitness_fn);

    [[nodiscard]] const Archive& archive() const { return archive_; }
    [[nodiscard]] std::size_t generation() const { return generation_; }

private:
    Config config_;
    Archive archive_;
    Mutator mutator_;
    std::size_t generation_ = 0;

    Real compute_novelty(const ProgramPtr& program) const;
};

} // namespace uik::meta_evolution
