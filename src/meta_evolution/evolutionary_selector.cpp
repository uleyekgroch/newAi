#include "meta_evolution/evolutionary_selector.hpp"
#include <cmath>

namespace uik::meta_evolution {

EvolutionarySelector::EvolutionarySelector()
    : EvolutionarySelector(Config{}) {}

EvolutionarySelector::EvolutionarySelector(Config config)
    : config_(config)
    , archive_(config.archive_size, config.seed)
    , mutator_(config.seed)
{}

void EvolutionarySelector::evolve_once(FitnessFunc fitness_fn) {
    ++generation_;

    if (archive_.empty()) {
        // Bootstrap with identity program
        auto id = identity();
        Real fit = fitness_fn(id);
        archive_.add(id, fit, 1.0, generation_);
        return;
    }

    for (std::size_t i = 0; i < config_.offspring_per_gen; ++i) {
        auto parent = archive_.sample();
        if (!parent) continue;

        ProgramPtr offspring;
        auto second = archive_.sample();
        if (second && (i % 3 == 0)) {
            offspring = mutator_.crossover(parent->program, second->program);
        } else {
            offspring = mutator_.mutate(parent->program);
        }

        Real fit = fitness_fn(offspring);
        Real nov = compute_novelty(offspring);
        archive_.add(std::move(offspring), fit, nov, generation_);
    }
}

void EvolutionarySelector::evolve(FitnessFunc fitness_fn,
                                    std::size_t generations) {
    for (std::size_t g = 0; g < generations; ++g) {
        evolve_once(fitness_fn);
    }
}

void EvolutionarySelector::seed(const std::vector<ProgramPtr>& programs,
                                  FitnessFunc fitness_fn) {
    for (const auto& prog : programs) {
        Real fit = fitness_fn(prog);
        Real nov = compute_novelty(prog);
        archive_.add(prog, fit, nov, 0);
    }
}

Real EvolutionarySelector::compute_novelty(const ProgramPtr& program) const {
    if (archive_.empty()) return 1.0;
    Real dl = static_cast<Real>(program->description_length());
    // Novelty = how different this program's structure is from archive average
    Real total = 0.0;
    auto top = archive_.top_k(10);
    for (const auto& entry : top) {
        Real other_dl = static_cast<Real>(entry.program->description_length());
        total += std::abs(dl - other_dl);
    }
    return total / static_cast<Real>(std::max(top.size(), std::size_t{1}));
}

} // namespace uik::meta_evolution
