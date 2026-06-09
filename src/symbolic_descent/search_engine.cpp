#include "symbolic_descent/search_engine.hpp"
#include <algorithm>

namespace uik::symbolic_descent {

SearchEngine::SearchEngine() : SearchEngine(Config{}) {}

SearchEngine::SearchEngine(Config config)
    : config_(config)
    , space_(config.max_depth, 10, 5, config.seed)
    , evaluator_(config.lambda)
{}

std::optional<ProgramPtr> SearchEngine::search(const Dataset& data,
                                                std::size_t max_iterations) {
    if (data.empty()) return std::nullopt;

    auto population = initialize_population(data);
    iterations_used_ = 0;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        ++iterations_used_;
        evolve_generation(population, data);
        sort_by_score(population);

        const auto& best = population.front();
        last_best_score_ = best.score;

        if (evaluator_.is_perfect_fit(best.program, data)) {
            return best.program;
        }
    }

    sort_by_score(population);
    last_best_score_ = population.front().score;
    return population.front().program;
}

std::vector<SearchEngine::Candidate>
SearchEngine::initialize_population(const Dataset& data) {
    std::vector<Candidate> population;
    population.reserve(config_.population_size);

    // Always include Identity as a baseline
    auto id_prog = identity();
    population.push_back({id_prog, evaluator_.score(id_prog, data)});

    for (std::size_t i = 1; i < config_.population_size; ++i) {
        auto prog = space_.random_program();
        population.push_back({prog, evaluator_.score(prog, data)});
    }
    return population;
}

void SearchEngine::evolve_generation(std::vector<Candidate>& population,
                                      const Dataset& data) {
    sort_by_score(population);

    std::vector<Candidate> next_gen;
    next_gen.reserve(config_.population_size);

    // Elitism: keep top candidates
    for (std::size_t i = 0; i < config_.elite_count && i < population.size(); ++i) {
        next_gen.push_back(population[i]);
    }

    // Generate offspring from elites
    for (std::size_t i = 0; i < config_.elite_count && i < population.size(); ++i) {
        auto neighbors = space_.neighborhood(population[i].program,
                                              config_.neighborhood_size);
        for (auto& neighbor : neighbors) {
            Real s = evaluator_.score(neighbor, data);
            next_gen.push_back({std::move(neighbor), s});
        }
    }

    // Fill remaining with random
    while (next_gen.size() < config_.population_size) {
        auto prog = space_.random_program();
        next_gen.push_back({prog, evaluator_.score(prog, data)});
    }

    sort_by_score(next_gen);
    next_gen.resize(config_.population_size);
    population = std::move(next_gen);
}

void SearchEngine::sort_by_score(std::vector<Candidate>& population) {
    std::sort(population.begin(), population.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.score > b.score;
              });
}

} // namespace uik::symbolic_descent
