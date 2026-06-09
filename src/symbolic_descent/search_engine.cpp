#include "symbolic_descent/search_engine.hpp"
#include "symbolic_descent/dsl.hpp"
#include <algorithm>
#include <cmath>

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

    // Run beam search in parallel with evolutionary search, return best
    std::optional<ProgramPtr> beam_result;
    if (config_.use_beam_search) {
        beam_result = beam_search(data, max_iterations / 2);
    }

    auto population = initialize_population(data);
    iterations_used_ = 0;

    // If beam search found something, seed the population with it
    if (beam_result.has_value()) {
        Real bs = evaluator_.score(*beam_result, data);
        population.push_back({*beam_result, bs});
    }

    std::size_t evo_iters = config_.use_beam_search ? max_iterations / 2 : max_iterations;
    for (std::size_t iter = 0; iter < evo_iters; ++iter) {
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

std::optional<ProgramPtr> SearchEngine::beam_search(const Dataset& data,
                                                      std::size_t max_iterations) {
    // Initialize beam with diverse seeds
    std::vector<Candidate> beam;
    beam.reserve(config_.beam_width);

    // Seed with Identity, simple primitives, and type-directed candidates
    auto id_prog = identity();
    beam.push_back({id_prog, evaluator_.score(id_prog, data)});

    for (std::size_t i = 1; i < config_.beam_width; ++i) {
        auto prog = space_.random_program();
        beam.push_back({prog, evaluator_.score(prog, data)});
    }
    sort_by_score(beam);

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        std::vector<Candidate> expansions;

        for (const auto& candidate : beam) {
            // Type-directed expansion: try mutations informed by data
            auto directed = type_directed_expand(candidate.program, data);
            for (auto& prog : directed) {
                Real s = evaluator_.score(prog, data);
                expansions.push_back({std::move(prog), s});
            }

            // Also try standard guided operators
            auto ta = space_.type_aware_mutate(candidate.program);
            expansions.push_back({ta, evaluator_.score(ta, data)});

            auto pp = space_.param_perturbation(candidate.program);
            expansions.push_back({pp, evaluator_.score(pp, data)});

            auto si = space_.simplify(candidate.program);
            expansions.push_back({si, evaluator_.score(si, data)});

            auto te = space_.template_guided_edit(candidate.program);
            expansions.push_back({te, evaluator_.score(te, data)});
        }

        // Merge beam + expansions, keep top beam_width
        for (auto& c : beam) expansions.push_back(std::move(c));
        sort_by_score(expansions);
        beam.clear();
        for (std::size_t i = 0; i < config_.beam_width && i < expansions.size(); ++i) {
            beam.push_back(std::move(expansions[i]));
        }

        if (evaluator_.is_perfect_fit(beam.front().program, data)) {
            return beam.front().program;
        }
    }

    if (!beam.empty()) {
        return beam.front().program;
    }
    return std::nullopt;
}

std::vector<ProgramPtr> SearchEngine::type_directed_expand(
    const ProgramPtr& base, const Dataset& data)
{
    std::vector<ProgramPtr> results;
    if (data.empty() || !base) return results;

    // Analyze the first data pair to understand I/O relationship
    const auto& [input, expected] = data[0];

    // Check if output is a rotation/flip/translation of input
    DSL dsl;
    std::vector<OpKind> single_ops = {
        OpKind::Rotate90, OpKind::FlipH, OpKind::FlipV, OpKind::Identity
    };
    for (auto op : single_ops) {
        auto prog = std::make_shared<ProgramNode>();
        prog->kind = op;
        try {
            Tensor out = dsl.execute(prog, input);
            // If this single op gets us closer, try composing with base
            auto composed = compose(base, prog);
            results.push_back(composed);
            auto composed2 = compose(prog, base);
            results.push_back(composed2);
        } catch (...) {}
    }

    // Try wrapping base in a Repeat for iterative refinement
    for (int rep : {2, 3}) {
        auto repeat = std::make_shared<ProgramNode>();
        repeat->kind = OpKind::Repeat;
        repeat->param1 = rep;
        repeat->children.push_back(base);
        results.push_back(repeat);
    }

    // Try adding post-processing (Add, Threshold, Filter)
    for (int val : {-1, 0, 1}) {
        auto add = std::make_shared<ProgramNode>();
        add->kind = OpKind::Add;
        add->param1 = val;
        results.push_back(compose(base, add));
    }

    return results;
}

} // namespace uik::symbolic_descent
