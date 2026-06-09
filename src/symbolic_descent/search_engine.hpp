#pragma once

#include "common/interfaces.hpp"
#include "symbolic_descent/program_space.hpp"
#include "symbolic_descent/mdl_evaluator.hpp"

namespace uik::symbolic_descent {

// Symbolic Descent: searches the discrete program space for the shortest
// program that explains the given input-output data pairs.
// Combines evolutionary search with hill-climbing refinement.
class SearchEngine final : public ISearchEngine {
public:
    struct Config {
        std::size_t population_size   = 50;
        std::size_t neighborhood_size = 10;
        std::size_t elite_count       = 5;
        Real lambda                   = 1.0;
        int max_depth                 = 4;
        unsigned seed                 = 42;
        // Beam search parameters
        std::size_t beam_width        = 8;
        std::size_t beam_expansions   = 5;   // expansions per beam candidate
        bool use_beam_search          = true;
    };

    SearchEngine();
    explicit SearchEngine(Config config);

    std::optional<ProgramPtr> search(const Dataset& data,
                                      std::size_t max_iterations) override;

    // Access last search statistics
    [[nodiscard]] Real last_best_score() const { return last_best_score_; }
    [[nodiscard]] std::size_t iterations_used() const { return iterations_used_; }

private:
    Config config_;
    ProgramSpace space_;
    MdlEvaluator evaluator_;
    Real last_best_score_ = 0.0;
    std::size_t iterations_used_ = 0;

    struct Candidate {
        ProgramPtr program;
        Real score = -std::numeric_limits<Real>::infinity();
    };

    std::vector<Candidate> initialize_population(const Dataset& data);
    void evolve_generation(std::vector<Candidate>& population,
                            const Dataset& data);
    void sort_by_score(std::vector<Candidate>& population);

    // Beam search: expand best candidates with type-directed operators
    std::optional<ProgramPtr> beam_search(const Dataset& data,
                                           std::size_t max_iterations);
    // Type-directed synthesis: generate candidates matching output structure
    std::vector<ProgramPtr> type_directed_expand(const ProgramPtr& base,
                                                  const Dataset& data);
};

} // namespace uik::symbolic_descent
