#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <vector>
#include <optional>
#include <random>

namespace uik::meta_evolution {

// Archive of programs (Darwin Gödel Machine style).
// Maintains a diverse collection of candidate solutions,
// not just the single best — enabling open-ended exploration.
class Archive {
public:
    struct Entry {
        ProgramPtr program;
        Real fitness    = 0.0;
        Real novelty    = 0.0;
        std::size_t generation = 0;
    };

    explicit Archive(std::size_t max_size = 200, unsigned seed = 42);

    void add(ProgramPtr program, Real fitness, Real novelty,
             std::size_t generation);

    // Sample with probability proportional to fitness + novelty
    [[nodiscard]] std::optional<Entry> sample() const;

    // Sample top-k by fitness
    [[nodiscard]] std::vector<Entry> top_k(std::size_t k) const;

    // Best entry by fitness
    [[nodiscard]] std::optional<Entry> best() const;

    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    [[nodiscard]] bool empty() const { return entries_.empty(); }

    // Diversity metric: average pairwise description-length difference
    [[nodiscard]] Real diversity() const;

private:
    std::size_t max_size_;
    mutable std::mt19937 rng_;
    std::vector<Entry> entries_;

    void prune();
};

} // namespace uik::meta_evolution
