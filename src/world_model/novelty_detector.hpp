#pragma once

#include "common/types.hpp"
#include <deque>

namespace uik::world_model {

// Detects novelty by comparing incoming states to a sliding window of recent states.
// Novelty score = average distance to K nearest neighbors in the window.
class NoveltyDetector {
public:
    explicit NoveltyDetector(std::size_t window_size = 100, std::size_t k = 5);

    [[nodiscard]] Real score(const State& state) const;
    void observe(const State& state);
    void reset();

    [[nodiscard]] std::size_t window_size() const { return window_size_; }

private:
    std::size_t window_size_;
    std::size_t k_;
    std::deque<State> history_;

    Real distance(const State& a, const State& b) const;
};

} // namespace uik::world_model
