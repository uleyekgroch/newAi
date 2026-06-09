#pragma once

#include "common/types.hpp"
#include <deque>
#include <vector>

namespace uik::world_model {

// Information-theoretic novelty detection.
// Combines kNN distance with information gain estimation:
// novelty = kNN_distance * (1 + prediction_surprise)
// where prediction_surprise = KL-divergence proxy from running distribution.
class NoveltyDetector {
public:
    explicit NoveltyDetector(std::size_t window_size = 100, std::size_t k = 5);

    [[nodiscard]] Real score(const State& state) const;
    void observe(const State& state);
    void reset();

    [[nodiscard]] std::size_t window_size() const { return window_size_; }
    [[nodiscard]] Real information_gain() const { return last_info_gain_; }

private:
    std::size_t window_size_;
    std::size_t k_;
    std::deque<State> history_;

    // Running distribution stats for information-theoretic component
    std::vector<Real> running_mean_;
    std::vector<Real> running_var_;
    std::size_t obs_count_ = 0;
    mutable Real last_info_gain_ = 0.0;

    Real distance(const State& a, const State& b) const;
    Real prediction_surprise(const State& state) const;
    void update_distribution(const State& state);
};

} // namespace uik::world_model
