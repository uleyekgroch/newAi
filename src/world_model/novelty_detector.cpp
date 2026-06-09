#include "world_model/novelty_detector.hpp"
#include <algorithm>
#include <limits>

namespace uik::world_model {

NoveltyDetector::NoveltyDetector(std::size_t window_size, std::size_t k)
    : window_size_(window_size), k_(k) {}

Real NoveltyDetector::score(const State& state) const {
    if (history_.empty()) return 1.0; // first observation is always novel

    std::vector<Real> distances;
    distances.reserve(history_.size());
    for (const auto& past : history_) {
        distances.push_back(distance(state, past));
    }
    std::sort(distances.begin(), distances.end());

    std::size_t count = std::min(k_, distances.size());
    Real sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += distances[i];
    }
    return sum / static_cast<Real>(count);
}

void NoveltyDetector::observe(const State& state) {
    history_.push_back(state);
    if (history_.size() > window_size_) {
        history_.pop_front();
    }
}

void NoveltyDetector::reset() {
    history_.clear();
}

Real NoveltyDetector::distance(const State& a, const State& b) const {
    if (a.latent.empty() || b.latent.empty()) return 0.0;
    Tensor diff = a.latent - b.latent;
    return diff.l2_norm();
}

} // namespace uik::world_model
