#include "world_model/novelty_detector.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

namespace uik::world_model {

NoveltyDetector::NoveltyDetector(std::size_t window_size, std::size_t k)
    : window_size_(window_size), k_(k) {}

Real NoveltyDetector::score(const State& state) const {
    if (history_.empty()) return 1.0;

    // kNN distance component
    std::vector<Real> distances;
    distances.reserve(history_.size());
    for (const auto& past : history_) {
        distances.push_back(distance(state, past));
    }
    std::sort(distances.begin(), distances.end());

    std::size_t count = std::min(k_, distances.size());
    Real knn_sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        knn_sum += distances[i];
    }
    Real knn_novelty = knn_sum / static_cast<Real>(count);

    // Information-theoretic component: prediction surprise
    Real surprise = prediction_surprise(state);
    last_info_gain_ = surprise;

    // Combined novelty: kNN * (1 + surprise)
    return knn_novelty * (1.0 + surprise);
}

Real NoveltyDetector::prediction_surprise(const State& state) const {
    if (obs_count_ < 2 || state.latent.empty()) return 0.0;

    // Mahalanobis-like distance from running distribution
    // surprise = sum((x_i - mean_i)^2 / var_i) / dim
    Real total = 0.0;
    Dim dim = state.latent.flat_size();
    Dim valid_dim = std::min(dim, static_cast<Dim>(running_mean_.size()));

    for (Dim i = 0; i < valid_dim; ++i) {
        Real diff = state.latent.at(i) - running_mean_[i];
        Real var = running_var_[i] + 1e-6;  // avoid division by zero
        total += (diff * diff) / var;
    }

    // Normalize and bound
    if (valid_dim > 0) {
        total /= static_cast<Real>(valid_dim);
    }
    // Convert to [0, 1) range using sigmoid-like transform
    return 1.0 - 1.0 / (1.0 + total);
}

void NoveltyDetector::observe(const State& state) {
    history_.push_back(state);
    if (history_.size() > window_size_) {
        history_.pop_front();
    }
    update_distribution(state);
}

void NoveltyDetector::update_distribution(const State& state) {
    if (state.latent.empty()) return;
    Dim dim = state.latent.flat_size();

    if (running_mean_.empty()) {
        running_mean_.assign(dim, 0.0);
        running_var_.assign(dim, 1.0);
    }

    ++obs_count_;
    Real n = static_cast<Real>(obs_count_);

    for (Dim i = 0; i < dim && i < running_mean_.size(); ++i) {
        Real x = state.latent.at(i);
        Real old_mean = running_mean_[i];
        // Welford's online algorithm
        running_mean_[i] += (x - old_mean) / n;
        running_var_[i] += (x - old_mean) * (x - running_mean_[i]);
    }
}

void NoveltyDetector::reset() {
    history_.clear();
    running_mean_.clear();
    running_var_.clear();
    obs_count_ = 0;
    last_info_gain_ = 0.0;
}

Real NoveltyDetector::distance(const State& a, const State& b) const {
    if (a.latent.empty() || b.latent.empty()) return 0.0;
    Tensor diff = a.latent - b.latent;
    return diff.l2_norm();
}

} // namespace uik::world_model
