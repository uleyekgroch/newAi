#include "meta_evolution/archive.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace uik::meta_evolution {

Archive::Archive(std::size_t max_size, unsigned seed)
    : max_size_(max_size), rng_(seed) {}

void Archive::add(ProgramPtr program, Real fitness, Real novelty,
                   std::size_t generation) {
    entries_.push_back({std::move(program), fitness, novelty, generation});
    if (entries_.size() > max_size_) {
        prune();
    }
}

std::optional<Archive::Entry> Archive::sample() const {
    if (entries_.empty()) return std::nullopt;

    // Compute sampling weights: fitness + novelty (shifted to be positive)
    std::vector<Real> weights(entries_.size());
    Real min_score = std::numeric_limits<Real>::max();
    for (const auto& e : entries_) {
        Real s = e.fitness + e.novelty;
        min_score = std::min(min_score, s);
    }
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        weights[i] = (entries_[i].fitness + entries_[i].novelty) - min_score + 1.0;
    }
    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    return entries_[dist(rng_)];
}

std::vector<Archive::Entry> Archive::top_k(std::size_t k) const {
    auto sorted = entries_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Entry& a, const Entry& b) {
                  return a.fitness > b.fitness;
              });
    if (sorted.size() > k) sorted.resize(k);
    return sorted;
}

std::optional<Archive::Entry> Archive::best() const {
    if (entries_.empty()) return std::nullopt;
    return *std::max_element(entries_.begin(), entries_.end(),
                              [](const Entry& a, const Entry& b) {
                                  return a.fitness < b.fitness;
                              });
}

Real Archive::diversity() const {
    if (entries_.size() < 2) return 0.0;
    Real total_diff = 0.0;
    std::size_t pairs = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        for (std::size_t j = i + 1; j < entries_.size(); ++j) {
            Real dl_i = static_cast<Real>(
                entries_[i].program->description_length());
            Real dl_j = static_cast<Real>(
                entries_[j].program->description_length());
            total_diff += std::abs(dl_i - dl_j);
            ++pairs;
        }
    }
    return total_diff / static_cast<Real>(pairs);
}

void Archive::prune() {
    // Remove lowest fitness entries, but keep diverse ones
    std::sort(entries_.begin(), entries_.end(),
              [](const Entry& a, const Entry& b) {
                  return (a.fitness + a.novelty) > (b.fitness + b.novelty);
              });
    entries_.resize(max_size_);
}

} // namespace uik::meta_evolution
