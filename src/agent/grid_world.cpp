#include "agent/grid_world.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace uik::agent {

GridWorld::GridWorld() : GridWorld(Config{}) {}

GridWorld::GridWorld(Config config)
    : config_(config), rng_(config.seed)
{
    grid_.resize(config.rows * config.cols, 0.0);
    target_.resize(config.rows * config.cols, 0.0);
}

Observation GridWorld::reset() {
    step_count_ = 0;
    std::fill(grid_.begin(), grid_.end(), 0.0);

    // Initialize grid with a simple pattern
    std::uniform_int_distribution<int> dist(0, config_.num_colors - 1);
    for (auto& cell : grid_) {
        cell = static_cast<Real>(dist(rng_));
    }

    generate_target();
    return observe();
}

StepResult GridWorld::step(const Action& action) {
    ++step_count_;
    Dim total = config_.rows * config_.cols;

    switch (action.id) {
        case kRotate90: {
            // Transpose + reverse rows (90° clockwise)
            std::vector<Real> rotated(total);
            for (Dim r = 0; r < config_.rows; ++r) {
                for (Dim c = 0; c < config_.cols; ++c) {
                    Dim new_r = c;
                    Dim new_c = config_.rows - 1 - r;
                    if (new_r < config_.cols && new_c < config_.rows) {
                        rotated[new_r * config_.rows + new_c] =
                            grid_[r * config_.cols + c];
                    }
                }
            }
            grid_ = std::move(rotated);
            break;
        }
        case kFlipH: {
            for (Dim r = 0; r < config_.rows; ++r) {
                for (Dim c = 0; c < config_.cols / 2; ++c) {
                    std::swap(grid_[r * config_.cols + c],
                              grid_[r * config_.cols + (config_.cols - 1 - c)]);
                }
            }
            break;
        }
        case kFlipV: {
            for (Dim r = 0; r < config_.rows / 2; ++r) {
                for (Dim c = 0; c < config_.cols; ++c) {
                    std::swap(grid_[r * config_.cols + c],
                              grid_[(config_.rows - 1 - r) * config_.cols + c]);
                }
            }
            break;
        }
        case kShiftRight: {
            for (Dim r = 0; r < config_.rows; ++r) {
                Real last = grid_[r * config_.cols + config_.cols - 1];
                for (Dim c = config_.cols - 1; c > 0; --c) {
                    grid_[r * config_.cols + c] = grid_[r * config_.cols + c - 1];
                }
                grid_[r * config_.cols] = last;
            }
            break;
        }
        case kIncrement: {
            for (auto& cell : grid_) {
                cell = std::fmod(cell + 1.0,
                                  static_cast<Real>(config_.num_colors));
            }
            break;
        }
        default:
            break; // noop
    }

    Real reward = compute_reward();
    bool done = (step_count_ >= config_.max_steps) || (reward > -0.01);
    return {observe(), reward, done};
}

void GridWorld::generate_target() {
    // Target = grid transformed by a known sequence (agent must discover it)
    target_ = grid_;

    // Apply: flipH then increment
    for (Dim r = 0; r < config_.rows; ++r) {
        for (Dim c = 0; c < config_.cols / 2; ++c) {
            std::swap(target_[r * config_.cols + c],
                      target_[r * config_.cols + (config_.cols - 1 - c)]);
        }
    }
    for (auto& cell : target_) {
        cell = std::fmod(cell + 1.0, static_cast<Real>(config_.num_colors));
    }
}

Real GridWorld::compute_reward() const {
    Real dist = 0.0;
    for (std::size_t i = 0; i < grid_.size(); ++i) {
        Real d = grid_[i] - target_[i];
        dist += d * d;
    }
    return -std::sqrt(dist) / static_cast<Real>(grid_.size());
}

Observation GridWorld::observe() const {
    std::vector<Real> data;
    data.reserve(grid_.size() + target_.size());
    data.insert(data.end(), grid_.begin(), grid_.end());
    data.insert(data.end(), target_.begin(), target_.end());
    Dim n = data.size();
    return {Tensor({n}, std::move(data))};
}

} // namespace uik::agent
