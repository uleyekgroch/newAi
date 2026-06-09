#pragma once

#include "common/interfaces.hpp"
#include <random>
#include <vector>

namespace uik::agent {

// 2D GridWorld: agent navigates a grid, discovers patterns, earns reward
// for matching a target pattern. Richer than 1D for testing world model +
// symbolic descent (grid transforms are natural DSL operations).
class GridWorld final : public IEnvironment {
public:
    struct Config {
        Dim rows            = 5;
        Dim cols            = 5;
        int num_colors      = 10;
        std::size_t max_steps = 200;
        unsigned seed       = 42;
    };

    GridWorld();
    explicit GridWorld(Config config);

    Observation reset() override;
    StepResult  step(const Action& action) override;
    int         action_space_size() const override { return 6; }

    // Access for testing
    [[nodiscard]] const std::vector<Real>& grid() const { return grid_; }
    [[nodiscard]] const std::vector<Real>& target() const { return target_; }
    [[nodiscard]] Dim rows() const { return config_.rows; }
    [[nodiscard]] Dim cols() const { return config_.cols; }

    // Actions: 0=noop, 1=rotate90, 2=flipH, 3=flipV, 4=shift_right, 5=increment
    static constexpr int kNoop       = 0;
    static constexpr int kRotate90   = 1;
    static constexpr int kFlipH      = 2;
    static constexpr int kFlipV      = 3;
    static constexpr int kShiftRight = 4;
    static constexpr int kIncrement  = 5;

private:
    Config config_;
    std::mt19937 rng_;
    std::vector<Real> grid_;
    std::vector<Real> target_;
    std::size_t step_count_ = 0;

    void generate_target();
    Real compute_reward() const;
    Observation observe() const;
};

} // namespace uik::agent
