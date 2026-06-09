#pragma once

#include "common/types.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace uik::meta_evolution {

// Self-modification engine: adjusts kernel parameters based on performance.
// Implements the "Darwin Gödel Machine" principle — the system modifies
// its own configuration to improve future performance.
class ParameterAdapter {
public:
    struct Config {
        std::size_t window_size   = 20;   // performance window for trend analysis
        Real adaptation_rate      = 0.05; // how fast parameters change
        Real min_learning_rate    = 0.001;
        Real max_learning_rate    = 0.1;
        Real min_exploration      = 0.01;
        Real max_exploration      = 0.5;
    };

    struct KernelParams {
        Real learning_rate     = 0.01;
        Real exploration_bonus = 0.1;
        Real curiosity_weight  = 0.5;
        Real lambda            = 1.0;  // MDL trade-off
    };

    struct PerformanceSnapshot {
        Real reward           = 0.0;
        Real compression_prog = 0.0;
        Real novelty          = 0.0;
        Real prediction_error = 0.0;
    };

    ParameterAdapter();
    explicit ParameterAdapter(Config config);

    // Record a performance snapshot and adapt parameters
    KernelParams adapt(const PerformanceSnapshot& snapshot);

    // Get current parameters
    [[nodiscard]] const KernelParams& current_params() const { return params_; }

    // Get adaptation statistics
    [[nodiscard]] std::size_t adaptations_count() const { return adaptations_; }
    [[nodiscard]] Real reward_trend() const;
    [[nodiscard]] Real compression_trend() const;

private:
    Config config_;
    KernelParams params_;
    std::deque<PerformanceSnapshot> history_;
    std::size_t adaptations_ = 0;

    Real compute_trend(const std::deque<Real>& values) const;
    void adapt_learning_rate(Real reward_trend, Real error_trend);
    void adapt_exploration(Real novelty_trend, Real reward_trend);
    void adapt_curiosity(Real compression_trend, Real reward_trend);
};

} // namespace uik::meta_evolution
