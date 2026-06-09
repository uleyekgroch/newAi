#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <vector>
#include <functional>

namespace uik::meta_evolution {

// Safety alignment constraints on self-modification.
// Prevents the kernel from self-modifying in ways that:
// 1. Exceed resource budgets (computation, memory, depth)
// 2. Violate invariants (non-negative rewards, bounded parameters)
// 3. Remove safety checks (cannot modify the SafetyGuard itself)
// 4. Diverge too far from proven-safe configurations
class SafetyGuard {
public:
    struct Config {
        std::size_t max_program_depth    = 10;
        std::size_t max_program_nodes    = 500;
        std::size_t max_loop_iterations  = 100;
        Real max_learning_rate           = 1.0;
        Real min_learning_rate           = 1e-6;
        Real max_exploration_bonus       = 5.0;
        Real max_curiosity_weight        = 2.0;
        Real max_modification_rate       = 0.5;
        std::size_t rollback_window      = 10;
        Real regression_threshold        = -0.1;
    };

    struct ValidationResult {
        bool safe = true;
        std::string reason;
    };

    SafetyGuard();
    explicit SafetyGuard(Config config);

    // Validate a program before it's used
    [[nodiscard]] ValidationResult validate_program(const ProgramPtr& prog) const;

    // Validate parameter changes before applying
    [[nodiscard]] ValidationResult validate_params(
        Real learning_rate, Real exploration_bonus, Real curiosity_weight) const;

    // Check if self-modification should be allowed given performance history
    [[nodiscard]] ValidationResult validate_modification(
        const std::vector<Real>& recent_rewards) const;

    // Clamp parameters to safe ranges
    void clamp_params(Real& learning_rate, Real& exploration_bonus,
                      Real& curiosity_weight) const;

    // Record performance for rollback detection
    void record_performance(Real reward);

    // Check if we should rollback to previous config
    [[nodiscard]] bool should_rollback() const;

    // Get the modification count that was rejected
    [[nodiscard]] std::size_t rejections() const { return rejections_; }

    // Get total validations performed
    [[nodiscard]] std::size_t validations() const { return validations_; }

private:
    Config config_;
    std::vector<Real> performance_history_;
    mutable std::size_t rejections_ = 0;
    mutable std::size_t validations_ = 0;

    bool check_program_depth(const ProgramPtr& prog) const;
    bool check_program_size(const ProgramPtr& prog) const;
    bool check_loop_bounds(const ProgramPtr& prog) const;
};

} // namespace uik::meta_evolution
