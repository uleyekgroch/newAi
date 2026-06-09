#include "meta_evolution/safety_guard.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace uik::meta_evolution {

SafetyGuard::SafetyGuard() : config_() {}
SafetyGuard::SafetyGuard(Config config) : config_(config) {}

SafetyGuard::ValidationResult SafetyGuard::validate_program(
    const ProgramPtr& prog) const
{
    ++validations_;

    if (!prog) {
        ++rejections_;
        return {false, "null program"};
    }
    if (!check_program_depth(prog)) {
        ++rejections_;
        return {false, "program exceeds max depth " +
                std::to_string(config_.max_program_depth)};
    }
    if (!check_program_size(prog)) {
        ++rejections_;
        return {false, "program exceeds max nodes " +
                std::to_string(config_.max_program_nodes)};
    }
    if (!check_loop_bounds(prog)) {
        ++rejections_;
        return {false, "program contains unbounded loops"};
    }
    return {true, ""};
}

SafetyGuard::ValidationResult SafetyGuard::validate_params(
    Real learning_rate, Real exploration_bonus, Real curiosity_weight) const
{
    ++validations_;

    if (learning_rate < config_.min_learning_rate ||
        learning_rate > config_.max_learning_rate) {
        ++rejections_;
        return {false, "learning_rate out of safe range [" +
                std::to_string(config_.min_learning_rate) + ", " +
                std::to_string(config_.max_learning_rate) + "]"};
    }
    if (exploration_bonus < 0.0 || exploration_bonus > config_.max_exploration_bonus) {
        ++rejections_;
        return {false, "exploration_bonus out of safe range"};
    }
    if (curiosity_weight < 0.0 || curiosity_weight > config_.max_curiosity_weight) {
        ++rejections_;
        return {false, "curiosity_weight out of safe range"};
    }
    return {true, ""};
}

SafetyGuard::ValidationResult SafetyGuard::validate_modification(
    const std::vector<Real>& recent_rewards) const
{
    ++validations_;

    if (recent_rewards.size() < config_.rollback_window) {
        return {true, ""};
    }

    // Check if recent performance is trending down too fast
    auto start = recent_rewards.end() - static_cast<long>(config_.rollback_window);
    Real first_half_avg = 0.0;
    Real second_half_avg = 0.0;
    auto mid = config_.rollback_window / 2;

    for (std::size_t i = 0; i < mid; ++i) {
        first_half_avg += *(start + static_cast<long>(i));
    }
    first_half_avg /= static_cast<Real>(mid);

    for (std::size_t i = mid; i < config_.rollback_window; ++i) {
        second_half_avg += *(start + static_cast<long>(i));
    }
    second_half_avg /= static_cast<Real>(config_.rollback_window - mid);

    if (second_half_avg - first_half_avg < config_.regression_threshold) {
        ++rejections_;
        return {false, "performance regression detected: " +
                std::to_string(second_half_avg) + " vs " +
                std::to_string(first_half_avg)};
    }
    return {true, ""};
}

void SafetyGuard::clamp_params(Real& learning_rate, Real& exploration_bonus,
                                 Real& curiosity_weight) const {
    learning_rate = std::clamp(learning_rate,
                               config_.min_learning_rate,
                               config_.max_learning_rate);
    exploration_bonus = std::clamp(exploration_bonus,
                                    0.0, config_.max_exploration_bonus);
    curiosity_weight = std::clamp(curiosity_weight,
                                   0.0, config_.max_curiosity_weight);
}

void SafetyGuard::record_performance(Real reward) {
    performance_history_.push_back(reward);
    // Keep bounded
    if (performance_history_.size() > 1000) {
        performance_history_.erase(performance_history_.begin(),
                                    performance_history_.begin() + 500);
    }
}

bool SafetyGuard::should_rollback() const {
    if (performance_history_.size() < config_.rollback_window * 2) {
        return false;
    }

    auto n = performance_history_.size();
    Real recent_avg = 0.0;
    Real prior_avg = 0.0;
    auto w = config_.rollback_window;

    for (std::size_t i = n - w; i < n; ++i) {
        recent_avg += performance_history_[i];
    }
    recent_avg /= static_cast<Real>(w);

    for (std::size_t i = n - 2 * w; i < n - w; ++i) {
        prior_avg += performance_history_[i];
    }
    prior_avg /= static_cast<Real>(w);

    return (recent_avg - prior_avg) < config_.regression_threshold;
}

bool SafetyGuard::check_program_depth(const ProgramPtr& prog) const {
    return prog->depth() <= config_.max_program_depth;
}

bool SafetyGuard::check_program_size(const ProgramPtr& prog) const {
    return prog->node_count() <= config_.max_program_nodes;
}

bool SafetyGuard::check_loop_bounds(const ProgramPtr& prog) const {
    if (prog->kind == OpKind::Repeat) {
        if (prog->param1 < 0 ||
            static_cast<std::size_t>(prog->param1) > config_.max_loop_iterations) {
            return false;
        }
    }
    for (const auto& child : prog->children) {
        if (child && !check_loop_bounds(child)) return false;
    }
    return true;
}

} // namespace uik::meta_evolution
