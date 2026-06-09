#include "meta_evolution/parameter_adapter.hpp"

namespace uik::meta_evolution {

ParameterAdapter::ParameterAdapter() : ParameterAdapter(Config{}) {}

ParameterAdapter::ParameterAdapter(Config config) : config_(config) {}

ParameterAdapter::KernelParams ParameterAdapter::adapt(
        const PerformanceSnapshot& snapshot) {
    history_.push_back(snapshot);
    if (history_.size() > config_.window_size) {
        history_.pop_front();
    }

    if (history_.size() < 3) return params_;

    ++adaptations_;

    // Compute trends from recent history
    std::deque<Real> rewards, compressions, novelties, errors;
    for (const auto& h : history_) {
        rewards.push_back(h.reward);
        compressions.push_back(h.compression_prog);
        novelties.push_back(h.novelty);
        errors.push_back(h.prediction_error);
    }

    Real r_trend = compute_trend(rewards);
    Real c_trend = compute_trend(compressions);
    Real n_trend = compute_trend(novelties);
    Real e_trend = compute_trend(errors);

    adapt_learning_rate(r_trend, e_trend);
    adapt_exploration(n_trend, r_trend);
    adapt_curiosity(c_trend, r_trend);

    return params_;
}

Real ParameterAdapter::reward_trend() const {
    if (history_.size() < 3) return 0.0;
    std::deque<Real> rewards;
    for (const auto& h : history_) rewards.push_back(h.reward);
    return compute_trend(rewards);
}

Real ParameterAdapter::compression_trend() const {
    if (history_.size() < 3) return 0.0;
    std::deque<Real> comp;
    for (const auto& h : history_) comp.push_back(h.compression_prog);
    return compute_trend(comp);
}

Real ParameterAdapter::compute_trend(const std::deque<Real>& values) const {
    if (values.size() < 3) return 0.0;

    // Simple linear regression slope
    auto n = static_cast<Real>(values.size());
    Real sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        Real x = static_cast<Real>(i);
        sum_x += x;
        sum_y += values[i];
        sum_xy += x * values[i];
        sum_xx += x * x;
    }
    Real denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) return 0.0;
    return (n * sum_xy - sum_x * sum_y) / denom;
}

void ParameterAdapter::adapt_learning_rate(Real reward_trend, Real error_trend) {
    // If prediction error is decreasing → model is learning, keep LR
    // If error is increasing or stagnant → increase LR to learn faster
    // If reward is decreasing → slow down to stabilize
    if (error_trend > 0.0) {
        params_.learning_rate *= (1.0 + config_.adaptation_rate);
    } else if (reward_trend < 0.0 && error_trend < -0.01) {
        params_.learning_rate *= (1.0 - config_.adaptation_rate * 0.5);
    }

    params_.learning_rate = std::clamp(params_.learning_rate,
                                        config_.min_learning_rate,
                                        config_.max_learning_rate);
}

void ParameterAdapter::adapt_exploration(Real novelty_trend, Real reward_trend) {
    // If novelty is decreasing → stuck in familiar territory → explore more
    // If novelty is high but reward is not improving → explore less, exploit more
    if (novelty_trend < -0.001) {
        params_.exploration_bonus *= (1.0 + config_.adaptation_rate);
    } else if (novelty_trend > 0.01 && reward_trend < 0.0) {
        params_.exploration_bonus *= (1.0 - config_.adaptation_rate);
    }

    params_.exploration_bonus = std::clamp(params_.exploration_bonus,
                                            config_.min_exploration,
                                            config_.max_exploration);
}

void ParameterAdapter::adapt_curiosity(Real compression_trend, Real reward_trend) {
    // If compression progress is positive → learning is happening → maintain curiosity
    // If compression stalls → reduce curiosity weight, focus on external reward
    if (compression_trend > 0.001) {
        params_.curiosity_weight = std::min(0.8,
            params_.curiosity_weight + config_.adaptation_rate * 0.2);
    } else if (compression_trend < -0.001 && reward_trend < 0.0) {
        params_.curiosity_weight = std::max(0.1,
            params_.curiosity_weight - config_.adaptation_rate * 0.2);
    }
}

} // namespace uik::meta_evolution
