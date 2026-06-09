#include "world_model/world_model.hpp"
#include <cmath>

namespace uik::world_model {

WorldModel::WorldModel(Config config)
    : config_(config)
    , encoder_(config.input_dim, config.latent_dim, config.seed)
    , dynamics_(config.latent_dim, config.action_space, config.seed)
    , novelty_(config.novelty_window, config.novelty_k)
{}

State WorldModel::encode(const Observation& obs) {
    return encoder_.encode(obs);
}

State WorldModel::predict_next(const State& current, const Action& action) {
    return dynamics_.predict(current, action);
}

Real WorldModel::compute_novelty(const State& state) {
    return novelty_.score(state);
}

void WorldModel::update(const Observation& obs) {
    State current = encoder_.encode(obs);

    if (has_previous_) {
        State predicted = dynamics_.predict(last_state_, last_action_);
        Real error = dynamics_.prediction_error(predicted, current);

        // Track compression progress: store previous average before updating
        if (prediction_count_ > 0) {
            prev_avg_error_ = cumulative_error_ / static_cast<Real>(prediction_count_);
            prev_avg_log_loss_ = cumulative_log_loss_ / static_cast<Real>(prediction_count_);
        }
        cumulative_error_ += error;
        // Log-likelihood proxy: -log(1 + error^2) — proper probabilistic measure
        Real log_loss = std::log(1.0 + error * error);
        cumulative_log_loss_ += log_loss;
        ++prediction_count_;

        dynamics_.learn(last_state_, last_action_, current, config_.learning_rate);
    }

    // Online encoder learning: reconstruct observation from latent
    encoder_.learn(obs, config_.learning_rate);

    novelty_.observe(current);
    last_state_ = current;
    has_previous_ = true;
    ++update_count_;
}

void WorldModel::record_action(const Action& action) {
    last_action_ = action;
}

Real WorldModel::compression_progress() const {
    if (prediction_count_ < 2) return 0.0;
    // Log-likelihood based compression progress:
    // Measures reduction in negative log-likelihood (bits saved)
    Real current_avg_log = cumulative_log_loss_ / static_cast<Real>(prediction_count_);
    Real log_progress = std::max(0.0, prev_avg_log_loss_ - current_avg_log);
    // Also keep the simple error-based measure as a component
    Real current_avg = cumulative_error_ / static_cast<Real>(prediction_count_);
    Real error_progress = std::max(0.0, prev_avg_error_ - current_avg);
    // Combined: weighted sum of both signals
    return 0.6 * log_progress + 0.4 * error_progress;
}

Real WorldModel::prediction_error_rate() const {
    if (prediction_count_ == 0) return 0.0;
    return cumulative_error_ / static_cast<Real>(prediction_count_);
}

} // namespace uik::world_model
