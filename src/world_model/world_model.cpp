#include "world_model/world_model.hpp"

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
        }
        cumulative_error_ += error;
        ++prediction_count_;

        dynamics_.learn(last_state_, last_action_, current, config_.learning_rate);
    }

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
    Real current_avg = cumulative_error_ / static_cast<Real>(prediction_count_);
    return std::max(0.0, prev_avg_error_ - current_avg);
}

Real WorldModel::prediction_error_rate() const {
    if (prediction_count_ == 0) return 0.0;
    return cumulative_error_ / static_cast<Real>(prediction_count_);
}

} // namespace uik::world_model
