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
        cumulative_error_ += error;
        dynamics_.learn(last_state_, last_action_, current, config_.learning_rate);
    }

    novelty_.observe(current);
    last_state_ = current;
    ++update_count_;
}

Real WorldModel::compression_progress() const {
    if (update_count_ < 2) return 0.0;
    // Compression progress = reduction in average prediction error
    Real avg_current = cumulative_error_ / static_cast<Real>(update_count_);
    Real avg_prev = prev_cumulative_error_ / static_cast<Real>(update_count_ - 1);
    return std::max(0.0, avg_prev - avg_current);
}

} // namespace uik::world_model
