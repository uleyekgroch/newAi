#include "world_model/dynamics_predictor.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace uik::world_model {

DynamicsPredictor::DynamicsPredictor(Dim latent_dim, int action_space_size,
                                      unsigned seed)
    : latent_dim_(latent_dim), action_space_size_(action_space_size)
{
    std::mt19937 rng(seed);
    Real scale = std::sqrt(2.0 / static_cast<Real>(latent_dim));
    std::normal_distribution<Real> dist(0.0, scale);

    transition_weights_.resize(static_cast<std::size_t>(action_space_size));
    transition_bias_.resize(static_cast<std::size_t>(action_space_size));

    for (int a = 0; a < action_space_size; ++a) {
        auto& w = transition_weights_[static_cast<std::size_t>(a)];
        auto& b = transition_bias_[static_cast<std::size_t>(a)];
        w.resize(latent_dim * latent_dim);
        b.resize(latent_dim, 0.0);
        for (auto& val : w) val = dist(rng);
    }
}

State DynamicsPredictor::predict(const State& current,
                                  const Action& action) const {
    if (action.id < 0 || action.id >= action_space_size_) {
        throw std::out_of_range("DynamicsPredictor::predict: invalid action id");
    }
    auto a_idx = static_cast<std::size_t>(action.id);
    const auto& w = transition_weights_[a_idx];
    const auto& b = transition_bias_[a_idx];

    std::vector<Real> next(latent_dim_, 0.0);
    for (Dim i = 0; i < latent_dim_; ++i) {
        Real sum = b[i];
        for (Dim j = 0; j < latent_dim_; ++j) {
            sum += w[i * latent_dim_ + j] * current.latent.at(j);
        }
        next[i] = std::tanh(sum);
    }
    return State{Tensor({latent_dim_}, std::move(next))};
}

void DynamicsPredictor::learn(const State& current, const Action& action,
                               const State& actual_next, Real learning_rate) {
    if (action.id < 0 || action.id >= action_space_size_) {
        throw std::out_of_range("DynamicsPredictor::learn: invalid action id");
    }
    State predicted = predict(current, action);
    auto a_idx = static_cast<std::size_t>(action.id);
    auto& w = transition_weights_[a_idx];
    auto& b = transition_bias_[a_idx];

    for (Dim i = 0; i < latent_dim_; ++i) {
        Real error = predicted.latent.at(i) - actual_next.latent.at(i);
        Real grad_tanh = 1.0 - predicted.latent.at(i) * predicted.latent.at(i);
        Real delta = error * grad_tanh;

        b[i] -= learning_rate * delta;
        for (Dim j = 0; j < latent_dim_; ++j) {
            w[i * latent_dim_ + j] -= learning_rate * delta * current.latent.at(j);
        }
    }
}

Real DynamicsPredictor::prediction_error(const State& predicted,
                                          const State& actual) const {
    Tensor diff = predicted.latent - actual.latent;
    return diff.l2_norm();
}

} // namespace uik::world_model
