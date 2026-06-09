#include "world_model/dynamics_predictor.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace uik::world_model {

// Backward-compatible constructor
DynamicsPredictor::DynamicsPredictor(Dim latent_dim, int action_space_size,
                                      unsigned seed)
    : DynamicsPredictor(Config{latent_dim, std::max(latent_dim, Dim{16}),
                               action_space_size, seed})
{}

DynamicsPredictor::DynamicsPredictor(Config config)
    : config_(config)
    , latent_dim_(config.latent_dim)
    , hidden_dim_(config.hidden_dim)
    , action_space_size_(config.action_space_size)
{
    std::mt19937 rng(config_.seed);
    auto he_init = [&](Dim fan_in) {
        return std::normal_distribution<Real>(
            0.0, std::sqrt(2.0 / static_cast<Real>(fan_in)));
    };

    action_mlps_.resize(static_cast<std::size_t>(action_space_size_));
    for (auto& mlp : action_mlps_) {
        // Layer 1: latent_dim → hidden_dim
        auto dist1 = he_init(latent_dim_);
        mlp.w1.resize(hidden_dim_ * latent_dim_);
        for (auto& v : mlp.w1) v = dist1(rng);
        mlp.b1.assign(hidden_dim_, 0.0);

        // Layer 2: hidden_dim → latent_dim
        auto dist2 = he_init(hidden_dim_);
        mlp.w2.resize(latent_dim_ * hidden_dim_);
        for (auto& v : mlp.w2) v = dist2(rng);
        mlp.b2.assign(latent_dim_, 0.0);
    }
}

Real DynamicsPredictor::gelu(Real x) {
    constexpr Real sqrt_2_over_pi = 0.7978845608028654;
    return x * 0.5 * (1.0 + std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}

State DynamicsPredictor::predict(const State& current,
                                  const Action& action) const {
    if (action.id < 0 || action.id >= action_space_size_) {
        throw std::out_of_range("DynamicsPredictor::predict: invalid action id");
    }
    auto a_idx = static_cast<std::size_t>(action.id);
    const auto& mlp = action_mlps_[a_idx];

    // Layer 1: hidden = gelu(W1 * s + b1)
    std::vector<Real> hidden(hidden_dim_, 0.0);
    for (Dim i = 0; i < hidden_dim_; ++i) {
        Real sum = mlp.b1[i];
        for (Dim j = 0; j < latent_dim_; ++j) {
            sum += mlp.w1[i * latent_dim_ + j] * current.latent.at(j);
        }
        hidden[i] = gelu(sum);
    }

    // Layer 2: delta = tanh(W2 * hidden + b2)
    // Residual: s' = tanh(s + delta)  — keeps values bounded
    std::vector<Real> next(latent_dim_, 0.0);
    for (Dim i = 0; i < latent_dim_; ++i) {
        Real sum = mlp.b2[i];
        for (Dim j = 0; j < hidden_dim_; ++j) {
            sum += mlp.w2[i * hidden_dim_ + j] * hidden[j];
        }
        // Residual connection: s' = tanh(current + delta)
        next[i] = std::tanh(current.latent.at(i) + sum);
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
    auto& mlp = action_mlps_[a_idx];

    // Recompute hidden for backprop
    std::vector<Real> hidden(hidden_dim_, 0.0);
    std::vector<Real> pre_act1(hidden_dim_, 0.0);
    for (Dim i = 0; i < hidden_dim_; ++i) {
        Real sum = mlp.b1[i];
        for (Dim j = 0; j < latent_dim_; ++j) {
            sum += mlp.w1[i * latent_dim_ + j] * current.latent.at(j);
        }
        pre_act1[i] = sum;
        hidden[i] = gelu(sum);
    }

    // d(loss)/d(predicted) = predicted - actual
    // d(predicted)/d(pre_tanh) = 1 - predicted^2  (tanh derivative)
    std::vector<Real> d_pre_out(latent_dim_);
    for (Dim i = 0; i < latent_dim_; ++i) {
        Real error = predicted.latent.at(i) - actual_next.latent.at(i);
        Real grad_tanh = 1.0 - predicted.latent.at(i) * predicted.latent.at(i);
        d_pre_out[i] = error * grad_tanh;
    }

    // Update layer 2: W2, b2
    std::vector<Real> d_hidden(hidden_dim_, 0.0);
    for (Dim i = 0; i < latent_dim_; ++i) {
        mlp.b2[i] -= learning_rate * d_pre_out[i];
        for (Dim j = 0; j < hidden_dim_; ++j) {
            mlp.w2[i * hidden_dim_ + j] -= learning_rate * d_pre_out[i] * hidden[j];
            d_hidden[j] += mlp.w2[i * hidden_dim_ + j] * d_pre_out[i];
        }
    }

    // d_hidden *= gelu'(pre_act1)
    // gelu'(x) ≈ 0.5*(1+tanh(c*(x+0.044715*x^3))) + x*0.5*sech^2(...)*c*(1+3*0.044715*x^2)
    // Simplified: use sigmoid approximation gelu'(x) ≈ sigmoid(1.702*x)
    for (Dim i = 0; i < hidden_dim_; ++i) {
        Real sigmoid_approx = 1.0 / (1.0 + std::exp(-1.702 * pre_act1[i]));
        d_hidden[i] *= sigmoid_approx;
    }

    // Update layer 1: W1, b1
    for (Dim i = 0; i < hidden_dim_; ++i) {
        mlp.b1[i] -= learning_rate * d_hidden[i];
        for (Dim j = 0; j < latent_dim_; ++j) {
            mlp.w1[i * latent_dim_ + j] -= learning_rate * d_hidden[i] * current.latent.at(j);
        }
    }
}

Real DynamicsPredictor::prediction_error(const State& predicted,
                                          const State& actual) const {
    Tensor diff = predicted.latent - actual.latent;
    return diff.l2_norm();
}

} // namespace uik::world_model
