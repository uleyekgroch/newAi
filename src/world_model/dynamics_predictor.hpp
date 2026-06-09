#pragma once

#include "common/types.hpp"
#include <vector>

namespace uik::world_model {

// Predicts the next latent state given current state + action.
// 2-layer MLP per action: s → hidden → s' with residual connection.
// Architecture: s' = s + tanh(W2_a * gelu(W1_a * s + b1_a) + b2_a)
class DynamicsPredictor {
public:
    struct Config {
        Dim latent_dim       = 16;
        Dim hidden_dim       = 32;   // wider hidden layer for nonlinear capacity
        int action_space_size = 4;
        unsigned seed        = 42;
    };

    explicit DynamicsPredictor(Dim latent_dim, int action_space_size,
                                unsigned seed = 42);
    explicit DynamicsPredictor(Config config);

    [[nodiscard]] State predict(const State& current, const Action& action) const;

    void learn(const State& current, const Action& action,
               const State& actual_next, Real learning_rate = 0.01);

    [[nodiscard]] Real prediction_error(const State& predicted,
                                         const State& actual) const;

private:
    Config config_;
    Dim latent_dim_;
    Dim hidden_dim_;
    int action_space_size_;

    // Per-action 2-layer MLP: layer1 (latent→hidden), layer2 (hidden→latent)
    struct ActionMLP {
        std::vector<Real> w1;  // hidden_dim × latent_dim
        std::vector<Real> b1;  // hidden_dim
        std::vector<Real> w2;  // latent_dim × hidden_dim
        std::vector<Real> b2;  // latent_dim
    };
    std::vector<ActionMLP> action_mlps_;

    static Real gelu(Real x);
};

} // namespace uik::world_model
