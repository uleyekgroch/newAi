#pragma once

#include "common/types.hpp"
#include <vector>

namespace uik::world_model {

// Predicts the next latent state given current state + action.
// Implements a simple learned transition model: s' = W_a * s + b_a
class DynamicsPredictor {
public:
    explicit DynamicsPredictor(Dim latent_dim, int action_space_size,
                                unsigned seed = 42);

    [[nodiscard]] State predict(const State& current, const Action& action) const;

    void learn(const State& current, const Action& action,
               const State& actual_next, Real learning_rate = 0.01);

    [[nodiscard]] Real prediction_error(const State& predicted,
                                         const State& actual) const;

private:
    Dim latent_dim_;
    int action_space_size_;
    // Per-action transition matrices: action_space_size x (latent_dim x latent_dim)
    std::vector<std::vector<Real>> transition_weights_;
    std::vector<std::vector<Real>> transition_bias_;
};

} // namespace uik::world_model
