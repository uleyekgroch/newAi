#include "world_model/latent_state.hpp"
#include <cmath>

namespace uik::world_model {

LatentEncoder::LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed)
    : input_dim_(input_dim), latent_dim_(latent_dim)
{
    std::mt19937 rng(seed);
    Real scale = std::sqrt(2.0 / static_cast<Real>(input_dim));
    std::normal_distribution<Real> dist(0.0, scale);

    weights_.resize(latent_dim * input_dim);
    for (auto& w : weights_) w = dist(rng);

    bias_.resize(latent_dim, 0.0);

    decode_w_.resize(input_dim * latent_dim);
    for (auto& w : decode_w_) w = dist(rng);
}

State LatentEncoder::encode(const Observation& obs) const {
    if (obs.data.flat_size() != input_dim_) {
        throw std::invalid_argument("LatentEncoder::encode: input size mismatch");
    }
    std::vector<Real> latent(latent_dim_, 0.0);
    for (Dim i = 0; i < latent_dim_; ++i) {
        Real sum = bias_[i];
        for (Dim j = 0; j < input_dim_; ++j) {
            sum += weights_[i * input_dim_ + j] * obs.data.at(j);
        }
        latent[i] = activation(sum);
    }
    return State{Tensor({latent_dim_}, std::move(latent))};
}

Tensor LatentEncoder::decode(const State& state) const {
    if (state.latent.flat_size() != latent_dim_) {
        throw std::invalid_argument("LatentEncoder::decode: latent size mismatch");
    }
    std::vector<Real> output(input_dim_, 0.0);
    for (Dim i = 0; i < input_dim_; ++i) {
        Real sum = 0.0;
        for (Dim j = 0; j < latent_dim_; ++j) {
            sum += decode_w_[i * latent_dim_ + j] * state.latent.at(j);
        }
        output[i] = activation(sum);
    }
    return Tensor({input_dim_}, std::move(output));
}

Real LatentEncoder::activation(Real x) const {
    return std::tanh(x);
}

} // namespace uik::world_model
