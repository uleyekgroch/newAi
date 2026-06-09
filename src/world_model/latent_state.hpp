#pragma once

#include "common/types.hpp"
#include <random>

namespace uik::world_model {

// Encodes raw observations into compact latent states.
// Uses a simple linear projection (W * obs + bias) as a minimal encoder.
class LatentEncoder {
public:
    explicit LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed = 42);

    [[nodiscard]] State encode(const Observation& obs) const;
    [[nodiscard]] Tensor decode(const State& state) const;

    [[nodiscard]] Dim input_dim() const { return input_dim_; }
    [[nodiscard]] Dim latent_dim() const { return latent_dim_; }

private:
    Dim input_dim_;
    Dim latent_dim_;
    std::vector<Real> weights_;  // latent_dim x input_dim
    std::vector<Real> bias_;     // latent_dim
    std::vector<Real> decode_w_; // input_dim x latent_dim

    Real activation(Real x) const;
};

} // namespace uik::world_model
