#pragma once

#include "common/types.hpp"
#include <random>
#include <vector>

namespace uik::world_model {

// Multi-layer encoder with residual connections and nonlinear activations.
// Inspired by TRM (Tiny Recursive Models): recursive refinement in latent space.
// Architecture: input → [Layer1 → act → Layer2 → act + residual] × N_recurse → latent
class LatentEncoder {
public:
    struct Config {
        Dim input_dim   = 64;
        Dim latent_dim  = 16;
        Dim hidden_dim  = 32;    // hidden layer width
        std::size_t num_layers  = 2;   // depth of encoder stack
        std::size_t num_recurse = 2;   // TRM-style recursive refinement passes
        unsigned seed   = 42;
    };

    explicit LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed = 42);
    explicit LatentEncoder(Config config);

    [[nodiscard]] State encode(const Observation& obs) const;
    [[nodiscard]] Tensor decode(const State& state) const;

    [[nodiscard]] Dim input_dim() const { return config_.input_dim; }
    [[nodiscard]] Dim latent_dim() const { return config_.latent_dim; }
    [[nodiscard]] std::size_t num_layers() const { return config_.num_layers; }
    [[nodiscard]] std::size_t num_recurse() const { return config_.num_recurse; }

private:
    Config config_;

    // Projection: input_dim → hidden_dim
    std::vector<Real> proj_w_;
    std::vector<Real> proj_b_;

    // Each layer: hidden_dim → hidden_dim with residual
    struct Layer {
        std::vector<Real> w1;  // hidden_dim × hidden_dim
        std::vector<Real> b1;
        std::vector<Real> w2;  // hidden_dim × hidden_dim
        std::vector<Real> b2;
    };
    std::vector<Layer> layers_;

    // Output: hidden_dim → latent_dim
    std::vector<Real> out_w_;
    std::vector<Real> out_b_;

    // Decoder: latent_dim → input_dim (2-layer)
    std::vector<Real> dec_w1_;  // hidden_dim × latent_dim
    std::vector<Real> dec_b1_;
    std::vector<Real> dec_w2_;  // input_dim × hidden_dim
    std::vector<Real> dec_b2_;

    static Real gelu(Real x);
    static Real layer_norm_scale(const std::vector<Real>& v);
    void apply_layer_norm(std::vector<Real>& v) const;
    std::vector<Real> matmul_bias_act(
        const std::vector<Real>& input, Dim rows, Dim cols,
        const std::vector<Real>& w, const std::vector<Real>& b,
        bool activate) const;
};

} // namespace uik::world_model
