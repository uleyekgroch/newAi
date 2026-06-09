#pragma once

#include "common/types.hpp"
#include <random>
#include <vector>

namespace uik::world_model {

// V-JEPA2/TRM hybrid encoder: self-attention + residual MLP + recursive refinement.
// Architecture: input → patch_embed → [SelfAttn → FFN + residual] × N_layers × N_recurse → latent
// Self-attention enables context-dependent encoding (V-JEPA2 style),
// while recursive application (TRM style) allows iterative refinement.
class LatentEncoder {
public:
    struct Config {
        Dim input_dim   = 64;
        Dim latent_dim  = 16;
        Dim hidden_dim  = 32;    // hidden layer width
        Dim num_heads   = 4;     // number of attention heads
        Dim patch_size  = 4;     // input tokens = input_dim / patch_size
        std::size_t num_layers  = 2;   // depth of encoder stack
        std::size_t num_recurse = 2;   // TRM-style recursive refinement passes
        unsigned seed   = 42;
    };

    explicit LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed = 42);
    explicit LatentEncoder(Config config);

    [[nodiscard]] State encode(const Observation& obs) const;
    [[nodiscard]] Tensor decode(const State& state) const;

    // Online learning: update encoder weights via reconstruction loss
    // Returns reconstruction loss for this observation
    Real learn(const Observation& obs, Real learning_rate = 0.001);
    [[nodiscard]] Real total_reconstruction_loss() const { return total_recon_loss_; }
    [[nodiscard]] std::size_t learn_count() const { return learn_count_; }

    [[nodiscard]] Dim input_dim() const { return config_.input_dim; }
    [[nodiscard]] Dim latent_dim() const { return config_.latent_dim; }
    [[nodiscard]] Dim num_heads() const { return config_.num_heads; }
    [[nodiscard]] std::size_t num_layers() const { return config_.num_layers; }
    [[nodiscard]] std::size_t num_recurse() const { return config_.num_recurse; }

private:
    Config config_;

    // Projection: input_dim → hidden_dim
    std::vector<Real> proj_w_;
    std::vector<Real> proj_b_;

    // Each layer: self-attention + FFN with residual
    struct AttentionLayer {
        // QKV projection: hidden_dim → 3 * hidden_dim
        std::vector<Real> wq;  // hidden_dim × hidden_dim
        std::vector<Real> wk;  // hidden_dim × hidden_dim
        std::vector<Real> wv;  // hidden_dim × hidden_dim
        std::vector<Real> wo;  // hidden_dim × hidden_dim (output projection)
        // FFN: hidden_dim → hidden_dim
        std::vector<Real> w1;  // hidden_dim × hidden_dim
        std::vector<Real> b1;
        std::vector<Real> w2;  // hidden_dim × hidden_dim
        std::vector<Real> b2;
    };
    std::vector<AttentionLayer> layers_;
    Dim num_tokens_ = 0;  // input_dim / patch_size

    // Output: hidden_dim → latent_dim
    std::vector<Real> out_w_;
    std::vector<Real> out_b_;

    // Decoder: latent_dim → input_dim (2-layer)
    std::vector<Real> dec_w1_;  // hidden_dim × latent_dim
    std::vector<Real> dec_b1_;
    std::vector<Real> dec_w2_;  // input_dim × hidden_dim
    std::vector<Real> dec_b2_;

    Real total_recon_loss_ = 0.0;
    std::size_t learn_count_ = 0;

    static Real gelu(Real x);
    static Real layer_norm_scale(const std::vector<Real>& v);
    void apply_layer_norm(std::vector<Real>& v) const;
    void apply_layer_norm_segment(std::vector<Real>& v, Dim start, Dim len) const;
    std::vector<Real> matmul_bias_act(
        const std::vector<Real>& input, Dim rows, Dim cols,
        const std::vector<Real>& w, const std::vector<Real>& b,
        bool activate) const;

    // Self-attention: Q,K,V over tokens
    void self_attention(std::vector<Real>& tokens,
                        const AttentionLayer& layer) const;
    // Softmax over a segment of a vector
    static void softmax(std::vector<Real>& v, Dim start, Dim len);

    // Backprop helpers for online learning
    void update_weights(std::vector<Real>& w, const std::vector<Real>& grad,
                        Dim size, Real lr);
    void update_bias(std::vector<Real>& b, const std::vector<Real>& grad,
                     Dim size, Real lr);
};

} // namespace uik::world_model
