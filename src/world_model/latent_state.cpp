#include "world_model/latent_state.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace uik::world_model {

// Backward-compatible constructor
LatentEncoder::LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed)
    : LatentEncoder(Config{input_dim, latent_dim,
                           std::max(latent_dim, Dim{16}),
                           /*num_heads=*/4, /*patch_size=*/4,
                           2, 2, seed})
{}

LatentEncoder::LatentEncoder(Config config)
    : config_(config)
{
    // Ensure patch_size divides input_dim; fall back to 1 token otherwise
    if (config_.patch_size == 0) config_.patch_size = config_.input_dim;
    num_tokens_ = config_.input_dim / config_.patch_size;
    if (num_tokens_ == 0) num_tokens_ = 1;

    // Ensure num_heads divides hidden_dim
    if (config_.num_heads == 0) config_.num_heads = 1;
    if (config_.hidden_dim % config_.num_heads != 0) {
        config_.num_heads = 1;
    }

    std::mt19937 rng(config_.seed);
    auto he_init = [&](Dim fan_in) {
        return std::normal_distribution<Real>(
            0.0, std::sqrt(2.0 / static_cast<Real>(fan_in)));
    };

    auto fill_weights = [&](std::vector<Real>& w, Dim rows, Dim cols) {
        auto dist = he_init(cols);
        w.resize(rows * cols);
        for (auto& v : w) v = dist(rng);
    };
    auto fill_bias = [](std::vector<Real>& b, Dim size) {
        b.assign(size, 0.0);
    };

    // Patch embedding projection: patch_size → hidden_dim per token
    // Applied to each patch independently
    fill_weights(proj_w_, config_.hidden_dim, config_.patch_size);
    fill_bias(proj_b_, config_.hidden_dim);

    // Attention + FFN layers
    layers_.resize(config_.num_layers);
    for (auto& layer : layers_) {
        // QKV + output projections
        fill_weights(layer.wq, config_.hidden_dim, config_.hidden_dim);
        fill_weights(layer.wk, config_.hidden_dim, config_.hidden_dim);
        fill_weights(layer.wv, config_.hidden_dim, config_.hidden_dim);
        fill_weights(layer.wo, config_.hidden_dim, config_.hidden_dim);
        // FFN
        fill_weights(layer.w1, config_.hidden_dim, config_.hidden_dim);
        fill_bias(layer.b1, config_.hidden_dim);
        fill_weights(layer.w2, config_.hidden_dim, config_.hidden_dim);
        fill_bias(layer.b2, config_.hidden_dim);
    }

    // Output projection: hidden_dim → latent_dim (pool over tokens)
    fill_weights(out_w_, config_.latent_dim, config_.hidden_dim);
    fill_bias(out_b_, config_.latent_dim);

    // Decoder: latent → hidden → input
    fill_weights(dec_w1_, config_.hidden_dim, config_.latent_dim);
    fill_bias(dec_b1_, config_.hidden_dim);
    fill_weights(dec_w2_, config_.input_dim, config_.hidden_dim);
    fill_bias(dec_b2_, config_.input_dim);
}

std::vector<Real> LatentEncoder::matmul_bias_act(
    const std::vector<Real>& input, Dim rows, Dim cols,
    const std::vector<Real>& w, const std::vector<Real>& b,
    bool activate) const
{
    std::vector<Real> out(rows, 0.0);
    for (Dim i = 0; i < rows; ++i) {
        Real sum = b[i];
        for (Dim j = 0; j < cols; ++j) {
            sum += w[i * cols + j] * input[j];
        }
        out[i] = activate ? gelu(sum) : std::tanh(sum);
    }
    return out;
}

void LatentEncoder::softmax(std::vector<Real>& v, Dim start, Dim len) {
    Real max_val = *std::max_element(v.begin() + static_cast<long>(start),
                                     v.begin() + static_cast<long>(start + len));
    Real sum = 0.0;
    for (Dim i = start; i < start + len; ++i) {
        v[i] = std::exp(v[i] - max_val);
        sum += v[i];
    }
    if (sum > 0.0) {
        for (Dim i = start; i < start + len; ++i) {
            v[i] /= sum;
        }
    }
}

void LatentEncoder::self_attention(std::vector<Real>& tokens,
                                    const AttentionLayer& layer) const {
    // tokens layout: [T tokens × D hidden_dim] flattened
    Dim T = num_tokens_;
    Dim D = config_.hidden_dim;
    Dim H = config_.num_heads;
    Dim head_dim = D / H;

    // Compute Q, K, V for all tokens
    std::vector<Real> Q(T * D, 0.0);
    std::vector<Real> K(T * D, 0.0);
    std::vector<Real> V(T * D, 0.0);

    for (Dim t = 0; t < T; ++t) {
        for (Dim i = 0; i < D; ++i) {
            Real q = 0.0, k = 0.0, v = 0.0;
            for (Dim j = 0; j < D; ++j) {
                q += layer.wq[i * D + j] * tokens[t * D + j];
                k += layer.wk[i * D + j] * tokens[t * D + j];
                v += layer.wv[i * D + j] * tokens[t * D + j];
            }
            Q[t * D + i] = q;
            K[t * D + i] = k;
            V[t * D + i] = v;
        }
    }

    // Multi-head attention: for each head, compute scaled dot-product
    Real scale = 1.0 / std::sqrt(static_cast<Real>(head_dim));
    std::vector<Real> attn_out(T * D, 0.0);

    for (Dim h = 0; h < H; ++h) {
        Dim offset = h * head_dim;

        // Attention scores: T × T
        std::vector<Real> scores(T * T, 0.0);
        for (Dim qi = 0; qi < T; ++qi) {
            for (Dim ki = 0; ki < T; ++ki) {
                Real dot = 0.0;
                for (Dim d = 0; d < head_dim; ++d) {
                    dot += Q[qi * D + offset + d] * K[ki * D + offset + d];
                }
                scores[qi * T + ki] = dot * scale;
            }
            // Softmax over keys for this query
            softmax(scores, qi * T, T);
        }

        // Weighted sum of values
        for (Dim qi = 0; qi < T; ++qi) {
            for (Dim d = 0; d < head_dim; ++d) {
                Real val = 0.0;
                for (Dim ki = 0; ki < T; ++ki) {
                    val += scores[qi * T + ki] * V[ki * D + offset + d];
                }
                attn_out[qi * D + offset + d] = val;
            }
        }
    }

    // Output projection: Wo * attn_out + residual
    for (Dim t = 0; t < T; ++t) {
        std::vector<Real> projected(D, 0.0);
        for (Dim i = 0; i < D; ++i) {
            for (Dim j = 0; j < D; ++j) {
                projected[i] += layer.wo[i * D + j] * attn_out[t * D + j];
            }
        }
        // Residual connection
        for (Dim i = 0; i < D; ++i) {
            tokens[t * D + i] += projected[i];
        }
        // Layer norm per token
        apply_layer_norm_segment(tokens, t * D, D);
    }
}

State LatentEncoder::encode(const Observation& obs) const {
    if (obs.data.flat_size() != config_.input_dim) {
        throw std::invalid_argument("LatentEncoder::encode: input size mismatch");
    }

    Dim T = num_tokens_;
    Dim D = config_.hidden_dim;

    // Patch embedding: split input into T patches, project each to hidden_dim
    std::vector<Real> tokens(T * D, 0.0);
    for (Dim t = 0; t < T; ++t) {
        Dim patch_start = t * config_.patch_size;
        for (Dim i = 0; i < D; ++i) {
            Real sum = proj_b_[i];
            for (Dim j = 0; j < config_.patch_size && (patch_start + j) < config_.input_dim; ++j) {
                sum += proj_w_[i * config_.patch_size + j] * obs.data.at(patch_start + j);
            }
            tokens[t * D + i] = gelu(sum);
        }
    }

    // TRM-style recursive refinement with self-attention
    for (std::size_t r = 0; r < config_.num_recurse; ++r) {
        for (const auto& layer : layers_) {
            // Self-attention (V-JEPA2 style)
            self_attention(tokens, layer);

            // FFN with residual per token
            for (Dim t = 0; t < T; ++t) {
                std::vector<Real> tok(D);
                for (Dim i = 0; i < D; ++i) tok[i] = tokens[t * D + i];

                // FFN: h' = h + act(W2 * act(W1 * h + b1) + b2)
                auto mid = matmul_bias_act(tok, D, D, layer.w1, layer.b1, true);
                auto ffn_out = matmul_bias_act(mid, D, D, layer.w2, layer.b2, false);

                for (Dim i = 0; i < D; ++i) {
                    tokens[t * D + i] = std::tanh(tokens[t * D + i] + ffn_out[i]);
                }
                apply_layer_norm_segment(tokens, t * D, D);
            }
        }
    }

    // Mean pooling over tokens → single hidden_dim vector
    std::vector<Real> pooled(D, 0.0);
    for (Dim t = 0; t < T; ++t) {
        for (Dim i = 0; i < D; ++i) {
            pooled[i] += tokens[t * D + i];
        }
    }
    Real inv_T = 1.0 / static_cast<Real>(T);
    for (Dim i = 0; i < D; ++i) pooled[i] *= inv_T;

    // Output projection: hidden → latent
    std::vector<Real> latent(config_.latent_dim);
    for (Dim i = 0; i < config_.latent_dim; ++i) {
        Real sum = out_b_[i];
        for (Dim j = 0; j < D; ++j) {
            sum += out_w_[i * D + j] * pooled[j];
        }
        latent[i] = std::tanh(sum);
    }
    return State{Tensor({config_.latent_dim}, std::move(latent))};
}

Tensor LatentEncoder::decode(const State& state) const {
    if (state.latent.flat_size() != config_.latent_dim) {
        throw std::invalid_argument("LatentEncoder::decode: latent size mismatch");
    }

    // latent → hidden
    std::vector<Real> input_vec(config_.latent_dim);
    for (Dim i = 0; i < config_.latent_dim; ++i) {
        input_vec[i] = state.latent.at(i);
    }
    auto h = matmul_bias_act(input_vec, config_.hidden_dim, config_.latent_dim,
                              dec_w1_, dec_b1_, true);

    // hidden → output
    auto out = matmul_bias_act(h, config_.input_dim, config_.hidden_dim,
                                dec_w2_, dec_b2_, false);
    return Tensor({config_.input_dim}, std::move(out));
}

Real LatentEncoder::gelu(Real x) {
    constexpr Real sqrt_2_over_pi = 0.7978845608028654;
    return x * 0.5 * (1.0 + std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}

Real LatentEncoder::layer_norm_scale(const std::vector<Real>& v) {
    if (v.empty()) return 1.0;
    Real mean = std::accumulate(v.begin(), v.end(), 0.0) /
                static_cast<Real>(v.size());
    Real var = 0.0;
    for (auto x : v) var += (x - mean) * (x - mean);
    var /= static_cast<Real>(v.size());
    return 1.0 / std::sqrt(var + 1e-5);
}

void LatentEncoder::apply_layer_norm(std::vector<Real>& v) const {
    if (v.empty()) return;
    Real mean = std::accumulate(v.begin(), v.end(), 0.0) /
                static_cast<Real>(v.size());
    Real var = 0.0;
    for (auto x : v) var += (x - mean) * (x - mean);
    var /= static_cast<Real>(v.size());
    Real inv_std = 1.0 / std::sqrt(var + 1e-5);
    for (auto& x : v) x = (x - mean) * inv_std;
}

void LatentEncoder::apply_layer_norm_segment(std::vector<Real>& v,
                                              Dim start, Dim len) const {
    if (len == 0) return;
    Real mean = 0.0;
    for (Dim i = start; i < start + len; ++i) mean += v[i];
    mean /= static_cast<Real>(len);
    Real var = 0.0;
    for (Dim i = start; i < start + len; ++i) var += (v[i] - mean) * (v[i] - mean);
    var /= static_cast<Real>(len);
    Real inv_std = 1.0 / std::sqrt(var + 1e-5);
    for (Dim i = start; i < start + len; ++i) v[i] = (v[i] - mean) * inv_std;
}

} // namespace uik::world_model
