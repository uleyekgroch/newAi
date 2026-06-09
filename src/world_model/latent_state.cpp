#include "world_model/latent_state.hpp"
#include <cmath>
#include <numeric>

namespace uik::world_model {

// Backward-compatible constructor
LatentEncoder::LatentEncoder(Dim input_dim, Dim latent_dim, unsigned seed)
    : LatentEncoder(Config{input_dim, latent_dim,
                           std::max(latent_dim, Dim{16}), 2, 2, seed})
{}

LatentEncoder::LatentEncoder(Config config)
    : config_(config)
{
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

    // Projection: input → hidden
    fill_weights(proj_w_, config_.hidden_dim, config_.input_dim);
    fill_bias(proj_b_, config_.hidden_dim);

    // Residual layers
    layers_.resize(config_.num_layers);
    for (auto& layer : layers_) {
        fill_weights(layer.w1, config_.hidden_dim, config_.hidden_dim);
        fill_bias(layer.b1, config_.hidden_dim);
        fill_weights(layer.w2, config_.hidden_dim, config_.hidden_dim);
        fill_bias(layer.b2, config_.hidden_dim);
    }

    // Output projection: hidden → latent
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

State LatentEncoder::encode(const Observation& obs) const {
    if (obs.data.flat_size() != config_.input_dim) {
        throw std::invalid_argument("LatentEncoder::encode: input size mismatch");
    }

    // Project input → hidden
    std::vector<Real> h(config_.hidden_dim);
    for (Dim i = 0; i < config_.hidden_dim; ++i) {
        Real sum = proj_b_[i];
        for (Dim j = 0; j < config_.input_dim; ++j) {
            sum += proj_w_[i * config_.input_dim + j] * obs.data.at(j);
        }
        h[i] = gelu(sum);
    }

    // TRM-style recursive refinement: run the same layers multiple times
    for (std::size_t r = 0; r < config_.num_recurse; ++r) {
        for (const auto& layer : layers_) {
            // Residual block: h' = h + act(W2 · act(W1 · h + b1) + b2)
            auto mid = matmul_bias_act(h, config_.hidden_dim, config_.hidden_dim,
                                        layer.w1, layer.b1, true);
            auto out = matmul_bias_act(mid, config_.hidden_dim, config_.hidden_dim,
                                        layer.w2, layer.b2, false);
            // Residual connection
            for (Dim i = 0; i < config_.hidden_dim; ++i) {
                h[i] = std::tanh(h[i] + out[i]);
            }
            apply_layer_norm(h);
        }
    }

    // Output projection: hidden → latent
    std::vector<Real> latent(config_.latent_dim);
    for (Dim i = 0; i < config_.latent_dim; ++i) {
        Real sum = out_b_[i];
        for (Dim j = 0; j < config_.hidden_dim; ++j) {
            sum += out_w_[i * config_.hidden_dim + j] * h[j];
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
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
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

} // namespace uik::world_model
