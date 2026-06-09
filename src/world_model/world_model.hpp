#pragma once

#include "common/interfaces.hpp"
#include "world_model/latent_state.hpp"
#include "world_model/dynamics_predictor.hpp"
#include "world_model/novelty_detector.hpp"

namespace uik::world_model {

// Aggregate root for the WorldModel bounded context.
// Integrates encoder, dynamics predictor, and novelty detector.
// Tracks compression progress as intrinsic reward signal.
class WorldModel final : public IWorldModel {
public:
    struct Config {
        Dim input_dim       = 64;
        Dim latent_dim      = 16;
        int action_space    = 4;
        std::size_t novelty_window = 100;
        std::size_t novelty_k      = 5;
        Real learning_rate  = 0.01;
        unsigned seed       = 42;
    };

    explicit WorldModel(Config config);

    State encode(const Observation& obs) override;
    State predict_next(const State& current, const Action& action) override;
    Real  compute_novelty(const State& state) override;
    void  update(const Observation& obs) override;
    void  record_action(const Action& action);
    void  set_learning_rate(Real lr) { config_.learning_rate = lr; }
    Real  compression_progress() const override;
    [[nodiscard]] Real prediction_error_rate() const;

private:
    Config config_;
    LatentEncoder encoder_;
    DynamicsPredictor dynamics_;
    NoveltyDetector novelty_;

    Real cumulative_error_     = 0.0;
    Real prev_avg_error_        = 0.0;
    std::size_t update_count_   = 0;
    std::size_t prediction_count_ = 0;

    State last_state_;
    Action last_action_;
    bool has_previous_ = false;
};

} // namespace uik::world_model
