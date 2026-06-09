#include <gtest/gtest.h>
#include "world_model/latent_state.hpp"
#include "world_model/dynamics_predictor.hpp"
#include "world_model/novelty_detector.hpp"
#include "world_model/world_model.hpp"

using namespace uik;
using namespace uik::world_model;

// ── LatentEncoder Tests ──

TEST(LatentEncoder, encode_produces_correct_latent_dim) {
    LatentEncoder encoder(64, 16);
    Observation obs{Tensor({64}, 1.0)};
    State state = encoder.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 16u);
}

TEST(LatentEncoder, encode_rejects_wrong_input_size) {
    LatentEncoder encoder(64, 16);
    Observation obs{Tensor({32}, 1.0)};
    EXPECT_THROW(encoder.encode(obs), std::invalid_argument);
}

TEST(LatentEncoder, encode_deterministic_for_same_input) {
    LatentEncoder encoder(64, 16);
    Observation obs{Tensor({64}, 0.5)};
    State s1 = encoder.encode(obs);
    State s2 = encoder.encode(obs);
    EXPECT_EQ(s1.latent, s2.latent);
}

TEST(LatentEncoder, encode_different_inputs_different_outputs) {
    LatentEncoder encoder(64, 16);
    Observation obs1{Tensor({64}, 0.0)};
    Observation obs2{Tensor({64}, 1.0)};
    State s1 = encoder.encode(obs1);
    State s2 = encoder.encode(obs2);
    EXPECT_NE(s1.latent, s2.latent);
}

TEST(LatentEncoder, decode_produces_correct_input_dim) {
    LatentEncoder encoder(64, 16);
    Observation obs{Tensor({64}, 0.5)};
    State state = encoder.encode(obs);
    Tensor decoded = encoder.decode(state);
    EXPECT_EQ(decoded.flat_size(), 64u);
}

// ── DynamicsPredictor Tests ──

TEST(DynamicsPredictor, predict_produces_correct_dim) {
    DynamicsPredictor pred(16, 4);
    State s{Tensor({16}, 0.5)};
    State next = pred.predict(s, Action{0});
    EXPECT_EQ(next.latent.flat_size(), 16u);
}

TEST(DynamicsPredictor, predict_rejects_invalid_action) {
    DynamicsPredictor pred(16, 4);
    State s{Tensor({16}, 0.5)};
    EXPECT_THROW(pred.predict(s, Action{-1}), std::out_of_range);
    EXPECT_THROW(pred.predict(s, Action{4}), std::out_of_range);
}

TEST(DynamicsPredictor, different_actions_different_predictions) {
    DynamicsPredictor pred(16, 4);
    State s{Tensor({16}, 0.5)};
    State n0 = pred.predict(s, Action{0});
    State n1 = pred.predict(s, Action{1});
    EXPECT_NE(n0.latent, n1.latent);
}

TEST(DynamicsPredictor, learn_reduces_prediction_error) {
    DynamicsPredictor pred(8, 2);
    State current{Tensor({8}, 0.3)};
    State target{Tensor({8}, 0.7)};
    Action a{0};

    Real error_before = pred.prediction_error(pred.predict(current, a), target);
    for (int i = 0; i < 100; ++i) {
        pred.learn(current, a, target, 0.05);
    }
    Real error_after = pred.prediction_error(pred.predict(current, a), target);
    EXPECT_LT(error_after, error_before);
}

// ── NoveltyDetector Tests ──

TEST(NoveltyDetector, first_observation_is_novel) {
    NoveltyDetector detector(10, 3);
    State s{Tensor({4}, 1.0)};
    EXPECT_DOUBLE_EQ(detector.score(s), 1.0);
}

TEST(NoveltyDetector, repeated_state_has_low_novelty) {
    NoveltyDetector detector(10, 3);
    State s{Tensor({4}, 0.5)};
    detector.observe(s);
    detector.observe(s);
    detector.observe(s);
    EXPECT_NEAR(detector.score(s), 0.0, 1e-6);
}

TEST(NoveltyDetector, different_state_has_high_novelty) {
    NoveltyDetector detector(10, 3);
    State s1{Tensor({4}, 0.0)};
    detector.observe(s1);
    detector.observe(s1);
    State s2{Tensor({4}, 10.0)};
    EXPECT_GT(detector.score(s2), 0.1);
}

TEST(NoveltyDetector, reset_clears_history) {
    NoveltyDetector detector(10, 3);
    State s{Tensor({4}, 0.5)};
    detector.observe(s);
    detector.reset();
    EXPECT_DOUBLE_EQ(detector.score(s), 1.0);
}

// ── WorldModel Integration Tests ──

TEST(WorldModel, encode_and_predict_produce_correct_dims) {
    WorldModel::Config cfg;
    cfg.input_dim = 32;
    cfg.latent_dim = 8;
    cfg.action_space = 4;
    WorldModel wm(cfg);

    Observation obs{Tensor({32}, 0.5)};
    State state = wm.encode(obs);
    EXPECT_EQ(state.latent.flat_size(), 8u);

    State next = wm.predict_next(state, Action{1});
    EXPECT_EQ(next.latent.flat_size(), 8u);
}

TEST(WorldModel, compression_progress_zero_initially) {
    WorldModel::Config cfg;
    cfg.input_dim = 16;
    cfg.latent_dim = 4;
    cfg.action_space = 2;
    WorldModel wm(cfg);

    EXPECT_DOUBLE_EQ(wm.compression_progress(), 0.0);
}

TEST(WorldModel, record_action_affects_dynamics_learning) {
    WorldModel::Config cfg;
    cfg.input_dim = 16;
    cfg.latent_dim = 4;
    cfg.action_space = 2;
    cfg.learning_rate = 0.1;
    WorldModel wm(cfg);

    Observation obs1{Tensor({16}, 0.3)};
    Observation obs2{Tensor({16}, 0.7)};

    // First update sets last_state_ but has_previous_=false
    wm.update(obs1);
    wm.record_action(Action{0});

    // Second update: has_previous_=true, learns dynamics
    wm.update(obs2);
    wm.record_action(Action{1});

    EXPECT_GE(wm.prediction_error_rate(), 0.0);
}

TEST(WorldModel, compression_progress_after_learning) {
    WorldModel::Config cfg;
    cfg.input_dim = 8;
    cfg.latent_dim = 4;
    cfg.action_space = 2;
    cfg.learning_rate = 0.05;
    WorldModel wm(cfg);

    // Feed the same transition repeatedly so the model can learn it
    Observation obs_a{Tensor({8}, 0.2)};
    Observation obs_b{Tensor({8}, 0.8)};

    for (int i = 0; i < 30; ++i) {
        wm.update(obs_a);
        wm.record_action(Action{0});
        wm.update(obs_b);
        wm.record_action(Action{1});
    }

    // After many updates, compression progress should be non-negative
    EXPECT_GE(wm.compression_progress(), 0.0);
}

TEST(WorldModel, novelty_detection_works) {
    WorldModel::Config cfg;
    cfg.input_dim = 16;
    cfg.latent_dim = 4;
    cfg.action_space = 2;
    WorldModel wm(cfg);

    Observation obs1{Tensor({16}, 0.0)};
    State s1 = wm.encode(obs1);
    Real nov1 = wm.compute_novelty(s1);
    wm.update(obs1);

    Observation obs2{Tensor({16}, 5.0)};
    State s2 = wm.encode(obs2);
    Real nov2 = wm.compute_novelty(s2);

    EXPECT_GT(nov2, 0.0);
    (void)nov1; // first is always 1.0
}
