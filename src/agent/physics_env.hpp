#pragma once

#include "common/interfaces.hpp"
#include <vector>
#include <random>

namespace uik::agent {

// Physics environment with causal intervention learning.
// Simulates a 2D world with objects that have mass, position, velocity.
// Agent can push/pull objects and observe gravitational effects.
// Goal: learn physical rules (F=ma, gravity, collision) through trial-and-error.
class PhysicsEnvironment final : public IEnvironment {
public:
    struct Object {
        Real x = 0.0, y = 0.0;       // position
        Real vx = 0.0, vy = 0.0;     // velocity
        Real mass = 1.0;
        int color = 1;
        bool fixed = false;           // immovable object (e.g. wall)
    };

    struct Config {
        Real world_width  = 10.0;
        Real world_height = 10.0;
        Real gravity      = -0.1;
        Real friction     = 0.98;
        Real push_force   = 0.5;
        int max_objects   = 5;
        Dim obs_dim       = 64;
        std::size_t max_steps = 200;
        unsigned seed     = 42;
    };

    PhysicsEnvironment();
    explicit PhysicsEnvironment(Config config);

    Observation reset() override;
    StepResult step(const Action& action) override;
    int action_space_size() const override;

    [[nodiscard]] const std::vector<Object>& objects() const { return objects_; }
    [[nodiscard]] std::size_t step_count() const { return step_count_; }

private:
    Config config_;
    std::mt19937 rng_;
    std::vector<Object> objects_;
    Object target_;           // goal position to push object to
    std::size_t step_count_ = 0;

    // Actions: push_left, push_right, push_up, push_down per object
    // + do_nothing + drop_object
    static constexpr int ACTIONS_PER_OBJ = 4;
    static constexpr int GLOBAL_ACTIONS  = 2; // noop, drop new object

    void generate_scene();
    void physics_step();
    void apply_gravity();
    void apply_friction();
    void resolve_collisions();
    void clamp_positions();
    Tensor build_observation() const;
    Real compute_reward() const;
};

} // namespace uik::agent
