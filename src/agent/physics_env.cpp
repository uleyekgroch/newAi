#include "agent/physics_env.hpp"
#include <cmath>
#include <algorithm>

namespace uik::agent {

PhysicsEnvironment::PhysicsEnvironment()
    : PhysicsEnvironment(Config{})
{}

PhysicsEnvironment::PhysicsEnvironment(Config config)
    : config_(config), rng_(config.seed)
{}

Observation PhysicsEnvironment::reset() {
    step_count_ = 0;
    generate_scene();
    return Observation{build_observation()};
}

StepResult PhysicsEnvironment::step(const Action& action) {
    ++step_count_;
    int act = action.id;

    // Decode action
    int n_obj = static_cast<int>(objects_.size());
    if (act < n_obj * ACTIONS_PER_OBJ) {
        int obj_idx = act / ACTIONS_PER_OBJ;
        int dir = act % ACTIONS_PER_OBJ;
        auto& obj = objects_[static_cast<std::size_t>(obj_idx)];
        if (!obj.fixed) {
            Real force = config_.push_force / obj.mass;
            switch (dir) {
                case 0: obj.vx -= force; break; // push left
                case 1: obj.vx += force; break; // push right
                case 2: obj.vy += force; break; // push up
                case 3: obj.vy -= force; break; // push down
            }
        }
    } else if (act == n_obj * ACTIONS_PER_OBJ + 1) {
        // Drop a new object if under limit
        if (static_cast<int>(objects_.size()) < config_.max_objects) {
            std::uniform_real_distribution<Real> px(1.0, config_.world_width - 1.0);
            Object obj;
            obj.x = px(rng_);
            obj.y = config_.world_height - 1.0;
            obj.mass = 1.0;
            obj.color = static_cast<int>(objects_.size()) + 1;
            objects_.push_back(obj);
        }
    }

    // Simulate physics
    physics_step();

    Real reward = compute_reward();
    bool done = step_count_ >= config_.max_steps;

    return StepResult{Observation{build_observation()}, reward, done};
}

int PhysicsEnvironment::action_space_size() const {
    return config_.max_objects * ACTIONS_PER_OBJ + GLOBAL_ACTIONS;
}

void PhysicsEnvironment::generate_scene() {
    objects_.clear();

    // Create 2-3 objects with random positions
    std::uniform_int_distribution<int> n_obj(2, 3);
    int count = n_obj(rng_);
    std::uniform_real_distribution<Real> px(1.0, config_.world_width - 1.0);
    std::uniform_real_distribution<Real> py(1.0, config_.world_height - 1.0);
    std::uniform_real_distribution<Real> mass_dist(0.5, 2.0);

    for (int i = 0; i < count; ++i) {
        Object obj;
        obj.x = px(rng_);
        obj.y = py(rng_);
        obj.mass = mass_dist(rng_);
        obj.color = i + 1;
        objects_.push_back(obj);
    }

    // Add a ground (fixed object)
    Object ground;
    ground.x = config_.world_width / 2.0;
    ground.y = 0.5;
    ground.mass = 1e6;
    ground.fixed = true;
    ground.color = 0;
    objects_.push_back(ground);

    // Target: where we want the first object to go
    target_.x = px(rng_);
    target_.y = py(rng_);
}

void PhysicsEnvironment::physics_step() {
    apply_gravity();
    for (auto& obj : objects_) {
        if (obj.fixed) continue;
        obj.x += obj.vx;
        obj.y += obj.vy;
    }
    apply_friction();
    resolve_collisions();
    clamp_positions();
}

void PhysicsEnvironment::apply_gravity() {
    for (auto& obj : objects_) {
        if (obj.fixed) continue;
        obj.vy += config_.gravity;
    }
}

void PhysicsEnvironment::apply_friction() {
    for (auto& obj : objects_) {
        if (obj.fixed) continue;
        obj.vx *= config_.friction;
        obj.vy *= config_.friction;
    }
}

void PhysicsEnvironment::resolve_collisions() {
    for (std::size_t i = 0; i < objects_.size(); ++i) {
        for (std::size_t j = i + 1; j < objects_.size(); ++j) {
            auto& a = objects_[i];
            auto& b = objects_[j];
            Real dx = a.x - b.x;
            Real dy = a.y - b.y;
            Real dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 1.0 && dist > 0.001) {
                // Simple elastic collision
                Real nx = dx / dist;
                Real ny = dy / dist;
                Real overlap = 1.0 - dist;

                if (!a.fixed && !b.fixed) {
                    Real total_mass = a.mass + b.mass;
                    a.x += nx * overlap * (b.mass / total_mass);
                    a.y += ny * overlap * (b.mass / total_mass);
                    b.x -= nx * overlap * (a.mass / total_mass);
                    b.y -= ny * overlap * (a.mass / total_mass);

                    // Exchange velocity components along collision normal
                    Real rel_v = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
                    if (rel_v > 0) {
                        Real impulse = rel_v / total_mass;
                        a.vx -= impulse * b.mass * nx;
                        a.vy -= impulse * b.mass * ny;
                        b.vx += impulse * a.mass * nx;
                        b.vy += impulse * a.mass * ny;
                    }
                } else if (a.fixed) {
                    b.x -= nx * overlap;
                    b.y -= ny * overlap;
                    Real rel_v = b.vx * nx + b.vy * ny;
                    if (rel_v < 0) {
                        b.vx -= 2.0 * rel_v * nx;
                        b.vy -= 2.0 * rel_v * ny;
                    }
                } else {
                    a.x += nx * overlap;
                    a.y += ny * overlap;
                    Real rel_v = a.vx * nx + a.vy * ny;
                    if (rel_v < 0) {
                        a.vx -= 2.0 * rel_v * nx;
                        a.vy -= 2.0 * rel_v * ny;
                    }
                }
            }
        }
    }
}

void PhysicsEnvironment::clamp_positions() {
    for (auto& obj : objects_) {
        if (obj.fixed) continue;
        obj.x = std::clamp(obj.x, 0.0, config_.world_width);
        obj.y = std::clamp(obj.y, 0.0, config_.world_height);
        if (obj.y <= 0.0) { obj.vy = std::abs(obj.vy) * 0.5; obj.y = 0.01; }
        if (obj.y >= config_.world_height) { obj.vy = -std::abs(obj.vy) * 0.5; }
        if (obj.x <= 0.0) { obj.vx = std::abs(obj.vx) * 0.5; obj.x = 0.01; }
        if (obj.x >= config_.world_width) { obj.vx = -std::abs(obj.vx) * 0.5; }
    }
}

Tensor PhysicsEnvironment::build_observation() const {
    // Encode: [obj1_x, obj1_y, obj1_vx, obj1_vy, obj1_mass, ...] + [target_x, target_y]
    // Pad to obs_dim
    std::vector<Real> obs(config_.obs_dim, 0.0);
    Dim idx = 0;
    for (const auto& obj : objects_) {
        if (idx + 5 > config_.obs_dim) break;
        obs[idx++] = obj.x / config_.world_width;
        obs[idx++] = obj.y / config_.world_height;
        obs[idx++] = obj.vx;
        obs[idx++] = obj.vy;
        obs[idx++] = obj.mass;
    }
    // Target
    if (idx + 2 <= config_.obs_dim) {
        obs[idx++] = target_.x / config_.world_width;
        obs[idx++] = target_.y / config_.world_height;
    }
    return Tensor({config_.obs_dim}, std::move(obs));
}

Real PhysicsEnvironment::compute_reward() const {
    if (objects_.empty()) return 0.0;
    // Reward: negative distance of first non-fixed object to target
    for (const auto& obj : objects_) {
        if (obj.fixed) continue;
        Real dx = obj.x - target_.x;
        Real dy = obj.y - target_.y;
        Real dist = std::sqrt(dx * dx + dy * dy);
        return -dist / (config_.world_width + config_.world_height);
    }
    return 0.0;
}

} // namespace uik::agent
