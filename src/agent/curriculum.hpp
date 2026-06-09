#pragma once

#include "common/interfaces.hpp"
#include "agent/grid_world.hpp"
#include "agent/arc_env.hpp"
#include "agent/physics_env.hpp"
#include <memory>
#include <vector>
#include <random>

namespace uik::agent {

// Curriculum learning: progressively harder environments.
// Manages a sequence of environments with increasing complexity,
// advancing the agent when it demonstrates competence.
class CurriculumManager {
public:
    struct StageConfig {
        std::string name;
        std::size_t max_steps         = 100;
        Real pass_threshold           = 0.5;   // reward threshold to advance
        std::size_t min_episodes      = 3;     // min episodes before advancing
    };

    struct Config {
        std::vector<StageConfig> stages;
        unsigned seed = 42;
    };

    CurriculumManager();
    explicit CurriculumManager(Config config);

    // Get current environment
    [[nodiscard]] IEnvironment& current_env();

    // Report episode result, returns true if advanced to next stage
    bool report_episode(Real total_reward, std::size_t steps);

    // Current stage index
    [[nodiscard]] std::size_t current_stage() const { return current_stage_; }

    // Total stages
    [[nodiscard]] std::size_t total_stages() const { return envs_.size(); }

    // Is curriculum complete?
    [[nodiscard]] bool complete() const {
        return current_stage_ >= envs_.size();
    }

    // Stage name
    [[nodiscard]] const std::string& stage_name() const;

    // Performance at current stage
    [[nodiscard]] Real average_reward() const;

    // Total episodes completed across all stages
    [[nodiscard]] std::size_t total_episodes() const { return total_episodes_; }

    // Open-ended environment generation: procedurally generate new stages
    // beyond the fixed curriculum
    bool open_ended() const { return open_ended_; }
    void enable_open_ended() { open_ended_ = true; }
    std::size_t generated_stages() const { return generated_count_; }

private:
    Config config_;
    std::vector<std::unique_ptr<IEnvironment>> envs_;
    std::vector<StageConfig> stages_;
    std::size_t current_stage_ = 0;
    std::vector<Real> stage_rewards_;
    std::size_t stage_episodes_ = 0;
    std::size_t total_episodes_ = 0;

    void build_environments();
    void generate_next_stage();  // procedurally generate a new harder stage
    bool open_ended_ = false;
    std::size_t generated_count_ = 0;
    std::mt19937 rng_;
};

} // namespace uik::agent
