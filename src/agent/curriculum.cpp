#include "agent/curriculum.hpp"
#include <numeric>

namespace uik::agent {

CurriculumManager::CurriculumManager()
    : CurriculumManager(Config{})
{}

CurriculumManager::CurriculumManager(Config config)
    : config_(config), rng_(config.seed)
{
    if (config_.stages.empty()) {
        // Default curriculum: GridWorld → ARC → Physics
        config_.stages = {
            {"grid_world_3x3", 50, 0.3, 3},
            {"grid_world_5x5", 100, 0.3, 3},
            {"arc_easy",       200, 0.5, 3},
            {"arc_medium",     300, 0.3, 5},
            {"physics_simple", 200, -0.2, 3},
            {"physics_complex", 400, -0.1, 5},
        };
    }
    stages_ = config_.stages;
    build_environments();
}

IEnvironment& CurriculumManager::current_env() {
    if (current_stage_ >= envs_.size()) {
        return *envs_.back();
    }
    return *envs_[current_stage_];
}

bool CurriculumManager::report_episode(Real total_reward, std::size_t /*steps*/) {
    ++stage_episodes_;
    ++total_episodes_;
    stage_rewards_.push_back(total_reward);

    if (current_stage_ >= stages_.size()) {
        // Open-ended: generate new stage when curriculum is exhausted
        if (open_ended_) {
            generate_next_stage();
        }
        return false;
    }
    const auto& stage = stages_[current_stage_];

    if (stage_episodes_ >= stage.min_episodes) {
        Real avg = average_reward();
        if (avg >= stage.pass_threshold) {
            ++current_stage_;
            stage_rewards_.clear();
            stage_episodes_ = 0;
            // If we passed the last fixed stage and open_ended, generate more
            if (current_stage_ >= stages_.size() && open_ended_) {
                generate_next_stage();
            }
            return true;
        }
    }
    return false;
}

const std::string& CurriculumManager::stage_name() const {
    static const std::string empty = "complete";
    if (current_stage_ >= stages_.size()) return empty;
    return stages_[current_stage_].name;
}

Real CurriculumManager::average_reward() const {
    if (stage_rewards_.empty()) return 0.0;
    return std::accumulate(stage_rewards_.begin(), stage_rewards_.end(), 0.0) /
           static_cast<Real>(stage_rewards_.size());
}

void CurriculumManager::build_environments() {
    envs_.clear();
    for (const auto& stage : stages_) {
        if (stage.name.find("grid_world_3x3") != std::string::npos) {
            GridWorld::Config gc;
            gc.rows = 3; gc.cols = 3; gc.num_colors = 5;
            gc.max_steps = stage.max_steps; gc.seed = config_.seed;
            envs_.push_back(std::make_unique<GridWorld>(gc));
        } else if (stage.name.find("grid_world_5x5") != std::string::npos) {
            GridWorld::Config gc;
            gc.rows = 5; gc.cols = 5; gc.num_colors = 10;
            gc.max_steps = stage.max_steps; gc.seed = config_.seed;
            envs_.push_back(std::make_unique<GridWorld>(gc));
        } else if (stage.name.find("arc_easy") != std::string::npos) {
            ArcEnvironment::Config ac;
            ac.grid_rows = 3; ac.grid_cols = 3; ac.num_colors = 5;
            ac.puzzles_per_episode = 3; ac.seed = config_.seed;
            envs_.push_back(std::make_unique<ArcEnvironment>(ac));
        } else if (stage.name.find("arc_medium") != std::string::npos) {
            ArcEnvironment::Config ac;
            ac.grid_rows = 5; ac.grid_cols = 5; ac.num_colors = 10;
            ac.puzzles_per_episode = 5; ac.seed = config_.seed;
            envs_.push_back(std::make_unique<ArcEnvironment>(ac));
        } else if (stage.name.find("physics_simple") != std::string::npos) {
            PhysicsEnvironment::Config pc;
            pc.max_objects = 3; pc.max_steps = stage.max_steps;
            pc.seed = config_.seed;
            envs_.push_back(std::make_unique<PhysicsEnvironment>(pc));
        } else if (stage.name.find("physics_complex") != std::string::npos) {
            PhysicsEnvironment::Config pc;
            pc.max_objects = 5; pc.gravity = -0.2; pc.max_steps = stage.max_steps;
            pc.seed = config_.seed;
            envs_.push_back(std::make_unique<PhysicsEnvironment>(pc));
        } else {
            // Default: simple grid world
            GridWorld::Config gc;
            gc.max_steps = stage.max_steps; gc.seed = config_.seed;
            envs_.push_back(std::make_unique<GridWorld>(gc));
        }
    }
}

void CurriculumManager::generate_next_stage() {
    ++generated_count_;

    // Structural variation: cycle through diverse task structures
    // Not just parameter scaling — different task compositions
    std::size_t structure_type = generated_count_ % 7;  // 7 different structures
    std::size_t difficulty = generated_count_;
    unsigned stage_seed = config_.seed + static_cast<unsigned>(generated_count_);

    int grid_size = static_cast<int>(std::min(Dim{3} + difficulty / 2, Dim{15}));
    int num_colors = static_cast<int>(std::min(Dim{5} + difficulty / 3, Dim{20}));
    std::size_t max_steps = 100 + difficulty * 30;
    Real pass_threshold = std::min(0.3 + 0.03 * static_cast<Real>(difficulty), 0.85);
    std::string stage_name = "gen_" + std::to_string(generated_count_);

    switch (structure_type) {
        case 0: {
            // Small grid, high color diversity (abstract pattern matching)
            stage_name += "_abstract_" + std::to_string(num_colors);
            ArcEnvironment::Config ac;
            ac.grid_rows = 3; ac.grid_cols = 3;
            ac.num_colors = num_colors;
            ac.puzzles_per_episode = std::min(difficulty + 3, std::size_t{15});
            ac.seed = stage_seed;
            auto env = std::make_unique<ArcEnvironment>(ac);
            env->use_benchmark_tasks();
            envs_.push_back(std::move(env));
            break;
        }
        case 1: {
            // Large grid, few colors (spatial reasoning)
            stage_name += "_spatial_" + std::to_string(grid_size);
            ArcEnvironment::Config ac;
            ac.grid_rows = static_cast<Dim>(grid_size);
            ac.grid_cols = static_cast<Dim>(grid_size);
            ac.num_colors = 3;
            ac.puzzles_per_episode = std::min(difficulty + 2, std::size_t{10});
            ac.seed = stage_seed;
            envs_.push_back(std::make_unique<ArcEnvironment>(ac));
            break;
        }
        case 2: {
            // Multi-object physics with increasing complexity
            stage_name += "_physics_multi_" + std::to_string(difficulty);
            PhysicsEnvironment::Config pc;
            pc.max_objects = static_cast<int>(std::min(difficulty + 3, std::size_t{15}));
            pc.gravity = -0.05 * static_cast<Real>(1 + difficulty % 5);
            pc.friction = 0.01 * static_cast<Real>(difficulty);
            pc.max_steps = max_steps;
            pc.seed = stage_seed;
            envs_.push_back(std::make_unique<PhysicsEnvironment>(pc));
            break;
        }
        case 3: {
            // Grid world with tight constraints (navigation)
            stage_name += "_maze_" + std::to_string(grid_size);
            GridWorld::Config gc;
            gc.rows = static_cast<Dim>(grid_size);
            gc.cols = static_cast<Dim>(grid_size);
            gc.num_colors = 3;  // sparse, forces precise navigation
            gc.max_steps = max_steps / 2;  // tighter budget
            gc.seed = stage_seed;
            envs_.push_back(std::make_unique<GridWorld>(gc));
            pass_threshold *= 1.2;
            break;
        }
        case 4: {
            // ARC with benchmark tasks (real patterns)
            stage_name += "_arc_bench_" + std::to_string(difficulty);
            ArcEnvironment::Config ac;
            ac.grid_rows = static_cast<Dim>(std::min(grid_size, 10));
            ac.grid_cols = static_cast<Dim>(std::min(grid_size, 10));
            ac.num_colors = num_colors;
            ac.puzzles_per_episode = std::min(difficulty + 5, std::size_t{20});
            ac.seed = stage_seed;
            auto env = std::make_unique<ArcEnvironment>(ac);
            env->use_benchmark_tasks();
            envs_.push_back(std::move(env));
            break;
        }
        case 5: {
            // Non-square grid (rectangular, tests generalization)
            stage_name += "_rect_" + std::to_string(grid_size);
            ArcEnvironment::Config ac;
            ac.grid_rows = static_cast<Dim>(std::max(grid_size / 2, 2));
            ac.grid_cols = static_cast<Dim>(grid_size);
            ac.num_colors = num_colors;
            ac.puzzles_per_episode = std::min(difficulty + 2, std::size_t{10});
            ac.seed = stage_seed;
            envs_.push_back(std::make_unique<ArcEnvironment>(ac));
            break;
        }
        default: {
            // Extreme physics (high gravity, many objects, chaotic)
            stage_name += "_chaos_" + std::to_string(difficulty);
            PhysicsEnvironment::Config pc;
            pc.max_objects = static_cast<int>(std::min(difficulty * 2, std::size_t{20}));
            pc.gravity = -0.3;
            pc.friction = 0.0;  // frictionless = harder
            pc.max_steps = max_steps * 2;
            pc.seed = stage_seed;
            envs_.push_back(std::make_unique<PhysicsEnvironment>(pc));
            break;
        }
    }

    stages_.push_back({stage_name, max_steps, pass_threshold,
                       std::max(std::size_t{3}, difficulty)});
}

} // namespace uik::agent
