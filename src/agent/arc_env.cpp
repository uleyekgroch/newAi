#include "agent/arc_env.hpp"
#include <cmath>
#include <algorithm>

namespace uik::agent {

ArcEnvironment::ArcEnvironment()
    : ArcEnvironment(Config{})
{}

ArcEnvironment::ArcEnvironment(Config config)
    : config_(config), rng_(config.seed)
{
    generate_puzzles();
}

Observation ArcEnvironment::reset() {
    puzzle_idx_ = 0;
    step_count_ = 0;
    puzzles_solved_ = 0;
    submitted_ = false;
    Dim grid_size = config_.grid_rows * config_.grid_cols;
    agent_output_ = Tensor({grid_size}, 0.0);
    return Observation{flatten_puzzle_obs()};
}

StepResult ArcEnvironment::step(const Action& action) {
    ++step_count_;

    Dim grid_size = config_.grid_rows * config_.grid_cols;
    int act = action.id;

    // Actions: 0..grid_size-1 = set cell value (cycle color)
    // grid_size = submit answer
    // grid_size+1 = next puzzle (skip)
    if (act >= 0 && static_cast<Dim>(act) < grid_size) {
        // Modify a cell in agent's output: cycle color at position
        Real cur = agent_output_.at(static_cast<Dim>(act));
        int next_color = (static_cast<int>(cur) + 1) % config_.num_colors;
        agent_output_.at(static_cast<Dim>(act)) = static_cast<Real>(next_color);

        return StepResult{Observation{flatten_puzzle_obs()}, 0.0, false};
    }

    if (static_cast<Dim>(act) == grid_size) {
        // Submit: compare agent output to expected
        Real sim = compute_similarity(agent_output_,
                                       puzzles_[puzzle_idx_].test_output);
        Real reward = sim;
        if (sim > 0.95) {
            ++puzzles_solved_;
            reward = 1.0;
        }

        // Move to next puzzle
        submitted_ = true;
        if (puzzle_idx_ + 1 < puzzles_.size()) {
            ++puzzle_idx_;
            agent_output_ = Tensor({grid_size}, 0.0);
            submitted_ = false;
            return StepResult{Observation{flatten_puzzle_obs()}, reward, false};
        }
        return StepResult{Observation{flatten_puzzle_obs()}, reward, true};
    }

    // Skip to next puzzle
    if (puzzle_idx_ + 1 < puzzles_.size()) {
        ++puzzle_idx_;
        agent_output_ = Tensor({grid_size}, 0.0);
        submitted_ = false;
        return StepResult{Observation{flatten_puzzle_obs()}, -0.1, false};
    }
    return StepResult{Observation{flatten_puzzle_obs()}, -0.1, true};
}

int ArcEnvironment::action_space_size() const {
    return static_cast<int>(config_.grid_rows * config_.grid_cols) + 2;
}

void ArcEnvironment::generate_puzzles() {
    puzzles_.clear();
    for (std::size_t i = 0; i < config_.puzzles_per_episode; ++i) {
        std::uniform_int_distribution<int> type(0, 4);
        switch (type(rng_)) {
            case 0: puzzles_.push_back(generate_rotation_puzzle()); break;
            case 1: puzzles_.push_back(generate_flip_puzzle()); break;
            case 2: puzzles_.push_back(generate_color_map_puzzle()); break;
            case 3: puzzles_.push_back(generate_fill_puzzle()); break;
            default: puzzles_.push_back(generate_translate_puzzle()); break;
        }
    }
}

Tensor ArcEnvironment::random_grid() {
    Dim size = config_.grid_rows * config_.grid_cols;
    std::vector<Real> data(size);
    std::uniform_int_distribution<int> color(0, config_.num_colors - 1);
    for (auto& v : data) v = static_cast<Real>(color(rng_));
    return Tensor({size}, std::move(data));
}

ArcEnvironment::Puzzle ArcEnvironment::generate_rotation_puzzle() {
    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        Tensor output = apply_rotation(input);
        p.train_pairs.emplace_back(input, output);
    }
    p.test_input = random_grid();
    p.test_output = apply_rotation(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_flip_puzzle() {
    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        Tensor output = apply_flip_h(input);
        p.train_pairs.emplace_back(input, output);
    }
    p.test_input = random_grid();
    p.test_output = apply_flip_h(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_color_map_puzzle() {
    std::uniform_int_distribution<int> color(0, config_.num_colors - 1);
    int from_c = color(rng_);
    int to_c = color(rng_);
    while (to_c == from_c) to_c = color(rng_);

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        Tensor output = apply_color_map(input, from_c, to_c);
        p.train_pairs.emplace_back(input, output);
    }
    p.test_input = random_grid();
    p.test_output = apply_color_map(p.test_input, from_c, to_c);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_fill_puzzle() {
    std::uniform_int_distribution<int> color(0, config_.num_colors - 1);
    int fill_val = color(rng_);

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        Tensor output = apply_fill(input, fill_val);
        p.train_pairs.emplace_back(input, output);
    }
    p.test_input = random_grid();
    p.test_output = apply_fill(p.test_input, fill_val);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_translate_puzzle() {
    std::uniform_int_distribution<int> shift(-2, 2);
    int dx = shift(rng_);
    int dy = shift(rng_);

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        Tensor output = apply_translate(input, dx, dy);
        p.train_pairs.emplace_back(input, output);
    }
    p.test_input = random_grid();
    p.test_output = apply_translate(p.test_input, dx, dy);
    return p;
}

Tensor ArcEnvironment::apply_rotation(const Tensor& grid) {
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::vector<Real> result(rows * cols, 0.0);
    for (Dim r = 0; r < rows; ++r) {
        for (Dim c = 0; c < cols; ++c) {
            Dim new_r = c;
            Dim new_c = (rows - 1) - r;
            if (new_r < rows && new_c < cols) {
                result[new_r * cols + new_c] = grid.at(r * cols + c);
            }
        }
    }
    return Tensor({rows * cols}, std::move(result));
}

Tensor ArcEnvironment::apply_flip_h(const Tensor& grid) {
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::vector<Real> result(rows * cols, 0.0);
    for (Dim r = 0; r < rows; ++r) {
        for (Dim c = 0; c < cols; ++c) {
            result[r * cols + (cols - 1 - c)] = grid.at(r * cols + c);
        }
    }
    return Tensor({rows * cols}, std::move(result));
}

Tensor ArcEnvironment::apply_color_map(const Tensor& grid,
                                         int from_c, int to_c) {
    Dim size = grid.flat_size();
    std::vector<Real> result(size);
    for (Dim i = 0; i < size; ++i) {
        Real val = grid.at(i);
        result[i] = (static_cast<int>(val) == from_c)
                    ? static_cast<Real>(to_c) : val;
    }
    return Tensor({size}, std::move(result));
}

Tensor ArcEnvironment::apply_fill(const Tensor& grid, int val) {
    Dim size = grid.flat_size();
    std::vector<Real> result(size, static_cast<Real>(val));
    (void)grid;
    return Tensor({size}, std::move(result));
}

Tensor ArcEnvironment::apply_translate(const Tensor& grid, int dx, int dy) {
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::vector<Real> result(rows * cols, 0.0);
    for (Dim r = 0; r < rows; ++r) {
        for (Dim c = 0; c < cols; ++c) {
            int src_r = static_cast<int>(r) - dy;
            int src_c = static_cast<int>(c) - dx;
            if (src_r >= 0 && static_cast<Dim>(src_r) < rows &&
                src_c >= 0 && static_cast<Dim>(src_c) < cols) {
                result[r * cols + c] = grid.at(
                    static_cast<Dim>(src_r) * cols + static_cast<Dim>(src_c));
            }
        }
    }
    return Tensor({rows * cols}, std::move(result));
}

Tensor ArcEnvironment::flatten_puzzle_obs() const {
    Dim grid_size = config_.grid_rows * config_.grid_cols;
    // Observation: [test_input | train_pair_0_input | train_pair_0_output | agent_output]
    // Flatten to a single vector for the world model
    const auto& puzzle = puzzles_[puzzle_idx_];
    Dim pair_count = std::min(puzzle.train_pairs.size(), std::size_t{1});
    Dim obs_size = grid_size * (2 + 2 * pair_count); // test_in + agent_out + pairs

    std::vector<Real> obs(obs_size, 0.0);
    Dim offset = 0;

    // Test input
    for (Dim i = 0; i < grid_size && i < puzzle.test_input.flat_size(); ++i) {
        obs[offset + i] = puzzle.test_input.at(i);
    }
    offset += grid_size;

    // First train pair
    if (!puzzle.train_pairs.empty()) {
        const auto& [tin, tout] = puzzle.train_pairs[0];
        for (Dim i = 0; i < grid_size && i < tin.flat_size(); ++i) {
            obs[offset + i] = tin.at(i);
        }
        offset += grid_size;
        for (Dim i = 0; i < grid_size && i < tout.flat_size(); ++i) {
            obs[offset + i] = tout.at(i);
        }
        offset += grid_size;
    }

    // Agent's current output
    for (Dim i = 0; i < grid_size && i < agent_output_.flat_size(); ++i) {
        obs[offset + i] = agent_output_.at(i);
    }

    return Tensor({obs_size}, std::move(obs));
}

Real ArcEnvironment::compute_similarity(const Tensor& a, const Tensor& b) const {
    if (a.flat_size() != b.flat_size()) return 0.0;
    Dim matches = 0;
    for (Dim i = 0; i < a.flat_size(); ++i) {
        if (std::abs(a.at(i) - b.at(i)) < 0.5) ++matches;
    }
    return static_cast<Real>(matches) / static_cast<Real>(a.flat_size());
}

} // namespace uik::agent
