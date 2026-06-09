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

void ArcEnvironment::use_benchmark_tasks() {
    use_benchmark_ = true;
    generate_benchmark_puzzles();
}

void ArcEnvironment::generate_benchmark_puzzles() {
    puzzles_.clear();
    // 10 benchmark puzzles covering real ARC-AGI task patterns
    puzzles_.push_back(generate_pattern_completion_puzzle());
    puzzles_.push_back(generate_border_fill_puzzle());
    puzzles_.push_back(generate_symmetry_puzzle());
    puzzles_.push_back(generate_count_and_fill_puzzle());
    puzzles_.push_back(generate_gravity_puzzle());
    // Add new ARC-AGI benchmark patterns
    puzzles_.push_back(generate_crop_nonzero_puzzle());
    puzzles_.push_back(generate_upscale_puzzle());
    puzzles_.push_back(generate_denoise_puzzle());
    puzzles_.push_back(generate_flood_fill_puzzle());
    puzzles_.push_back(generate_diagonal_puzzle());
    // Add standard transforms as well
    puzzles_.push_back(generate_rotation_puzzle());
    puzzles_.push_back(generate_flip_puzzle());
    puzzles_.push_back(generate_color_map_puzzle());
    puzzles_.push_back(generate_fill_puzzle());
    puzzles_.push_back(generate_translate_puzzle());
}

ArcEnvironment::Puzzle ArcEnvironment::generate_pattern_completion_puzzle() {
    // Pattern: repeat a row pattern to fill the grid
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::uniform_int_distribution<int> color(1, std::max(2, config_.num_colors - 1));

    // Create a repeating pattern (e.g., [1, 2, 1, 2, ...])
    std::vector<int> pattern(cols);
    for (Dim c = 0; c < cols; ++c) pattern[c] = color(rng_);

    auto make_input = [&]() {
        std::vector<Real> data(rows * cols, 0.0);
        // Fill first row with pattern, rest is 0
        for (Dim c = 0; c < cols; ++c) {
            data[c] = static_cast<Real>(pattern[c]);
        }
        return Tensor({rows * cols}, std::move(data));
    };
    auto make_output = [&]() {
        std::vector<Real> data(rows * cols, 0.0);
        // Fill ALL rows with pattern
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                data[r * cols + c] = static_cast<Real>(pattern[c]);
            }
        }
        return Tensor({rows * cols}, std::move(data));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        // Regenerate pattern for each pair
        for (Dim c = 0; c < cols; ++c) pattern[c] = color(rng_);
        p.train_pairs.emplace_back(make_input(), make_output());
    }
    for (Dim c = 0; c < cols; ++c) pattern[c] = color(rng_);
    p.test_input = make_input();
    p.test_output = make_output();
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_border_fill_puzzle() {
    // Rule: fill the border cells with a specific color, keep interior
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::uniform_int_distribution<int> color(1, std::max(2, config_.num_colors - 1));
    int border_color = color(rng_);

    auto apply_border = [&](const Tensor& grid) {
        std::vector<Real> result(rows * cols);
        for (Dim i = 0; i < rows * cols; ++i) result[i] = grid.at(i);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
                    result[r * cols + c] = static_cast<Real>(border_color);
                }
            }
        }
        return Tensor({rows * cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_border(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_border(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_symmetry_puzzle() {
    // Rule: mirror the left half to the right half (horizontal symmetry)
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto apply_symmetry = [&](const Tensor& grid) {
        std::vector<Real> result(rows * cols);
        for (Dim i = 0; i < rows * cols; ++i) result[i] = grid.at(i);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = cols / 2; c < cols; ++c) {
                Dim mirror_c = cols - 1 - c;
                result[r * cols + c] = result[r * cols + mirror_c];
            }
        }
        return Tensor({rows * cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_symmetry(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_symmetry(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_count_and_fill_puzzle() {
    // Rule: count cells of a specific color, fill output row 0 with that count
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::uniform_int_distribution<int> color(1, std::max(2, config_.num_colors - 1));
    int target_color = color(rng_);

    auto apply_count_fill = [&](const Tensor& grid) {
        int count = 0;
        for (Dim i = 0; i < grid.flat_size(); ++i) {
            if (static_cast<int>(grid.at(i)) == target_color) ++count;
        }
        std::vector<Real> result(rows * cols, 0.0);
        // Fill first min(count, cols) cells of row 0
        for (int c = 0; c < std::min(count, static_cast<int>(cols)); ++c) {
            result[static_cast<Dim>(c)] = static_cast<Real>(target_color);
        }
        return Tensor({rows * cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_count_fill(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_count_fill(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_gravity_puzzle() {
    // Rule: non-zero cells "fall" to the bottom of each column
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto apply_gravity = [&](const Tensor& grid) {
        std::vector<Real> result(rows * cols, 0.0);
        for (Dim c = 0; c < cols; ++c) {
            // Collect non-zero values in this column
            std::vector<Real> non_zero;
            for (Dim r = 0; r < rows; ++r) {
                Real val = grid.at(r * cols + c);
                if (static_cast<int>(val) != 0) non_zero.push_back(val);
            }
            // Place at bottom
            Dim dest = rows - non_zero.size();
            for (std::size_t i = 0; i < non_zero.size(); ++i) {
                result[(dest + i) * cols + c] = non_zero[i];
            }
        }
        return Tensor({rows * cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_gravity(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_gravity(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_crop_nonzero_puzzle() {
    // Rule: find bounding box of non-zero cells, output is the cropped region
    // (padded to grid size with zeros)
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto make_sparse_grid = [&]() {
        std::vector<Real> data(rows * cols, 0.0);
        std::uniform_int_distribution<int> color(1, std::max(2, config_.num_colors - 1));
        std::uniform_int_distribution<Dim> rpos(1, rows - 2);
        std::uniform_int_distribution<Dim> cpos(1, cols - 2);
        // Place a small shape (2x2 block)
        Dim r0 = rpos(rng_);
        Dim c0 = cpos(rng_);
        int c = color(rng_);
        for (Dim dr = 0; dr < 2 && r0+dr < rows; ++dr)
            for (Dim dc = 0; dc < 2 && c0+dc < cols; ++dc)
                data[(r0+dr)*cols + (c0+dc)] = static_cast<Real>(c);
        return Tensor({rows * cols}, std::move(data));
    };

    auto apply_crop = [&](const Tensor& grid) {
        // Find bounding box then write cropped content to top-left
        Dim min_r = rows, max_r = 0, min_c = cols, max_c = 0;
        for (Dim r = 0; r < rows; ++r) {
            for (Dim cc = 0; cc < cols; ++cc) {
                if (static_cast<int>(grid.at(r*cols+cc)) != 0) {
                    min_r = std::min(min_r, r);
                    max_r = std::max(max_r, r);
                    min_c = std::min(min_c, cc);
                    max_c = std::max(max_c, cc);
                }
            }
        }
        std::vector<Real> result(rows*cols, 0.0);
        if (min_r <= max_r && min_c <= max_c) {
            for (Dim r = min_r; r <= max_r; ++r)
                for (Dim cc = min_c; cc <= max_c; ++cc)
                    result[(r-min_r)*cols + (cc-min_c)] = grid.at(r*cols+cc);
        }
        return Tensor({rows*cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = make_sparse_grid();
        p.train_pairs.emplace_back(input, apply_crop(input));
    }
    p.test_input = make_sparse_grid();
    p.test_output = apply_crop(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_upscale_puzzle() {
    // Rule: 2x upscale — each cell becomes a 2x2 block
    // Since grid stays same size, we effectively show top-left quarter input
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto apply_upscale = [&](const Tensor& grid) {
        std::vector<Real> result(rows*cols, 0.0);
        Dim half_r = rows / 2;
        Dim half_c = cols / 2;
        for (Dim r = 0; r < half_r; ++r) {
            for (Dim c = 0; c < half_c; ++c) {
                Real val = grid.at(r*cols + c);
                result[(2*r)*cols + 2*c] = val;
                if (2*c+1 < cols) result[(2*r)*cols + 2*c+1] = val;
                if (2*r+1 < rows) result[(2*r+1)*cols + 2*c] = val;
                if (2*r+1 < rows && 2*c+1 < cols) result[(2*r+1)*cols + 2*c+1] = val;
            }
        }
        return Tensor({rows*cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_upscale(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_upscale(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_denoise_puzzle() {
    // Rule: remove isolated cells (cells with no same-color neighbors)
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto apply_denoise = [&](const Tensor& grid) {
        std::vector<Real> result(rows*cols, 0.0);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                Real val = grid.at(r*cols+c);
                if (static_cast<int>(val) == 0) continue;
                // Check 4-neighbors for same color
                bool has_neighbor = false;
                if (r > 0 && static_cast<int>(grid.at((r-1)*cols+c)) == static_cast<int>(val)) has_neighbor = true;
                if (r+1 < rows && static_cast<int>(grid.at((r+1)*cols+c)) == static_cast<int>(val)) has_neighbor = true;
                if (c > 0 && static_cast<int>(grid.at(r*cols+c-1)) == static_cast<int>(val)) has_neighbor = true;
                if (c+1 < cols && static_cast<int>(grid.at(r*cols+c+1)) == static_cast<int>(val)) has_neighbor = true;
                if (has_neighbor) result[r*cols+c] = val;
            }
        }
        return Tensor({rows*cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_denoise(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_denoise(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_flood_fill_puzzle() {
    // Rule: flood fill from top-left corner with a specific color
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;
    std::uniform_int_distribution<int> color(1, std::max(2, config_.num_colors - 1));
    int fill_color = color(rng_);

    auto apply_flood = [&](const Tensor& grid) {
        std::vector<Real> result(rows*cols);
        for (Dim i = 0; i < rows*cols; ++i) result[i] = grid.at(i);
        // BFS from (0,0)
        int start_val = static_cast<int>(grid.at(0));
        if (start_val == fill_color) {
            return Tensor({rows*cols}, std::move(result));
        }
        std::vector<std::pair<Dim,Dim>> stack;
        stack.push_back({0, 0});
        std::vector<bool> visited(rows*cols, false);
        while (!stack.empty()) {
            auto [r, c] = stack.back();
            stack.pop_back();
            if (r >= rows || c >= cols) continue;
            Dim idx = r*cols+c;
            if (visited[idx]) continue;
            if (static_cast<int>(result[idx]) != start_val) continue;
            visited[idx] = true;
            result[idx] = static_cast<Real>(fill_color);
            if (r > 0) stack.push_back({r-1, c});
            if (r+1 < rows) stack.push_back({r+1, c});
            if (c > 0) stack.push_back({r, c-1});
            if (c+1 < cols) stack.push_back({r, c+1});
        }
        return Tensor({rows*cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_flood(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_flood(p.test_input);
    return p;
}

ArcEnvironment::Puzzle ArcEnvironment::generate_diagonal_puzzle() {
    // Rule: extract the main diagonal and fill the rest with 0
    Dim rows = config_.grid_rows;
    Dim cols = config_.grid_cols;

    auto apply_diagonal = [&](const Tensor& grid) {
        std::vector<Real> result(rows*cols, 0.0);
        Dim diag_len = std::min(rows, cols);
        for (Dim i = 0; i < diag_len; ++i) {
            result[i*cols + i] = grid.at(i*cols + i);
        }
        return Tensor({rows*cols}, std::move(result));
    };

    Puzzle p;
    for (int i = 0; i < 3; ++i) {
        Tensor input = random_grid();
        p.train_pairs.emplace_back(input, apply_diagonal(input));
    }
    p.test_input = random_grid();
    p.test_output = apply_diagonal(p.test_input);
    return p;
}

} // namespace uik::agent
