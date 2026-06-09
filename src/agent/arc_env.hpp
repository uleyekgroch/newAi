#pragma once

#include "common/interfaces.hpp"
#include <vector>
#include <random>
#include <functional>

namespace uik::agent {

// ARC-AGI style environment: present grid transformation puzzles.
// Each puzzle has input grids and expected output grids.
// The agent must discover the transformation rule (program) from examples,
// then apply it to a test input.
class ArcEnvironment final : public IEnvironment {
public:
    struct Puzzle {
        std::vector<std::pair<Tensor, Tensor>> train_pairs;  // example I/O
        Tensor test_input;
        Tensor test_output;
    };

    struct Config {
        Dim grid_rows   = 5;
        Dim grid_cols   = 5;
        int num_colors  = 10;
        std::size_t puzzles_per_episode = 5;
        unsigned seed   = 42;
    };

    ArcEnvironment();
    explicit ArcEnvironment(Config config);

    Observation reset() override;
    StepResult step(const Action& action) override;
    int action_space_size() const override;

    [[nodiscard]] const Puzzle& current_puzzle() const { return puzzles_[puzzle_idx_]; }
    [[nodiscard]] std::size_t puzzles_solved() const { return puzzles_solved_; }
    [[nodiscard]] std::size_t total_puzzles() const { return puzzles_.size(); }

private:
    Config config_;
    std::mt19937 rng_;
    std::vector<Puzzle> puzzles_;
    std::size_t puzzle_idx_ = 0;
    std::size_t step_count_ = 0;
    std::size_t puzzles_solved_ = 0;
    Tensor agent_output_;      // agent's current answer
    bool submitted_ = false;

    void generate_puzzles();
    Puzzle generate_rotation_puzzle();
    Puzzle generate_flip_puzzle();
    Puzzle generate_color_map_puzzle();
    Puzzle generate_fill_puzzle();
    Puzzle generate_translate_puzzle();
    Tensor random_grid();
    Tensor apply_rotation(const Tensor& grid);
    Tensor apply_flip_h(const Tensor& grid);
    Tensor apply_color_map(const Tensor& grid, int from_c, int to_c);
    Tensor apply_fill(const Tensor& grid, int val);
    Tensor apply_translate(const Tensor& grid, int dx, int dy);
    Tensor flatten_puzzle_obs() const;
    Real compute_similarity(const Tensor& a, const Tensor& b) const;

    // Built-in ARC-AGI benchmark tasks (real patterns from the public dataset)
    void generate_benchmark_puzzles();
    Puzzle generate_pattern_completion_puzzle();   // complete a repeating pattern
    Puzzle generate_border_fill_puzzle();           // fill border with specific color
    Puzzle generate_symmetry_puzzle();              // mirror to make symmetric
    Puzzle generate_count_and_fill_puzzle();        // count objects → fill output
    Puzzle generate_gravity_puzzle();               // objects fall down
    Puzzle generate_crop_nonzero_puzzle();            // crop to bounding box of nonzero
    Puzzle generate_upscale_puzzle();                 // 2× upscale each cell
    Puzzle generate_denoise_puzzle();                 // remove isolated noise cells
    Puzzle generate_flood_fill_puzzle();              // flood fill from a seed
    Puzzle generate_diagonal_puzzle();                // extract/fill diagonal

public:
    // Use built-in ARC-AGI benchmark tasks instead of random ones
    void use_benchmark_tasks();
    [[nodiscard]] bool using_benchmark() const { return use_benchmark_; }

private:
    bool use_benchmark_ = false;
};

} // namespace uik::agent
