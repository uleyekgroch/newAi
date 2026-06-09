#pragma once

#include "common/types.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace uik::agent {

// Loads real ARC-AGI puzzle data from JSON files.
// JSON format: {"train": [{"input": [[int,...]], "output": [[int,...]]},...],
//               "test":  [{"input": [[int,...]], "output": [[int,...]]},...]}
struct ArcTask {
    struct Pair {
        std::vector<std::vector<int>> input;
        std::vector<std::vector<int>> output;
    };
    std::string id;              // filename without extension
    std::vector<Pair> train;     // demonstration pairs
    std::vector<Pair> test;      // test pairs (agent must solve)
};

class ArcLoader {
public:
    // Load a single task from a JSON file
    static ArcTask load_task(const std::filesystem::path& path);

    // Load all tasks from a directory
    static std::vector<ArcTask> load_directory(const std::filesystem::path& dir,
                                                std::size_t max_tasks = 0);

    // Convert 2D grid to flat Tensor (row-major, values as Real)
    static Tensor grid_to_tensor(const std::vector<std::vector<int>>& grid);

    // Convert flat Tensor back to 2D grid
    static std::vector<std::vector<int>> tensor_to_grid(const Tensor& t,
                                                         Dim rows, Dim cols);

    // Check if agent output matches expected output exactly
    static bool check_exact_match(const std::vector<std::vector<int>>& output,
                                   const std::vector<std::vector<int>>& expected);

    // Compute cell-level accuracy
    static double cell_accuracy(const std::vector<std::vector<int>>& output,
                                 const std::vector<std::vector<int>>& expected);
};

} // namespace uik::agent
