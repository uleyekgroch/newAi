#include <gtest/gtest.h>
#include "agent/arc_loader.hpp"
#include "agent/arc_benchmark.hpp"
#include <filesystem>
#include <iostream>

using namespace uik;
using namespace uik::agent;

namespace fs = std::filesystem;

// Path to ARC training data (relative to build dir, or absolute)
static const fs::path ARC_TRAINING_DIR = fs::path(ARC_DATA_DIR) / "arc_training";
static const fs::path ARC_EVALUATION_DIR = fs::path(ARC_DATA_DIR) / "arc_evaluation";

// ═══════════════════════════════════════════════════════════
// ARC Loader Tests
// ═══════════════════════════════════════════════════════════

TEST(ArcLoader, LoadSingleTask) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";
    auto tasks = ArcLoader::load_directory(ARC_TRAINING_DIR, 1);
    ASSERT_FALSE(tasks.empty());
    const auto& task = tasks[0];
    EXPECT_FALSE(task.id.empty());
    EXPECT_FALSE(task.train.empty());
    EXPECT_FALSE(task.test.empty());

    // Verify grid structure
    const auto& first_pair = task.train[0];
    EXPECT_GT(first_pair.input.size(), 0u);
    EXPECT_GT(first_pair.input[0].size(), 0u);
    EXPECT_GT(first_pair.output.size(), 0u);
}

TEST(ArcLoader, LoadMultipleTasks) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";
    auto tasks = ArcLoader::load_directory(ARC_TRAINING_DIR, 10);
    EXPECT_EQ(tasks.size(), 10u);
    for (const auto& task : tasks) {
        EXPECT_FALSE(task.id.empty());
        EXPECT_FALSE(task.train.empty());
    }
}

TEST(ArcLoader, GridToTensorRoundTrip) {
    std::vector<std::vector<int>> grid = {
        {1, 2, 3},
        {4, 5, 6}
    };
    Tensor t = ArcLoader::grid_to_tensor(grid);
    EXPECT_EQ(t.flat_size(), 6u);
    EXPECT_DOUBLE_EQ(t.at(0), 1.0);
    EXPECT_DOUBLE_EQ(t.at(5), 6.0);

    auto back = ArcLoader::tensor_to_grid(t, 2, 3);
    EXPECT_TRUE(ArcLoader::check_exact_match(back, grid));
}

TEST(ArcLoader, CellAccuracy) {
    std::vector<std::vector<int>> output = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> expected = {{1, 2, 0}, {4, 5, 6}};
    double acc = ArcLoader::cell_accuracy(output, expected);
    EXPECT_NEAR(acc, 5.0 / 6.0, 1e-6);
}

TEST(ArcLoader, ExactMatch) {
    std::vector<std::vector<int>> a = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> b = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> c = {{1, 2}, {3, 5}};
    EXPECT_TRUE(ArcLoader::check_exact_match(a, b));
    EXPECT_FALSE(ArcLoader::check_exact_match(a, c));
}

TEST(ArcLoader, LoadAllTraining) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";
    auto tasks = ArcLoader::load_directory(ARC_TRAINING_DIR);
    EXPECT_GE(tasks.size(), 395u);  // Some files may have edge-case formatting
}

// ═══════════════════════════════════════════════════════════
// ARC Benchmark on Real Data
// ═══════════════════════════════════════════════════════════

TEST(ArcBenchmark, RunOnSmallSubset) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";

    ArcBenchmark::Config cfg;
    cfg.max_search_iterations = 100;
    cfg.max_tasks = 20;
    cfg.verbose = true;

    ArcBenchmark bench(cfg);
    auto summary = bench.run(ARC_TRAINING_DIR);

    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "ARC Benchmark Results (20 tasks)\n";
    std::cout << "══════════════════════════════════════\n";
    std::cout << "Tasks tested:      " << summary.total_tasks << "\n";
    std::cout << "Exact matches:     " << summary.solved_exact << "\n";
    std::cout << "Avg cell accuracy: " << summary.avg_cell_accuracy << "\n";
    std::cout << "══════════════════════════════════════\n\n";

    EXPECT_EQ(summary.total_tasks, 20u);
    // We don't require solving any — this measures actual capability honestly
}

TEST(ArcBenchmark, RunOn100Tasks) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";

    ArcBenchmark::Config cfg;
    cfg.max_search_iterations = 150;
    cfg.max_tasks = 100;
    cfg.verbose = false;

    ArcBenchmark bench(cfg);
    auto summary = bench.run(ARC_TRAINING_DIR);

    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "ARC Benchmark Results (100 tasks)\n";
    std::cout << "══════════════════════════════════════\n";
    std::cout << "Tasks tested:      " << summary.total_tasks << "\n";
    std::cout << "Exact matches:     " << summary.solved_exact << "\n";
    std::cout << "Solve rate:        " << (summary.total_tasks > 0 ?
        100.0 * summary.solved_exact / summary.total_tasks : 0.0) << "%\n";
    std::cout << "Avg cell accuracy: " << summary.avg_cell_accuracy << "\n";
    std::cout << "══════════════════════════════════════\n\n";

    // Report top results
    std::vector<BenchmarkResult> sorted = summary.results;
    std::sort(sorted.begin(), sorted.end(),
              [](const BenchmarkResult& a, const BenchmarkResult& b) {
                  return a.cell_accuracy > b.cell_accuracy;
              });
    std::cout << "Top 10 by cell accuracy:\n";
    for (std::size_t i = 0; i < 10 && i < sorted.size(); ++i) {
        std::cout << "  " << sorted[i].task_id
                  << " | acc=" << sorted[i].cell_accuracy
                  << " | solved=" << (sorted[i].solved ? "YES" : "no")
                  << " | prog=" << sorted[i].best_program_desc << "\n";
    }

    EXPECT_EQ(summary.total_tasks, 100u);
}

// ═══════════════════════════════════════════════════════════
// End-to-end: AgentKernel on ARC tasks
// ═══════════════════════════════════════════════════════════

TEST(ArcEndToEnd, AgentKernelOnArcTask) {
    if (!fs::exists(ARC_TRAINING_DIR)) GTEST_SKIP() << "ARC data not found";

    auto tasks = ArcLoader::load_directory(ARC_TRAINING_DIR, 5);
    ASSERT_FALSE(tasks.empty());

    // Use the first task — configure agent for it
    const auto& task = tasks[0];
    auto first_input = ArcLoader::grid_to_tensor(task.train[0].input);
    Dim obs_dim = first_input.flat_size();

    AgentKernel::Config acfg;
    acfg.wm_config.input_dim = obs_dim;
    acfg.wm_config.latent_dim = std::min(obs_dim, Dim{16});
    acfg.wm_config.action_space = 10;  // ARC: 10 colors
    acfg.planner_config.action_space = 10;
    AgentKernel agent(acfg);

    // Feed training examples as observations
    for (const auto& pair : task.train) {
        Tensor input_t = ArcLoader::grid_to_tensor(pair.input);
        Observation obs{input_t};
        Action action = agent.step(obs, 0.0);
        (void)action;  // Agent processes and learns from observations
    }

    EXPECT_GT(agent.step_count(), 0u);
    std::cout << "[E2E] Agent processed " << task.train.size()
              << " ARC training examples from task " << task.id << "\n";
}
