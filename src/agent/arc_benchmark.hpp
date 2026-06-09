#pragma once

#include "agent/arc_loader.hpp"
#include "agent/agent_kernel.hpp"
#include "symbolic_descent/search_engine.hpp"
#include "symbolic_descent/dsl.hpp"
#include "symbolic_descent/mdl_evaluator.hpp"
#include "common/interfaces.hpp"
#include <string>
#include <vector>
#include <iostream>

namespace uik::agent {

// Benchmark runner: tests the kernel's symbolic descent engine on real ARC tasks.
// For each task: uses training pairs as a Dataset, searches for a program,
// then evaluates on test pairs.
struct BenchmarkResult {
    std::string task_id;
    bool solved          = false;    // exact match on test output
    double cell_accuracy = 0.0;     // cell-level accuracy on test output
    std::size_t search_iterations = 0;
    std::string best_program_desc;  // human-readable description
};

struct BenchmarkSummary {
    std::size_t total_tasks     = 0;
    std::size_t solved_exact    = 0;
    double avg_cell_accuracy    = 0.0;
    std::vector<BenchmarkResult> results;
};

class ArcBenchmark {
public:
    struct Config {
        std::size_t max_search_iterations = 200;
        std::size_t max_tasks             = 0;    // 0 = all tasks
        Real mdl_lambda                   = 1.0;
        bool verbose                      = false;
    };

    ArcBenchmark();
    explicit ArcBenchmark(Config config);

    // Run benchmark on all tasks in a directory
    BenchmarkSummary run(const std::filesystem::path& task_dir);

    // Run benchmark on a single task
    BenchmarkResult run_task(const ArcTask& task);

private:
    Config config_;
    symbolic_descent::DSL dsl_;

    using Dataset = ISearchEngine::Dataset;

    // Convert ARC training pairs to a Dataset for symbolic search
    Dataset task_to_dataset(const ArcTask& task) const;

    // Try to apply a found program to test input
    std::vector<std::vector<int>> apply_program(
        const ProgramPtr& prog, const std::vector<std::vector<int>>& input) const;
};

} // namespace uik::agent
