#include "agent/arc_benchmark.hpp"
#include "common/program.hpp"
#include "common/interfaces.hpp"
#include <numeric>

namespace uik::agent {

ArcBenchmark::ArcBenchmark() : config_{} {}
ArcBenchmark::ArcBenchmark(Config config) : config_(config) {}

BenchmarkSummary ArcBenchmark::run(const std::filesystem::path& task_dir) {
    auto tasks = ArcLoader::load_directory(task_dir, config_.max_tasks);
    BenchmarkSummary summary;
    summary.total_tasks = tasks.size();

    double total_accuracy = 0.0;

    for (const auto& task : tasks) {
        auto result = run_task(task);
        if (result.solved) ++summary.solved_exact;
        total_accuracy += result.cell_accuracy;

        if (config_.verbose) {
            std::cout << "[ARC] " << result.task_id
                      << " | solved=" << (result.solved ? "YES" : "no")
                      << " | cell_acc=" << result.cell_accuracy
                      << " | iters=" << result.search_iterations
                      << "\n";
        }

        summary.results.push_back(std::move(result));
    }

    summary.avg_cell_accuracy = summary.total_tasks > 0
        ? total_accuracy / static_cast<double>(summary.total_tasks)
        : 0.0;

    return summary;
}

BenchmarkResult ArcBenchmark::run_task(const ArcTask& task) {
    BenchmarkResult result;
    result.task_id = task.id;

    if (task.train.empty() || task.test.empty()) return result;

    // Build dataset from training pairs
    Dataset data = task_to_dataset(task);
    if (data.empty()) return result;

    // Search for a program
    symbolic_descent::SearchEngine::Config sc;
    sc.population_size = 100;
    sc.elite_count = 10;
    sc.neighborhood_size = 15;
    sc.lambda = config_.mdl_lambda;
    sc.max_depth = 5;
    sc.beam_width = 12;
    sc.beam_expansions = 8;
    sc.use_beam_search = true;
    symbolic_descent::SearchEngine engine(sc);

    auto found = engine.search(data, config_.max_search_iterations);
    result.search_iterations = engine.iterations_used();

    if (!found.has_value()) return result;

    result.best_program_desc = (*found)->to_string();

    // Evaluate on test pairs
    bool all_solved = true;
    double total_acc = 0.0;
    std::size_t test_count = 0;

    for (const auto& test_pair : task.test) {
        auto output = apply_program(*found, test_pair.input);
        bool match = ArcLoader::check_exact_match(output, test_pair.output);
        double acc = ArcLoader::cell_accuracy(output, test_pair.output);

        if (!match) all_solved = false;
        total_acc += acc;
        ++test_count;
    }

    result.solved = all_solved;
    result.cell_accuracy = test_count > 0 ? total_acc / static_cast<double>(test_count) : 0.0;

    return result;
}

ArcBenchmark::Dataset ArcBenchmark::task_to_dataset(const ArcTask& task) const {
    Dataset data;
    for (const auto& pair : task.train) {
        Tensor input = ArcLoader::grid_to_tensor(pair.input);
        Tensor output = ArcLoader::grid_to_tensor(pair.output);
        data.push_back({input, output});
    }
    return data;
}

std::vector<std::vector<int>> ArcBenchmark::apply_program(
    const ProgramPtr& prog, const std::vector<std::vector<int>>& input) const
{
    Tensor input_tensor = ArcLoader::grid_to_tensor(input);
    try {
        Tensor output_tensor = dsl_.execute(prog, input_tensor);
        // Determine output grid dimensions from tensor
        Dim total = output_tensor.flat_size();
        if (total == 0) return {};

        // Try to match the input dimensions first
        Dim in_rows = input.size();
        Dim in_cols = input.empty() ? 0 : input[0].size();

        if (total == in_rows * in_cols) {
            return ArcLoader::tensor_to_grid(output_tensor, in_rows, in_cols);
        }

        // Fallback: treat as 1D → reshape to single row
        return ArcLoader::tensor_to_grid(output_tensor, 1, total);
    } catch (...) {
        return {};
    }
}

} // namespace uik::agent
