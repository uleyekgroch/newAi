#pragma once

#include "common/program.hpp"
#include <random>
#include <vector>

namespace uik::symbolic_descent {

// Manages the discrete program space: generation, mutation, crossover.
class ProgramSpace {
public:
    explicit ProgramSpace(int max_depth = 4, int color_range = 10,
                           int grid_range = 5, unsigned seed = 42);

    [[nodiscard]] ProgramPtr random_program(int depth = 0);
    [[nodiscard]] ProgramPtr mutate(const ProgramPtr& program);
    [[nodiscard]] ProgramPtr crossover(const ProgramPtr& a, const ProgramPtr& b);
    [[nodiscard]] std::vector<ProgramPtr> neighborhood(const ProgramPtr& program,
                                                        std::size_t count = 10);

private:
    int max_depth_;
    int color_range_;
    int grid_range_;
    std::mt19937 rng_;

    [[nodiscard]] ProgramPtr random_leaf();
    [[nodiscard]] ProgramPtr random_primitive();
    [[nodiscard]] ProgramPtr replace_random_subtree(const ProgramPtr& program);
    [[nodiscard]] ProgramPtr pick_random_node(const ProgramPtr& program);
};

} // namespace uik::symbolic_descent
