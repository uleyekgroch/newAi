#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <random>
#include <vector>
#include <functional>

namespace uik::symbolic_descent {

// Manages the discrete program space: generation, mutation, crossover.
// Supports guided search via type-aware mutations and analogy-based operators.
class ProgramSpace {
public:
    explicit ProgramSpace(int max_depth = 4, int color_range = 10,
                           int grid_range = 5, unsigned seed = 42);

    [[nodiscard]] ProgramPtr random_program(int depth = 0);
    [[nodiscard]] ProgramPtr mutate(const ProgramPtr& program);
    [[nodiscard]] ProgramPtr crossover(const ProgramPtr& a, const ProgramPtr& b);
    [[nodiscard]] std::vector<ProgramPtr> neighborhood(const ProgramPtr& program,
                                                        std::size_t count = 10);

    // Guided search operators (1e improvement)

    // Type-aware mutation: only mutate to compatible op kinds
    [[nodiscard]] ProgramPtr type_aware_mutate(const ProgramPtr& program);

    // Analogy-based crossover: find structurally similar subtrees
    [[nodiscard]] ProgramPtr analogy_crossover(const ProgramPtr& a,
                                                const ProgramPtr& b);

    // Guided neighborhood: biased toward promising mutation types
    // Uses scores from previous evaluations to weight operator selection
    [[nodiscard]] std::vector<ProgramPtr> guided_neighborhood(
        const ProgramPtr& program,
        const std::vector<Real>& op_weights,
        std::size_t count = 10);

    // Parameter-only perturbation: keep structure, tweak params
    [[nodiscard]] ProgramPtr param_perturbation(const ProgramPtr& program);

    // Simplification: try to reduce program without changing behavior
    [[nodiscard]] ProgramPtr simplify(const ProgramPtr& program);

    // Template-guided edit: learn patterns from successful programs
    // and apply them as targeted mutations (non-LLM alternative to LLM_guided_edit)
    void add_template(const ProgramPtr& successful_program);
    [[nodiscard]] ProgramPtr template_guided_edit(const ProgramPtr& program);
    [[nodiscard]] std::size_t template_count() const { return templates_.size(); }

private:
    int max_depth_;
    int color_range_;
    int grid_range_;
    std::mt19937 rng_;

    [[nodiscard]] ProgramPtr random_leaf();
    [[nodiscard]] ProgramPtr random_primitive();
    [[nodiscard]] ProgramPtr replace_random_subtree(const ProgramPtr& program);
    [[nodiscard]] ProgramPtr pick_random_node(const ProgramPtr& program);

    // Helper: get compatible op kinds for a given op
    static std::vector<OpKind> compatible_ops(OpKind kind);

    // Helper: count structure similarity
    static Real structural_similarity(const ProgramPtr& a, const ProgramPtr& b);

    // Template library: patterns extracted from successful programs
    struct Template {
        ProgramPtr pattern;    // subtree pattern
        Real fitness;          // fitness of source program
    };
    std::vector<Template> templates_;
    static constexpr std::size_t MAX_TEMPLATES = 50;

    // Extract useful subtrees from a program
    void extract_subtrees(const ProgramPtr& prog, std::vector<ProgramPtr>& out,
                          int depth = 0);
};

} // namespace uik::symbolic_descent
