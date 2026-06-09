#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <unordered_map>

namespace uik::symbolic_descent {

// Domain-Specific Language: defines primitive operations on grid-like tensors.
// Phase 1: grid transforms (rotate, flip, translate, fill, map_color)
// Phase 3: general computation (add, multiply, modulo, threshold, count,
//           repeat/loop, fold/reduce, zip, filter, store/load variables)
class DSL {
public:
    // Execute a program on input data
    [[nodiscard]] Tensor execute(const ProgramPtr& program,
                                  const Tensor& input) const;

    // List of available primitive OpKinds
    [[nodiscard]] static std::vector<OpKind> primitives();

    // Extended primitives including Phase 3 ops
    [[nodiscard]] static std::vector<OpKind> extended_primitives();

    // Max loop iterations to prevent infinite loops
    static constexpr int MAX_LOOP_ITERATIONS = 100;

    // Max variable slots
    static constexpr int MAX_VARIABLE_SLOTS = 16;

private:
    // Mutable variable store for Store/Load ops
    mutable std::unordered_map<int, Tensor> variable_store_;

    [[nodiscard]] Tensor exec_identity(const Tensor& input) const;
    [[nodiscard]] Tensor exec_constant(int value, const Tensor& input) const;
    [[nodiscard]] Tensor exec_rotate90(const Tensor& input) const;
    [[nodiscard]] Tensor exec_flip_h(const Tensor& input) const;
    [[nodiscard]] Tensor exec_flip_v(const Tensor& input) const;
    [[nodiscard]] Tensor exec_translate(int dx, int dy, const Tensor& input) const;
    [[nodiscard]] Tensor exec_fill(int from_val, int to_val,
                                    const Tensor& input) const;
    [[nodiscard]] Tensor exec_map_color(int from_c, int to_c,
                                         const Tensor& input) const;

    // Phase 3 ops
    [[nodiscard]] Tensor exec_add(int offset, const Tensor& input) const;
    [[nodiscard]] Tensor exec_multiply(int factor, const Tensor& input) const;
    [[nodiscard]] Tensor exec_modulo(int divisor, const Tensor& input) const;
    [[nodiscard]] Tensor exec_threshold(int thresh, int val,
                                         const Tensor& input) const;
    [[nodiscard]] Tensor exec_count(int target, const Tensor& input) const;
    [[nodiscard]] Tensor exec_repeat(const ProgramPtr& child, int times,
                                      const Tensor& input) const;
    [[nodiscard]] Tensor exec_fold(const ProgramPtr& child,
                                    const Tensor& input) const;
    [[nodiscard]] Tensor exec_zip(const ProgramPtr& child0,
                                   const ProgramPtr& child1,
                                   const Tensor& input) const;
    [[nodiscard]] Tensor exec_filter(int target, const Tensor& input) const;
    [[nodiscard]] Tensor exec_store(int slot, const Tensor& input) const;
    [[nodiscard]] Tensor exec_load(int slot, const Tensor& input) const;
};

} // namespace uik::symbolic_descent
