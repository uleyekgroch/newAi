#pragma once

#include "common/types.hpp"
#include "common/program.hpp"

namespace uik::symbolic_descent {

// Domain-Specific Language: defines primitive operations on grid-like tensors.
// Each operation transforms an input Tensor into an output Tensor.
class DSL {
public:
    // Execute a program on input data
    [[nodiscard]] Tensor execute(const ProgramPtr& program,
                                  const Tensor& input) const;

    // List of available primitive OpKinds
    [[nodiscard]] static std::vector<OpKind> primitives();

private:
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
};

} // namespace uik::symbolic_descent
