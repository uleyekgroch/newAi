#include "symbolic_descent/dsl.hpp"
#include <cmath>
#include <stdexcept>

namespace uik::symbolic_descent {

std::vector<OpKind> DSL::primitives() {
    return {
        OpKind::Identity, OpKind::Constant, OpKind::Rotate90,
        OpKind::FlipH, OpKind::FlipV, OpKind::Translate,
        OpKind::Fill, OpKind::MapColor
    };
}

Tensor DSL::execute(const ProgramPtr& program, const Tensor& input) const {
    if (!program) {
        throw std::invalid_argument("DSL::execute: null program");
    }

    switch (program->kind) {
        case OpKind::Identity:
            return exec_identity(input);
        case OpKind::Constant:
            return exec_constant(program->param1, input);
        case OpKind::Rotate90:
            return exec_rotate90(input);
        case OpKind::FlipH:
            return exec_flip_h(input);
        case OpKind::FlipV:
            return exec_flip_v(input);
        case OpKind::Translate:
            return exec_translate(program->param1, program->param2, input);
        case OpKind::Fill:
            return exec_fill(program->param1, program->param2, input);
        case OpKind::MapColor:
            return exec_map_color(program->param1, program->param2, input);
        case OpKind::Compose: {
            if (program->children.size() != 2) {
                throw std::invalid_argument("Compose requires exactly 2 children");
            }
            Tensor intermediate = execute(program->children[0], input);
            return execute(program->children[1], intermediate);
        }
        case OpKind::Conditional: {
            // Simple conditional: if any cell matches param1, apply child[0], else child[1]
            if (program->children.size() != 2) {
                throw std::invalid_argument("Conditional requires exactly 2 children");
            }
            bool condition = false;
            for (Dim i = 0; i < input.flat_size(); ++i) {
                if (static_cast<int>(std::round(input.at(i))) == program->param1) {
                    condition = true;
                    break;
                }
            }
            return condition ? execute(program->children[0], input)
                             : execute(program->children[1], input);
        }
    }
    throw std::logic_error("DSL::execute: unhandled OpKind");
}

Tensor DSL::exec_identity(const Tensor& input) const {
    return input;
}

Tensor DSL::exec_constant(int value, const Tensor& input) const {
    return Tensor(input.shape(), static_cast<Real>(value));
}

Tensor DSL::exec_rotate90(const Tensor& input) const {
    // Works on 1D (treated as square-ish) or 2D tensors
    // For 1D: simply reverse
    if (input.shape().size() == 1) {
        std::vector<Real> rotated(input.flat_size());
        for (Dim i = 0; i < input.flat_size(); ++i) {
            rotated[i] = input.at(input.flat_size() - 1 - i);
        }
        return Tensor(input.shape(), std::move(rotated));
    }
    // For 2D: transpose + reverse rows
    if (input.shape().size() == 2) {
        Dim rows = input.shape()[0];
        Dim cols = input.shape()[1];
        std::vector<Real> rotated(rows * cols);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                rotated[c * rows + (rows - 1 - r)] = input.at(r * cols + c);
            }
        }
        return Tensor({cols, rows}, std::move(rotated));
    }
    return input;
}

Tensor DSL::exec_flip_h(const Tensor& input) const {
    if (input.shape().size() == 1) {
        std::vector<Real> flipped(input.flat_size());
        for (Dim i = 0; i < input.flat_size(); ++i) {
            flipped[i] = input.at(input.flat_size() - 1 - i);
        }
        return Tensor(input.shape(), std::move(flipped));
    }
    if (input.shape().size() == 2) {
        Dim rows = input.shape()[0];
        Dim cols = input.shape()[1];
        std::vector<Real> flipped(rows * cols);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                flipped[r * cols + (cols - 1 - c)] = input.at(r * cols + c);
            }
        }
        return Tensor(input.shape(), std::move(flipped));
    }
    return input;
}

Tensor DSL::exec_flip_v(const Tensor& input) const {
    if (input.shape().size() == 1) {
        return input; // no vertical flip for 1D
    }
    if (input.shape().size() == 2) {
        Dim rows = input.shape()[0];
        Dim cols = input.shape()[1];
        std::vector<Real> flipped(rows * cols);
        for (Dim r = 0; r < rows; ++r) {
            for (Dim c = 0; c < cols; ++c) {
                flipped[(rows - 1 - r) * cols + c] = input.at(r * cols + c);
            }
        }
        return Tensor(input.shape(), std::move(flipped));
    }
    return input;
}

Tensor DSL::exec_translate(int dx, int dy, const Tensor& input) const {
    if (input.shape().size() != 2) return input;
    Dim rows = input.shape()[0];
    Dim cols = input.shape()[1];
    std::vector<Real> result(rows * cols, 0.0);
    for (Dim r = 0; r < rows; ++r) {
        for (Dim c = 0; c < cols; ++c) {
            auto nr = static_cast<int>(r) + dy;
            auto nc = static_cast<int>(c) + dx;
            if (nr >= 0 && nr < static_cast<int>(rows) &&
                nc >= 0 && nc < static_cast<int>(cols)) {
                result[static_cast<Dim>(nr) * cols + static_cast<Dim>(nc)] =
                    input.at(r * cols + c);
            }
        }
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_fill(int from_val, int to_val, const Tensor& input) const {
    std::vector<Real> filled(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        int cell = static_cast<int>(std::round(input.at(i)));
        filled[i] = (cell == from_val) ? static_cast<Real>(to_val) : input.at(i);
    }
    return Tensor(input.shape(), std::move(filled));
}

Tensor DSL::exec_map_color(int from_c, int to_c, const Tensor& input) const {
    return exec_fill(from_c, to_c, input);
}

} // namespace uik::symbolic_descent
