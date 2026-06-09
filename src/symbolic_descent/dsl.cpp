#include "symbolic_descent/dsl.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace uik::symbolic_descent {

std::vector<OpKind> DSL::primitives() {
    return {
        OpKind::Identity, OpKind::Constant, OpKind::Rotate90,
        OpKind::FlipH, OpKind::FlipV, OpKind::Translate,
        OpKind::Fill, OpKind::MapColor
    };
}

std::vector<OpKind> DSL::extended_primitives() {
    return {
        OpKind::Identity, OpKind::Constant, OpKind::Rotate90,
        OpKind::FlipH, OpKind::FlipV, OpKind::Translate,
        OpKind::Fill, OpKind::MapColor,
        OpKind::Add, OpKind::Multiply, OpKind::Modulo,
        OpKind::Threshold, OpKind::Count, OpKind::Filter,
        OpKind::Store, OpKind::Load
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

        // Phase 3 general computation ops
        case OpKind::Add:
            return exec_add(program->param1, input);
        case OpKind::Multiply:
            return exec_multiply(program->param1, input);
        case OpKind::Modulo:
            return exec_modulo(program->param1, input);
        case OpKind::Threshold:
            return exec_threshold(program->param1, program->param2, input);
        case OpKind::Count:
            return exec_count(program->param1, input);
        case OpKind::Repeat: {
            if (program->children.empty()) return input;
            return exec_repeat(program->children[0], program->param1, input);
        }
        case OpKind::Fold: {
            if (program->children.empty()) return input;
            return exec_fold(program->children[0], input);
        }
        case OpKind::Zip: {
            if (program->children.size() < 2) return input;
            return exec_zip(program->children[0], program->children[1], input);
        }
        case OpKind::Filter:
            return exec_filter(program->param1, input);
        case OpKind::Store:
            return exec_store(program->param1, input);
        case OpKind::Load:
            return exec_load(program->param1, input);
    }
    throw std::logic_error("DSL::execute: unhandled OpKind");
}

// ── Phase 1 grid ops ──

Tensor DSL::exec_identity(const Tensor& input) const {
    return input;
}

Tensor DSL::exec_constant(int value, const Tensor& input) const {
    return Tensor(input.shape(), static_cast<Real>(value));
}

Tensor DSL::exec_rotate90(const Tensor& input) const {
    if (input.shape().size() == 1) {
        std::vector<Real> rotated(input.flat_size());
        for (Dim i = 0; i < input.flat_size(); ++i) {
            rotated[i] = input.at(input.flat_size() - 1 - i);
        }
        return Tensor(input.shape(), std::move(rotated));
    }
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
        return input;
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
            int src_r = static_cast<int>(r) - dy;
            int src_c = static_cast<int>(c) - dx;
            if (src_r >= 0 && static_cast<Dim>(src_r) < rows &&
                src_c >= 0 && static_cast<Dim>(src_c) < cols) {
                result[r * cols + c] = input.at(
                    static_cast<Dim>(src_r) * cols + static_cast<Dim>(src_c));
            }
        }
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_fill(int from_val, int to_val, const Tensor& input) const {
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        int v = static_cast<int>(std::round(input.at(i)));
        result[i] = (v == from_val) ? static_cast<Real>(to_val) : input.at(i);
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_map_color(int from_c, int to_c, const Tensor& input) const {
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        int v = static_cast<int>(std::round(input.at(i)));
        result[i] = (v == from_c) ? static_cast<Real>(to_c) : input.at(i);
    }
    return Tensor(input.shape(), std::move(result));
}

// ── Phase 3 general computation ops ──

Tensor DSL::exec_add(int offset, const Tensor& input) const {
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        result[i] = input.at(i) + static_cast<Real>(offset);
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_multiply(int factor, const Tensor& input) const {
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        result[i] = input.at(i) * static_cast<Real>(factor);
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_modulo(int divisor, const Tensor& input) const {
    if (divisor == 0) return input;
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        int v = static_cast<int>(std::round(input.at(i)));
        result[i] = static_cast<Real>(((v % divisor) + divisor) % divisor);
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_threshold(int thresh, int val, const Tensor& input) const {
    std::vector<Real> result(input.flat_size());
    for (Dim i = 0; i < input.flat_size(); ++i) {
        result[i] = (input.at(i) > static_cast<Real>(thresh))
                    ? static_cast<Real>(val) : 0.0;
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_count(int target, const Tensor& input) const {
    Dim count = 0;
    for (Dim i = 0; i < input.flat_size(); ++i) {
        if (static_cast<int>(std::round(input.at(i))) == target) ++count;
    }
    return Tensor(input.shape(), static_cast<Real>(count));
}

Tensor DSL::exec_repeat(const ProgramPtr& child, int times,
                          const Tensor& input) const {
    int iterations = std::clamp(times, 0, MAX_LOOP_ITERATIONS);
    Tensor current = input;
    for (int i = 0; i < iterations; ++i) {
        current = execute(child, current);
    }
    return current;
}

Tensor DSL::exec_fold(const ProgramPtr& child, const Tensor& input) const {
    // Reduce: start with first element, apply child to accumulate
    if (input.flat_size() == 0) return input;

    // Fold by repeatedly applying child to single-element slices
    Tensor acc({1}, std::vector<Real>{input.at(0)});
    for (Dim i = 1; i < input.flat_size(); ++i) {
        // Create a 2-element tensor [acc, element] and apply child
        Tensor pair({2}, std::vector<Real>{acc.at(0), input.at(i)});
        Tensor result = execute(child, pair);
        acc = Tensor({1}, std::vector<Real>{result.at(0)});
    }
    // Broadcast result to input shape
    return Tensor(input.shape(), acc.at(0));
}

Tensor DSL::exec_zip(const ProgramPtr& child0, const ProgramPtr& child1,
                       const Tensor& input) const {
    Tensor a = execute(child0, input);
    Tensor b = execute(child1, input);
    // Elementwise combination: sum of two branches
    if (a.flat_size() != b.flat_size()) return a;
    std::vector<Real> result(a.flat_size());
    for (Dim i = 0; i < a.flat_size(); ++i) {
        result[i] = a.at(i) + b.at(i);
    }
    return Tensor(a.shape(), std::move(result));
}

Tensor DSL::exec_filter(int target, const Tensor& input) const {
    std::vector<Real> result(input.flat_size(), 0.0);
    for (Dim i = 0; i < input.flat_size(); ++i) {
        if (static_cast<int>(std::round(input.at(i))) == target) {
            result[i] = input.at(i);
        }
    }
    return Tensor(input.shape(), std::move(result));
}

Tensor DSL::exec_store(int slot, const Tensor& input) const {
    int safe_slot = std::clamp(slot, 0, MAX_VARIABLE_SLOTS - 1);
    variable_store_[safe_slot] = input;
    return input;
}

Tensor DSL::exec_load(int slot, const Tensor& input) const {
    int safe_slot = std::clamp(slot, 0, MAX_VARIABLE_SLOTS - 1);
    auto it = variable_store_.find(safe_slot);
    if (it != variable_store_.end()) {
        return it->second;
    }
    return input; // default: return input if slot empty
}

} // namespace uik::symbolic_descent
