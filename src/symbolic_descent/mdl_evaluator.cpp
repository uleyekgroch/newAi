#include "symbolic_descent/mdl_evaluator.hpp"
#include <cmath>
#include <limits>

namespace uik::symbolic_descent {

MdlEvaluator::MdlEvaluator(Real lambda) : lambda_(lambda) {}

Real MdlEvaluator::score(const ProgramPtr& program, const Dataset& data) const {
    if (!program) return -std::numeric_limits<Real>::infinity();
    Real dl = description_length(program);
    Real loss = prediction_loss(program, data);
    return -(dl + lambda_ * loss);
}

Real MdlEvaluator::description_length(const ProgramPtr& program) const {
    if (!program) return std::numeric_limits<Real>::infinity();
    return static_cast<Real>(program->description_length());
}

Real MdlEvaluator::prediction_loss(const ProgramPtr& program,
                                     const Dataset& data) const {
    if (!program || data.empty()) return std::numeric_limits<Real>::infinity();

    Real total_loss = 0.0;
    for (const auto& [input, expected] : data) {
        try {
            Tensor output = dsl_.execute(program, input);
            total_loss += tensor_distance(output, expected);
        } catch (...) {
            total_loss += 1000.0; // penalty for runtime errors
        }
    }
    return total_loss / static_cast<Real>(data.size());
}

bool MdlEvaluator::is_perfect_fit(const ProgramPtr& program,
                                    const Dataset& data) const {
    if (!program || data.empty()) return false;
    constexpr Real epsilon = 1e-6;
    for (const auto& [input, expected] : data) {
        try {
            Tensor output = dsl_.execute(program, input);
            if (tensor_distance(output, expected) > epsilon) return false;
        } catch (...) {
            return false;
        }
    }
    return true;
}

Real MdlEvaluator::tensor_distance(const Tensor& a, const Tensor& b) const {
    if (a.flat_size() != b.flat_size()) {
        return 1000.0; // shape mismatch penalty
    }
    Real sum = 0.0;
    for (Dim i = 0; i < a.flat_size(); ++i) {
        Real diff = a.at(i) - b.at(i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

} // namespace uik::symbolic_descent
