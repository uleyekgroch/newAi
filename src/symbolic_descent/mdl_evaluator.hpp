#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include "symbolic_descent/dsl.hpp"
#include <vector>

namespace uik::symbolic_descent {

// Evaluates programs using the Minimum Description Length principle.
// Score = -(description_length + lambda * prediction_loss)
// Lower description length + lower loss = better program.
class MdlEvaluator {
public:
    using DataPair = std::pair<Tensor, Tensor>;
    using Dataset  = std::vector<DataPair>;

    explicit MdlEvaluator(Real lambda = 1.0);

    // Total MDL score (higher is better)
    [[nodiscard]] Real score(const ProgramPtr& program,
                              const Dataset& data) const;

    // Description length (Kolmogorov complexity approximation)
    [[nodiscard]] Real description_length(const ProgramPtr& program) const;

    // Prediction loss across all data pairs
    [[nodiscard]] Real prediction_loss(const ProgramPtr& program,
                                        const Dataset& data) const;

    // Check if program perfectly fits all data
    [[nodiscard]] bool is_perfect_fit(const ProgramPtr& program,
                                       const Dataset& data) const;

private:
    Real lambda_;
    DSL dsl_;

    Real tensor_distance(const Tensor& a, const Tensor& b) const;
};

} // namespace uik::symbolic_descent
