#pragma once

#include "common/interfaces.hpp"
#include "symbolic_descent/program_space.hpp"

namespace uik::meta_evolution {

// Mutates and recombines programs in the archive.
// Implements the "self-modification" aspect of the Darwin Gödel Machine.
class Mutator final : public IMutator {
public:
    explicit Mutator(unsigned seed = 42);

    ProgramPtr mutate(const ProgramPtr& program) override;
    ProgramPtr crossover(const ProgramPtr& a, const ProgramPtr& b) override;

private:
    symbolic_descent::ProgramSpace space_;
};

} // namespace uik::meta_evolution
