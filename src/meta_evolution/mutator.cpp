#include "meta_evolution/mutator.hpp"

namespace uik::meta_evolution {

Mutator::Mutator(unsigned seed) : space_(4, 10, 5, seed) {}

ProgramPtr Mutator::mutate(const ProgramPtr& program) {
    return space_.mutate(program);
}

ProgramPtr Mutator::crossover(const ProgramPtr& a, const ProgramPtr& b) {
    return space_.crossover(a, b);
}

} // namespace uik::meta_evolution
