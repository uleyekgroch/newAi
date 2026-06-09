#include "symbolic_descent/program_space.hpp"
#include <algorithm>

namespace uik::symbolic_descent {

ProgramSpace::ProgramSpace(int max_depth, int color_range,
                            int grid_range, unsigned seed)
    : max_depth_(max_depth), color_range_(color_range)
    , grid_range_(grid_range), rng_(seed)
{}

ProgramPtr ProgramSpace::random_program(int depth) {
    if (depth >= max_depth_) {
        return random_leaf();
    }

    std::uniform_int_distribution<int> choice(0, 9);
    int c = choice(rng_);

    if (c < 6) {
        return random_primitive();
    }
    // Compose two sub-programs
    auto left = random_program(depth + 1);
    auto right = random_program(depth + 1);
    return compose(left, right);
}

ProgramPtr ProgramSpace::mutate(const ProgramPtr& program) {
    if (!program) return random_program();
    return replace_random_subtree(program);
}

ProgramPtr ProgramSpace::crossover(const ProgramPtr& a, const ProgramPtr& b) {
    if (!a) return b;
    if (!b) return a;
    // Take root structure from a, replace a random subtree with one from b
    auto donor = pick_random_node(b);
    auto result = std::make_shared<ProgramNode>(*a);

    if (!result->children.empty()) {
        std::uniform_int_distribution<std::size_t> idx(0, result->children.size() - 1);
        result->children[idx(rng_)] = donor;
    } else {
        // a is a leaf: compose a and donor
        return compose(std::make_shared<ProgramNode>(*a), donor);
    }
    return result;
}

std::vector<ProgramPtr> ProgramSpace::neighborhood(const ProgramPtr& program,
                                                     std::size_t count) {
    std::vector<ProgramPtr> neighbors;
    neighbors.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uniform_int_distribution<int> op(0, 2);
        switch (op(rng_)) {
            case 0:
                neighbors.push_back(mutate(program));
                break;
            case 1: {
                auto partner = random_program();
                neighbors.push_back(crossover(program, partner));
                break;
            }
            default:
                neighbors.push_back(random_program());
                break;
        }
    }
    return neighbors;
}

ProgramPtr ProgramSpace::random_leaf() {
    return random_primitive();
}

ProgramPtr ProgramSpace::random_primitive() {
    auto prims = std::vector<OpKind>{
        OpKind::Identity, OpKind::Rotate90, OpKind::FlipH, OpKind::FlipV,
        OpKind::Translate, OpKind::Fill, OpKind::MapColor
    };
    std::uniform_int_distribution<std::size_t> idx(0, prims.size() - 1);
    OpKind kind = prims[idx(rng_)];

    int p1 = 0, p2 = 0;
    if (kind == OpKind::Translate) {
        std::uniform_int_distribution<int> shift(-grid_range_, grid_range_);
        p1 = shift(rng_);
        p2 = shift(rng_);
    } else if (kind == OpKind::Fill || kind == OpKind::MapColor) {
        std::uniform_int_distribution<int> color(0, color_range_ - 1);
        p1 = color(rng_);
        p2 = color(rng_);
    }
    return make_program(kind, p1, p2);
}

ProgramPtr ProgramSpace::replace_random_subtree(const ProgramPtr& program) {
    auto result = std::make_shared<ProgramNode>(*program);
    if (result->children.empty()) {
        // Leaf: replace entire node with new primitive
        auto replacement = random_primitive();
        return replacement;
    }
    std::uniform_int_distribution<std::size_t> idx(0, result->children.size() - 1);
    std::size_t target = idx(rng_);

    std::uniform_int_distribution<int> recurse(0, 1);
    if (recurse(rng_) == 0 && result->children[target]->children.size() > 0) {
        result->children[target] = replace_random_subtree(result->children[target]);
    } else {
        result->children[target] = random_primitive();
    }
    return result;
}

ProgramPtr ProgramSpace::pick_random_node(const ProgramPtr& program) {
    if (!program || program->children.empty()) return program;
    std::uniform_int_distribution<int> choice(0, 1);
    if (choice(rng_) == 0) return program;
    std::uniform_int_distribution<std::size_t> idx(0, program->children.size() - 1);
    return pick_random_node(program->children[idx(rng_)]);
}

} // namespace uik::symbolic_descent
