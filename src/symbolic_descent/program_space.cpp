#include "symbolic_descent/program_space.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

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
        std::uniform_int_distribution<int> op(0, 4);
        switch (op(rng_)) {
            case 0:
                neighbors.push_back(mutate(program));
                break;
            case 1: {
                auto partner = random_program();
                neighbors.push_back(crossover(program, partner));
                break;
            }
            case 2:
                neighbors.push_back(type_aware_mutate(program));
                break;
            case 3:
                neighbors.push_back(param_perturbation(program));
                break;
            default:
                neighbors.push_back(random_program());
                break;
        }
    }
    return neighbors;
}

// ── Guided search operators ──

ProgramPtr ProgramSpace::type_aware_mutate(const ProgramPtr& program) {
    if (!program) return random_program();

    auto result = std::make_shared<ProgramNode>(*program);
    auto compatible = compatible_ops(result->kind);
    if (compatible.empty()) return mutate(program);

    std::uniform_int_distribution<std::size_t> idx(0, compatible.size() - 1);
    result->kind = compatible[idx(rng_)];

    // Adjust params for new kind
    if (result->kind == OpKind::Translate) {
        std::uniform_int_distribution<int> shift(-grid_range_, grid_range_);
        result->param1 = shift(rng_);
        result->param2 = shift(rng_);
    } else if (result->kind == OpKind::Fill || result->kind == OpKind::MapColor) {
        std::uniform_int_distribution<int> color(0, color_range_ - 1);
        result->param1 = color(rng_);
        result->param2 = color(rng_);
    }

    // Also mutate a random child if children exist
    if (!result->children.empty()) {
        std::uniform_int_distribution<int> recurse(0, 2);
        if (recurse(rng_) == 0) {
            std::uniform_int_distribution<std::size_t> ci(0, result->children.size() - 1);
            result->children[ci(rng_)] = type_aware_mutate(result->children[ci(rng_)]);
        }
    }
    return result;
}

ProgramPtr ProgramSpace::analogy_crossover(const ProgramPtr& a,
                                             const ProgramPtr& b) {
    if (!a || !b) return a ? a : b;

    // Find the most structurally similar subtrees and swap
    auto result = std::make_shared<ProgramNode>(*a);

    if (!result->children.empty() && !b->children.empty()) {
        // Find best matching child pair by structural similarity
        Real best_sim = -1.0;
        std::size_t best_a_idx = 0;
        std::size_t best_b_idx = 0;
        for (std::size_t i = 0; i < result->children.size(); ++i) {
            for (std::size_t j = 0; j < b->children.size(); ++j) {
                Real sim = structural_similarity(result->children[i], b->children[j]);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_a_idx = i;
                    best_b_idx = j;
                }
            }
        }
        // Replace the most similar subtree in a with one from b
        result->children[best_a_idx] = std::make_shared<ProgramNode>(*b->children[best_b_idx]);
    } else {
        return crossover(a, b);
    }
    return result;
}

std::vector<ProgramPtr> ProgramSpace::guided_neighborhood(
    const ProgramPtr& program,
    const std::vector<Real>& op_weights,
    std::size_t count)
{
    // op_weights: [mutate, crossover, type_aware, param_perturb, random]
    // Normalize to probabilities
    std::vector<Real> weights = op_weights;
    if (weights.size() < 5) weights.resize(5, 1.0);
    Real sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sum <= 0.0) sum = 5.0;

    std::vector<Real> cumulative(5);
    cumulative[0] = weights[0] / sum;
    for (std::size_t i = 1; i < 5; ++i) {
        cumulative[i] = cumulative[i-1] + weights[i] / sum;
    }

    std::vector<ProgramPtr> neighbors;
    neighbors.reserve(count);
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    for (std::size_t i = 0; i < count; ++i) {
        Real r = dist(rng_);
        if (r < cumulative[0]) {
            neighbors.push_back(mutate(program));
        } else if (r < cumulative[1]) {
            auto partner = random_program();
            neighbors.push_back(analogy_crossover(program, partner));
        } else if (r < cumulative[2]) {
            neighbors.push_back(type_aware_mutate(program));
        } else if (r < cumulative[3]) {
            neighbors.push_back(param_perturbation(program));
        } else {
            neighbors.push_back(simplify(program));
        }
    }
    return neighbors;
}

ProgramPtr ProgramSpace::param_perturbation(const ProgramPtr& program) {
    if (!program) return random_program();

    auto result = std::make_shared<ProgramNode>(*program);
    // Only change parameters, keep structure
    std::uniform_int_distribution<int> delta(-2, 2);
    if (result->kind == OpKind::Translate) {
        result->param1 += delta(rng_);
        result->param2 += delta(rng_);
        result->param1 = std::clamp(result->param1, -grid_range_, grid_range_);
        result->param2 = std::clamp(result->param2, -grid_range_, grid_range_);
    } else if (result->kind == OpKind::Fill || result->kind == OpKind::MapColor
               || result->kind == OpKind::Constant) {
        result->param1 = std::clamp(result->param1 + delta(rng_), 0, color_range_ - 1);
        result->param2 = std::clamp(result->param2 + delta(rng_), 0, color_range_ - 1);
    }

    // Recurse into children
    for (auto& child : result->children) {
        std::uniform_int_distribution<int> recurse(0, 2);
        if (recurse(rng_) == 0) {
            child = param_perturbation(child);
        }
    }
    return result;
}

ProgramPtr ProgramSpace::simplify(const ProgramPtr& program) {
    if (!program) return program;

    auto result = std::make_shared<ProgramNode>(*program);

    // Rule: Compose(Id, X) → X, Compose(X, Id) → X
    if (result->kind == OpKind::Compose && result->children.size() == 2) {
        if (result->children[0] && result->children[0]->kind == OpKind::Identity) {
            return std::make_shared<ProgramNode>(*result->children[1]);
        }
        if (result->children[1] && result->children[1]->kind == OpKind::Identity) {
            return std::make_shared<ProgramNode>(*result->children[0]);
        }
    }

    // Rule: FlipH(FlipH(X)) → X
    if (result->kind == OpKind::FlipH && result->children.size() == 1 &&
        result->children[0] && result->children[0]->kind == OpKind::FlipH) {
        if (!result->children[0]->children.empty()) {
            return std::make_shared<ProgramNode>(*result->children[0]->children[0]);
        }
        return identity();
    }

    // Rule: FlipV(FlipV(X)) → X
    if (result->kind == OpKind::FlipV && result->children.size() == 1 &&
        result->children[0] && result->children[0]->kind == OpKind::FlipV) {
        if (!result->children[0]->children.empty()) {
            return std::make_shared<ProgramNode>(*result->children[0]->children[0]);
        }
        return identity();
    }

    // Recursively simplify children
    for (auto& child : result->children) {
        child = simplify(child);
    }
    return result;
}

// ── Private helpers ──

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

std::vector<OpKind> ProgramSpace::compatible_ops(OpKind kind) {
    // Group ops by type: geometric transforms, color transforms, structural
    switch (kind) {
        case OpKind::Rotate90:
        case OpKind::FlipH:
        case OpKind::FlipV:
            return {OpKind::Rotate90, OpKind::FlipH, OpKind::FlipV};
        case OpKind::Translate:
            return {OpKind::Translate, OpKind::Rotate90, OpKind::FlipH, OpKind::FlipV};
        case OpKind::Fill:
        case OpKind::MapColor:
        case OpKind::Constant:
            return {OpKind::Fill, OpKind::MapColor, OpKind::Constant};
        case OpKind::Compose:
            return {OpKind::Compose, OpKind::Conditional};
        case OpKind::Conditional:
            return {OpKind::Conditional, OpKind::Compose};
        case OpKind::Identity:
            return {OpKind::Identity, OpKind::Rotate90, OpKind::FlipH, OpKind::FlipV};
        case OpKind::Add:
        case OpKind::Multiply:
        case OpKind::Modulo:
            return {OpKind::Add, OpKind::Multiply, OpKind::Modulo};
        case OpKind::Threshold:
        case OpKind::Filter:
            return {OpKind::Threshold, OpKind::Filter};
        case OpKind::Count:
            return {OpKind::Count, OpKind::Filter};
        case OpKind::Repeat:
        case OpKind::Fold:
            return {OpKind::Repeat, OpKind::Fold};
        case OpKind::Zip:
            return {OpKind::Zip, OpKind::Compose};
        case OpKind::Store:
        case OpKind::Load:
            return {OpKind::Store, OpKind::Load};
    }
    return {};
}

Real ProgramSpace::structural_similarity(const ProgramPtr& a, const ProgramPtr& b) {
    if (!a && !b) return 1.0;
    if (!a || !b) return 0.0;

    Real score = (a->kind == b->kind) ? 1.0 : 0.0;
    std::size_t max_children = std::max(a->children.size(), b->children.size());
    if (max_children == 0) return score;

    Real child_score = 0.0;
    std::size_t min_children = std::min(a->children.size(), b->children.size());
    for (std::size_t i = 0; i < min_children; ++i) {
        child_score += structural_similarity(a->children[i], b->children[i]);
    }
    return (score + child_score) / (1.0 + static_cast<Real>(max_children));
}

// ── Template-guided edit ──

void ProgramSpace::extract_subtrees(const ProgramPtr& prog,
                                     std::vector<ProgramPtr>& out, int depth) {
    if (!prog || depth > 3) return;
    if (prog->kind != OpKind::Identity) {
        out.push_back(prog);
    }
    for (const auto& child : prog->children) {
        extract_subtrees(child, out, depth + 1);
    }
}

void ProgramSpace::add_template(const ProgramPtr& successful_program) {
    if (!successful_program) return;

    std::vector<ProgramPtr> subtrees;
    extract_subtrees(successful_program, subtrees);

    for (const auto& sub : subtrees) {
        if (templates_.size() >= MAX_TEMPLATES) {
            // Replace lowest-fitness template
            auto worst = std::min_element(templates_.begin(), templates_.end(),
                [](const Template& a, const Template& b) { return a.fitness < b.fitness; });
            if (worst != templates_.end()) {
                worst->pattern = std::make_shared<ProgramNode>(*sub);
                worst->fitness = static_cast<Real>(sub->description_length());
            }
        } else {
            templates_.push_back({
                std::make_shared<ProgramNode>(*sub),
                static_cast<Real>(sub->description_length())
            });
        }
    }
}

ProgramPtr ProgramSpace::template_guided_edit(const ProgramPtr& program) {
    if (!program) return random_program();
    if (templates_.empty()) return type_aware_mutate(program);

    // Select a template weighted by fitness
    std::vector<Real> weights(templates_.size());
    Real max_f = 0.0;
    for (const auto& t : templates_) max_f = std::max(max_f, t.fitness);
    for (std::size_t i = 0; i < templates_.size(); ++i) {
        weights[i] = std::max(0.1, templates_[i].fitness - max_f + 10.0);
    }
    Real total = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::uniform_real_distribution<Real> dist(0.0, total);
    Real r = dist(rng_);
    Real cum = 0.0;
    std::size_t chosen = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        cum += weights[i];
        if (r <= cum) { chosen = i; break; }
    }

    auto& templ = templates_[chosen].pattern;

    // Strategy: graft the template subtree into the program
    auto result = std::make_shared<ProgramNode>(*program);
    if (!result->children.empty()) {
        // Replace a random child with a composition of child + template
        std::uniform_int_distribution<std::size_t> idx(0, result->children.size() - 1);
        auto target = idx(rng_);
        std::uniform_int_distribution<int> strategy(0, 2);
        switch (strategy(rng_)) {
            case 0:
                // Replace child with template
                result->children[target] = std::make_shared<ProgramNode>(*templ);
                break;
            case 1:
                // Compose child with template
                result->children[target] = compose(result->children[target],
                                                    std::make_shared<ProgramNode>(*templ));
                break;
            default:
                // Analogy: match template's op kind to child
                result->children[target]->kind = templ->kind;
                result->children[target]->param1 = templ->param1;
                result->children[target]->param2 = templ->param2;
                break;
        }
    } else {
        // Leaf: compose with template
        return compose(result, std::make_shared<ProgramNode>(*templ));
    }
    return result;
}

} // namespace uik::symbolic_descent
