#include "meta_evolution/self_modifier.hpp"
#include <algorithm>

namespace uik::meta_evolution {

SelfModifier::SelfModifier()
    : SelfModifier(Config{})
{}

SelfModifier::SelfModifier(Config config)
    : config_(config)
    , archive_(200)
    , space_(3, 10, 5, config.seed)
    , rng_(config.seed)
{
    // Initialize default strategies for all kinds
    for (auto kind : all_kinds()) {
        strategies_.push_back({kind, identity(), 0.0});
    }
}

bool SelfModifier::try_modify(
    StrategyKind kind,
    std::function<Real(const ProgramPtr&)> eval_fn)
{
    ensure_strategy_exists(kind);
    auto idx = find_strategy_idx(kind);
    auto& current = strategies_[idx];

    Real current_fitness = eval_fn(current.program);
    current.fitness = current_fitness;

    // Generate candidates via mutation/crossover of current strategy
    bool improved = false;
    for (std::size_t c = 0; c < config_.candidates_per_round; ++c) {
        ProgramPtr candidate;
        std::uniform_int_distribution<int> op(0, 2);
        switch (op(rng_)) {
            case 0:
                candidate = space_.mutate(current.program);
                break;
            case 1: {
                auto partner = space_.random_program();
                candidate = space_.crossover(current.program, partner);
                break;
            }
            default:
                candidate = space_.random_program();
                break;
        }

        Real candidate_fitness = eval_fn(candidate);

        // Archive all attempts (Darwin Gödel Machine: keep diversity)
        archive_.add(candidate, candidate_fitness,
                     static_cast<Real>(candidate->description_length()), 0);

        // Apply improvement if above threshold
        if (candidate_fitness > current_fitness + config_.improvement_threshold) {
            current.program = candidate;
            current.fitness = candidate_fitness;
            current_fitness = candidate_fitness;
            improved = true;
        }
    }

    if (improved) {
        ++modifications_;
    }
    return improved;
}

const ProgramPtr& SelfModifier::current_strategy(StrategyKind kind) const {
    auto idx = find_strategy_idx(kind);
    return strategies_[idx].program;
}

void SelfModifier::record_strategy(StrategyKind kind, ProgramPtr prog,
                                     Real fitness) {
    ensure_strategy_exists(kind);
    auto idx = find_strategy_idx(kind);
    strategies_[idx].program = std::move(prog);
    strategies_[idx].fitness = fitness;
    archive_.add(strategies_[idx].program, fitness,
                 static_cast<Real>(strategies_[idx].program->description_length()), 0);
}

std::size_t SelfModifier::find_strategy_idx(StrategyKind kind) const {
    for (std::size_t i = 0; i < strategies_.size(); ++i) {
        if (strategies_[i].kind == kind) return i;
    }
    return 0;
}

void SelfModifier::ensure_strategy_exists(StrategyKind kind) {
    for (const auto& s : strategies_) {
        if (s.kind == kind) return;
    }
    strategies_.push_back({kind, identity(), 0.0});
}

std::vector<SelfModifier::StrategyKind> SelfModifier::all_kinds() {
    return {
        StrategyKind::FitnessWeighting,
        StrategyKind::NeighborhoodBias,
        StrategyKind::ExplorationPolicy,
        StrategyKind::GoalFunction,
        StrategyKind::RewardShaping,
        StrategyKind::MutationStrategy,
        StrategyKind::SimplificationRule,
    };
}

std::size_t SelfModifier::evolve_all(
    std::function<Real(StrategyKind, const ProgramPtr&)> eval_fn)
{
    std::size_t improved = 0;
    for (auto kind : all_kinds()) {
        auto wrapper = [&](const ProgramPtr& p) { return eval_fn(kind, p); };
        if (try_modify(kind, wrapper)) {
            ++improved;
        }
    }
    return improved;
}

} // namespace uik::meta_evolution
