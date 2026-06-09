#include "symbolic_descent/rule_library.hpp"
#include <algorithm>

namespace uik::symbolic_descent {

RuleLibrary::RuleLibrary(std::size_t max_rules) : max_rules_(max_rules) {}

void RuleLibrary::add_rule(std::string context, ProgramPtr program, Real fitness) {
    rules_.push_back({std::move(context), std::move(program), fitness, 0});
    if (rules_.size() > max_rules_) {
        prune(max_rules_);
    }
}

void RuleLibrary::update_fitness(std::size_t index, Real new_fitness) {
    if (index < rules_.size()) {
        rules_[index].fitness = new_fitness;
    }
}

std::optional<RuleLibrary::Rule> RuleLibrary::find_best(
        const std::string& context) const {
    std::optional<Rule> best;
    Real best_fitness = -std::numeric_limits<Real>::infinity();

    for (const auto& rule : rules_) {
        bool matches = context.empty() || rule.context.empty() ||
                       context.find(rule.context) != std::string::npos ||
                       rule.context.find(context) != std::string::npos;
        if (matches && rule.fitness > best_fitness) {
            best_fitness = rule.fitness;
            best = rule;
        }
    }
    return best;
}

std::vector<RuleLibrary::Rule> RuleLibrary::all_sorted() const {
    auto sorted = rules_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Rule& a, const Rule& b) {
                  return a.fitness > b.fitness;
              });
    return sorted;
}

void RuleLibrary::mark_used(std::size_t index) {
    if (index < rules_.size()) {
        ++rules_[index].use_count;
    }
}

void RuleLibrary::prune(std::size_t keep_count) {
    if (rules_.size() <= keep_count) return;
    std::sort(rules_.begin(), rules_.end(),
              [](const Rule& a, const Rule& b) {
                  return (a.fitness + static_cast<Real>(a.use_count) * 0.1) >
                         (b.fitness + static_cast<Real>(b.use_count) * 0.1);
              });
    rules_.resize(keep_count);
}

std::vector<ProgramPtr> RuleLibrary::extract_programs() const {
    std::vector<ProgramPtr> programs;
    programs.reserve(rules_.size());
    for (const auto& rule : rules_) {
        programs.push_back(rule.program);
    }
    return programs;
}

} // namespace uik::symbolic_descent
