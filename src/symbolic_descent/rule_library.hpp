#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <vector>
#include <optional>
#include <string>

namespace uik::symbolic_descent {

// Stores discovered program rules (induced from observations).
// Each rule maps a pattern (input context) to a transformation (program).
class RuleLibrary {
public:
    struct Rule {
        std::string context;       // description of when rule applies
        ProgramPtr  program;
        Real        fitness = 0.0; // how well this rule explains data
        std::size_t use_count = 0;
    };

    explicit RuleLibrary(std::size_t max_rules = 100);

    void add_rule(std::string context, ProgramPtr program, Real fitness);
    void update_fitness(std::size_t index, Real new_fitness);

    // Find best rule for a given context (substring match)
    [[nodiscard]] std::optional<Rule> find_best(const std::string& context) const;

    // Get all rules sorted by fitness
    [[nodiscard]] std::vector<Rule> all_sorted() const;

    // Mark a rule as used
    void mark_used(std::size_t index);

    // Prune low-fitness, unused rules
    void prune(std::size_t keep_count);

    [[nodiscard]] std::size_t size() const { return rules_.size(); }
    [[nodiscard]] bool empty() const { return rules_.empty(); }
    [[nodiscard]] const std::vector<Rule>& rules() const { return rules_; }

    // Extract programs for seeding evolution
    [[nodiscard]] std::vector<ProgramPtr> extract_programs() const;

private:
    std::size_t max_rules_;
    std::vector<Rule> rules_;
};

} // namespace uik::symbolic_descent
