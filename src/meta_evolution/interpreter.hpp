#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include "symbolic_descent/dsl.hpp"
#include "meta_evolution/safety_guard.hpp"
#include <unordered_map>
#include <vector>
#include <functional>

namespace uik::meta_evolution {

// Runtime interpreter for DSL programs with self-modification capability.
// Programs can inspect and rewrite themselves during execution.
// Bounded by SafetyGuard to prevent runaway modifications.
class Interpreter {
public:
    struct Config {
        std::size_t max_execution_steps = 10000;
        std::size_t max_rewrite_depth   = 5;
        SafetyGuard::Config safety_config{};
    };

    struct ExecutionResult {
        Tensor output;
        std::size_t steps_used  = 0;
        std::size_t rewrites    = 0;
        bool halted_safely      = true;
        std::string halt_reason;
    };

    Interpreter();
    explicit Interpreter(Config config);

    // Execute a program with self-modification capability
    [[nodiscard]] ExecutionResult execute(ProgramPtr program,
                                           const Tensor& input);

    // Execute without self-modification (safe mode)
    [[nodiscard]] ExecutionResult execute_safe(const ProgramPtr& program,
                                                const Tensor& input);

    // Rewrite a program node at a given path
    // path = indices into children: [0, 1, 0] means root.children[0].children[1].children[0]
    [[nodiscard]] ProgramPtr rewrite(const ProgramPtr& program,
                                      const std::vector<int>& path,
                                      const ProgramPtr& replacement);

    // Get execution statistics
    [[nodiscard]] std::size_t total_executions() const { return total_executions_; }
    [[nodiscard]] std::size_t total_rewrites() const { return total_rewrites_; }
    [[nodiscard]] const SafetyGuard& safety() const { return safety_; }

private:
    Config config_;
    symbolic_descent::DSL dsl_;
    SafetyGuard safety_;
    std::size_t total_executions_ = 0;
    std::size_t total_rewrites_ = 0;

    // Internal recursive execution with step counter
    Tensor execute_internal(const ProgramPtr& program, const Tensor& input,
                            std::size_t& steps, std::size_t& rewrites,
                            std::size_t rewrite_depth);
};

} // namespace uik::meta_evolution
