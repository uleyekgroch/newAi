#include "meta_evolution/interpreter.hpp"
#include <stdexcept>

namespace uik::meta_evolution {

Interpreter::Interpreter()
    : Interpreter(Config{})
{}

Interpreter::Interpreter(Config config)
    : config_(config), safety_(config.safety_config)
{}

Interpreter::ExecutionResult Interpreter::execute(ProgramPtr program,
                                                    const Tensor& input) {
    ++total_executions_;

    // Validate program before execution
    auto validation = safety_.validate_program(program);
    if (!validation.safe) {
        return {input, 0, 0, false, validation.reason};
    }

    std::size_t steps = 0;
    std::size_t rewrites = 0;

    try {
        Tensor output = execute_internal(program, input, steps, rewrites, 0);
        total_rewrites_ += rewrites;
        return {output, steps, rewrites, true, ""};
    } catch (const std::exception& e) {
        return {input, steps, rewrites, false, e.what()};
    }
}

Interpreter::ExecutionResult Interpreter::execute_safe(
    const ProgramPtr& program, const Tensor& input)
{
    ++total_executions_;

    auto validation = safety_.validate_program(program);
    if (!validation.safe) {
        return {input, 0, 0, false, validation.reason};
    }

    try {
        Tensor output = dsl_.execute(program, input);
        return {output, 1, 0, true, ""};
    } catch (const std::exception& e) {
        return {input, 1, 0, false, e.what()};
    }
}

ProgramPtr Interpreter::rewrite(const ProgramPtr& program,
                                  const std::vector<int>& path,
                                  const ProgramPtr& replacement) {
    // Validate replacement
    auto validation = safety_.validate_program(replacement);
    if (!validation.safe) {
        return program; // reject unsafe rewrite
    }

    if (path.empty()) {
        return replacement;
    }

    auto result = std::make_shared<ProgramNode>(*program);
    ProgramPtr current = result;

    // Navigate to parent of target node
    for (std::size_t i = 0; i < path.size() - 1; ++i) {
        int idx = path[i];
        if (idx < 0 || static_cast<std::size_t>(idx) >= current->children.size()) {
            return program; // invalid path
        }
        // Deep copy along the path
        current->children[static_cast<std::size_t>(idx)] =
            std::make_shared<ProgramNode>(*current->children[static_cast<std::size_t>(idx)]);
        current = current->children[static_cast<std::size_t>(idx)];
    }

    // Replace target node
    int last_idx = path.back();
    if (last_idx < 0 ||
        static_cast<std::size_t>(last_idx) >= current->children.size()) {
        return program; // invalid path
    }
    current->children[static_cast<std::size_t>(last_idx)] = replacement;

    // Validate resulting program
    validation = safety_.validate_program(result);
    if (!validation.safe) {
        return program; // reject if result is unsafe
    }

    ++total_rewrites_;
    return result;
}

Tensor Interpreter::execute_internal(
    const ProgramPtr& program, const Tensor& input,
    std::size_t& steps, std::size_t& rewrites,
    std::size_t rewrite_depth)
{
    ++steps;
    if (steps > config_.max_execution_steps) {
        throw std::runtime_error("execution step limit exceeded");
    }

    // For Repeat (loop), handle internally with step counting
    if (program->kind == OpKind::Repeat && !program->children.empty()) {
        int iterations = std::clamp(program->param1, 0, 100);
        Tensor current = input;
        for (int i = 0; i < iterations; ++i) {
            current = execute_internal(program->children[0], current,
                                        steps, rewrites, rewrite_depth);
        }
        return current;
    }

    // For Compose, execute sequentially with step counting
    if (program->kind == OpKind::Compose && program->children.size() == 2) {
        Tensor mid = execute_internal(program->children[0], input,
                                       steps, rewrites, rewrite_depth);
        return execute_internal(program->children[1], mid,
                                 steps, rewrites, rewrite_depth);
    }

    // For other ops, delegate to DSL
    return dsl_.execute(program, input);
}

} // namespace uik::meta_evolution
