#pragma once

#include "common/types.hpp"
#include "common/program.hpp"
#include <optional>
#include <vector>
#include <concepts>

namespace uik {

// ── Concepts (C++20) ──

template<typename T>
concept Encodable = requires(T encoder, const Observation& obs) {
    { encoder.encode(obs) } -> std::same_as<State>;
};

template<typename T>
concept Predictable = requires(T predictor, const State& s, const Action& a) {
    { predictor.predict(s, a) } -> std::same_as<State>;
};

// ── Abstract Interfaces (Dependency Inversion) ──

class IWorldModel {
public:
    virtual ~IWorldModel() = default;
    virtual State encode(const Observation& obs) = 0;
    virtual State predict_next(const State& current, const Action& action) = 0;
    virtual Real  compute_novelty(const State& state) = 0;
    virtual void  update(const Observation& obs) = 0;
    virtual Real  compression_progress() const = 0;
};

class ISearchEngine {
public:
    virtual ~ISearchEngine() = default;
    using DataPair = std::pair<Tensor, Tensor>; // input -> output
    using Dataset  = std::vector<DataPair>;
    virtual std::optional<ProgramPtr> search(const Dataset& data,
                                              std::size_t max_iterations) = 0;
};

class IMutator {
public:
    virtual ~IMutator() = default;
    virtual ProgramPtr mutate(const ProgramPtr& program) = 0;
    virtual ProgramPtr crossover(const ProgramPtr& a, const ProgramPtr& b) = 0;
};

class IGoalSetter {
public:
    virtual ~IGoalSetter() = default;
    virtual State set_goal(const State& current,
                           Real compression_progress,
                           Real external_reward) = 0;
};

class IPlanner {
public:
    virtual ~IPlanner() = default;
    virtual std::vector<Action> plan(const State& current,
                                      const State& goal,
                                      IWorldModel& world_model,
                                      int horizon) = 0;
};

// ── Environment interface ──

class IEnvironment {
public:
    virtual ~IEnvironment() = default;
    virtual Observation reset() = 0;
    virtual StepResult  step(const Action& action) = 0;
    virtual int         action_space_size() const = 0;
};

} // namespace uik
