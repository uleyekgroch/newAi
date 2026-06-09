# Universal Intelligence Kernel — Implementation Plan

## Architecture: DDD Bounded Contexts

```
src/
├── common/              # Shared Kernel — value objects, interfaces
│   ├── types.hpp        # State, Action, Observation, Reward
│   ├── program.hpp      # Program value object (AST node)
│   └── interfaces.hpp   # Abstract interfaces (IWorldModel, ISearchEngine, etc.)
├── world_model/         # Bounded Context 1 — Perception Layer
│   ├── latent_state.hpp/cpp
│   ├── dynamics_predictor.hpp/cpp
│   ├── novelty_detector.hpp/cpp
│   └── world_model.hpp/cpp
├── symbolic_descent/    # Bounded Context 2 — Cognitive Layer
│   ├── dsl.hpp/cpp
│   ├── program_space.hpp/cpp
│   ├── mdl_evaluator.hpp/cpp
│   └── search_engine.hpp/cpp
├── meta_evolution/      # Bounded Context 3 — Meta Layer
│   ├── archive.hpp/cpp
│   ├── mutator.hpp/cpp
│   └── evolutionary_selector.hpp/cpp
├── agent/               # Bounded Context 4 — Orchestration
│   ├── goal_setter.hpp/cpp
│   ├── planner.hpp/cpp
│   └── agent_kernel.hpp/cpp
└── main.cpp

tests/
├── world_model/
├── symbolic_descent/
├── meta_evolution/
└── agent/
```

## Task List

### Phase 0: Project Skeleton
- [x] Create tasks/todo.md
- [x] CMakeLists.txt (root + subdirs)
- [x] GoogleTest integration via FetchContent
- [x] Verify empty build compiles

### Phase 1: Shared Kernel (common/)
- [x] types.hpp — State, Action, Observation, Reward, Tensor value objects
- [x] program.hpp — Program AST node (shared_ptr-based tree)
- [x] interfaces.hpp — IWorldModel, ISearchEngine, IMutator, IGoalSetter, IPlanner, IEnvironment

### Phase 2: World Model (TDD) — 16 tests
- [x] TEST: LatentState encode/decode round-trip (5 tests)
- [x] LatentState implementation
- [x] TEST: DynamicsPredictor next-state prediction (4 tests)
- [x] DynamicsPredictor implementation
- [x] TEST: NoveltyDetector scoring (4 tests)
- [x] NoveltyDetector implementation
- [x] TEST: WorldModel compression progress (3 tests)
- [x] WorldModel integration

### Phase 3: Symbolic Descent (TDD) — 16 tests
- [x] TEST: DSL primitive operations (9 tests)
- [x] DSL implementation (grid transform primitives)
- [x] TEST: ProgramSpace neighborhood generation (4 tests)
- [x] ProgramSpace implementation
- [x] TEST: MdlEvaluator scoring (5 tests)
- [x] MdlEvaluator implementation
- [x] TEST: SearchEngine finds shortest program for simple cases (3 tests)
- [x] SearchEngine implementation (evolutionary + hill-climbing)

### Phase 4: Meta Evolution (TDD) — 15 tests
- [x] TEST: Archive add/sample/diversity (8 tests)
- [x] Archive implementation
- [x] TEST: Mutator program mutation (3 tests)
- [x] Mutator implementation
- [x] TEST: EvolutionarySelector fitness-based selection (4 tests)
- [x] EvolutionarySelector implementation

### Phase 5: Agent Kernel (TDD) — 11 tests
- [x] TEST: GoalSetter intrinsic motivation (4 tests)
- [x] GoalSetter implementation
- [x] TEST: Planner action sequence from state to goal (2 tests)
- [x] Planner implementation
- [x] TEST: AgentKernel full loop (5 tests)
- [x] AgentKernel implementation

### Phase 6: Integration & Verification
- [x] main.cpp demo entry point
- [x] Full build clean compile (zero warnings, -Wall -Wextra -Wpedantic -Werror)
- [x] All 64 tests pass
- [ ] PR creation

## Design Decisions

1. **C++20 features**: concepts, ranges, std::variant, constexpr, structured bindings, modules (if supported)
2. **No external ML framework**: pure C++ for the kernel logic; world model uses simple matrix ops
3. **Value semantics**: prefer value types; use std::variant for Program AST
4. **Dependency injection**: all domains depend on interfaces, not concretions (DIP)
5. **GoogleTest**: TDD with test-first for all domain logic
