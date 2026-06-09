cmake_minimum_required(VERSION 3.20)
project(universal_intelligence_kernel
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Minimal self-evolving AGI kernel using world models and symbolic descent"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# GCC 12 emits spurious -Warray-bounds and -Wstringop-overflow false positives
# in Release builds when inlining std::vector copy through Tensor value types.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
    add_compile_options(-Wno-array-bounds -Wno-stringop-overflow)
endif()

# ── Library: core kernel ──
add_library(uik_core
    src/world_model/latent_state.cpp
    src/world_model/dynamics_predictor.cpp
    src/world_model/novelty_detector.cpp
    src/world_model/world_model.cpp
    src/symbolic_descent/dsl.cpp
    src/symbolic_descent/program_space.cpp
    src/symbolic_descent/mdl_evaluator.cpp
    src/symbolic_descent/search_engine.cpp
    src/symbolic_descent/rule_library.cpp
    src/meta_evolution/archive.cpp
    src/meta_evolution/mutator.cpp
    src/meta_evolution/evolutionary_selector.cpp
    src/meta_evolution/parameter_adapter.cpp
    src/meta_evolution/self_modifier.cpp
    src/meta_evolution/safety_guard.cpp
    src/meta_evolution/interpreter.cpp
    src/agent/goal_setter.cpp
    src/agent/planner.cpp
    src/agent/agent_kernel.cpp
    src/agent/grid_world.cpp
    src/agent/arc_env.cpp
    src/agent/physics_env.cpp
    src/agent/curriculum.cpp
    src/agent/arc_loader.cpp
    src/agent/arc_benchmark.cpp
)
target_include_directories(uik_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# ── Executable ──
add_executable(uik_main src/main.cpp)
target_link_libraries(uik_main PRIVATE uik_core)

# ── Testing ──
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_uik_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE uik_core GTest::gtest_main)
    # GCC 12 emits a false-positive -Wrestrict inside gtest type-util headers
    target_compile_options(${TEST_NAME} PRIVATE -Wno-restrict)
    gtest_discover_tests(${TEST_NAME})
endfunction()

add_uik_test(test_world_model      tests/world_model/test_world_model.cpp)
add_uik_test(test_symbolic          tests/symbolic_descent/test_symbolic_descent.cpp)
add_uik_test(test_rule_library      tests/symbolic_descent/test_rule_library.cpp)
add_uik_test(test_meta_evolution    tests/meta_evolution/test_meta_evolution.cpp)
add_uik_test(test_param_adapter     tests/meta_evolution/test_parameter_adapter.cpp)
add_uik_test(test_agent             tests/agent/test_agent.cpp)
add_uik_test(test_grid_world        tests/agent/test_grid_world.cpp)
add_uik_test(test_serialization     tests/common/test_serialization.cpp)
add_uik_test(test_logger            tests/common/test_logger.cpp)
add_uik_test(test_integration       tests/integration/test_integration.cpp)
add_uik_test(test_self_modifier      tests/meta_evolution/test_self_modifier.cpp)
add_uik_test(test_safety_guard       tests/meta_evolution/test_safety_guard.cpp)
add_uik_test(test_interpreter        tests/meta_evolution/test_interpreter.cpp)
add_uik_test(test_arc_env            tests/agent/test_arc_env.cpp)
add_uik_test(test_physics_env        tests/agent/test_physics_env.cpp)
add_uik_test(test_curriculum          tests/agent/test_curriculum.cpp)
add_uik_test(test_extended_dsl       tests/symbolic_descent/test_extended_dsl.cpp)
add_uik_test(test_guided_search      tests/symbolic_descent/test_guided_search.cpp)
add_uik_test(test_v04_integration    tests/integration/test_v04_integration.cpp)
add_uik_test(test_v05_gaps           tests/integration/test_v05_gaps.cpp)
add_uik_test(test_v06_deep_fixes     tests/integration/test_v06_deep_fixes.cpp)

# ARC benchmark test (needs data path)
add_executable(test_arc_benchmark tests/integration/test_arc_benchmark.cpp)
target_link_libraries(test_arc_benchmark PRIVATE uik_core GTest::gtest_main)
target_compile_options(test_arc_benchmark PRIVATE -Wno-restrict)
target_compile_definitions(test_arc_benchmark PRIVATE ARC_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
gtest_discover_tests(test_arc_benchmark)
