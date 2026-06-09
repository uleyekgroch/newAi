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
    src/agent/goal_setter.cpp
    src/agent/planner.cpp
    src/agent/agent_kernel.cpp
    src/agent/grid_world.cpp
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

add_uik_test(test_world_model    tests/world_model/test_world_model.cpp)
add_uik_test(test_symbolic        tests/symbolic_descent/test_symbolic_descent.cpp)
add_uik_test(test_rule_library    tests/symbolic_descent/test_rule_library.cpp)
add_uik_test(test_meta_evolution  tests/meta_evolution/test_meta_evolution.cpp)
add_uik_test(test_agent           tests/agent/test_agent.cpp)
add_uik_test(test_grid_world      tests/agent/test_grid_world.cpp)
