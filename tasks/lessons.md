# Lessons Learned

## GCC 12 + C++20 Aggregate Initialization
- **Issue**: GCC 12 does not allow `Config{}` as a default argument for a constructor when `Config` is a nested struct with default member initializers, declared inside the enclosing class.
- **Fix**: Provide separate default constructor + explicit Config constructor instead of `explicit Ctor(Config config = Config{})`.
- **Rule**: When using nested config structs with DMI, always split into two constructors.

## GCC 12 + GoogleTest -Wrestrict False Positive
- **Issue**: GCC 12 emits a spurious `-Wrestrict` warning inside GoogleTest's `gtest-type-util.h` when compiled with `-Werror`.
- **Fix**: Add `-Wno-restrict` to test targets only: `target_compile_options(${TEST} PRIVATE -Wno-restrict)`.
- **Rule**: Keep `-Werror` on production code; suppress known false positives only in test targets.

## Unspecified Evaluation Order with std::move
- **Issue**: `Tensor({static_cast<Dim>(data.size())}, std::move(data))` — evaluation order of constructor arguments is unspecified in C++. `data.size()` may be evaluated after `std::move(data)` moves from the vector.
- **Fix**: Compute `Dim n = data.size();` before constructing: `Tensor({n}, std::move(data))`.
- **Rule**: Always capture values from a container before moving it in the same expression.

## Planner Action Space Mismatch
- **Issue**: Planner hardcoded `max_action_search = 8` but the world model may have fewer actions, causing `predict_next` to throw and truncate plans.
- **Fix**: Made action space configurable via `Planner::Config::action_space`.
- **Rule**: Never hardcode environment-dependent constants; always inject via configuration.
