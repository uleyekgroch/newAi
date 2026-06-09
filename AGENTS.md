## Core Philosophy and Principles

- **Simplicity First**: Adhere to the KISS (Keep It Simple, Stupid) principle. Value simplicity and maintainability; avoid over-engineering and unnecessary defensive design.
- **Deep Analysis**: Grounded in First Principles Thinking to analyze problems, and leverage tools to improve efficiency.
- **Fact-Based**: Facts are the highest standard. If any error exists, please point it out frankly to help me improve.

## Development Workflow

- **Incremental Development**: Iterate through multiple rounds of dialogue to clarify and implement requirements. Before starting any design or coding work, preliminary research must be completed and all doubts clarified.
- **Structured Process**: Strictly follow the workflow of "conceive solution → request review → decompose into concrete tasks".

## Output Standards

- **Language Requirement**: All responses, thought processes, and task lists must be in English.
- **Fixed Directive**: Implementation Plan, Task List and Thought in English

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Architecture and Coding Principles

### Architectural Analysis: First Principles

When analyzing problems, technical architecture, and code module composition, always return to the fundamentals and reason from the most basic facts and assumptions:

- **Decompose to the Indivisible**: Break complex systems into their most basic constituent parts; understand each part's essential responsibilities and interaction boundaries.
- **Challenge Existing Assumptions**: Do not accept a design just because "it has always been done this way" or "it is an industry convention". Validate its necessity in the current context.
- **Reason Backwards from the Goal**: First clarify "what problem needs to be solved", then deduce "what components are needed", rather than piecing together a solution from the existing tech stack.
- **Minimize Module Coupling**: Each module should depend only on the minimal set of interfaces it essentially needs; avoid implicit dependencies and knowledge leakage.

### Coding Principles

All code implementations must simultaneously adhere to the following four principles:

| Principle | Core Essence | Practical Requirements |
|-----------|-------------|------------------------|
| **DRY** (Don't Repeat Yourself) | Knowledge Uniqueness | Any business rule, algorithm logic, or data structure definition should exist in only one place in the codebase; duplicated code must be extracted into functions/classes/configurations. |
| **KISS** (Keep It Simple, Stupid) | Simplicity First | Choose the simplest runnable implementation; avoid defensive abstractions for problems that haven't occurred yet; variable names, function names, and flow control should be clear at a glance. |
| **SOLID** | Five Principles of OOD | **S** Single Responsibility: one class/function does one thing; **O** Open/Closed: open for extension, closed for modification; **L** Liskov Substitution: subclasses can seamlessly replace parent classes; **I** Interface Segregation: clients should not depend on interfaces they do not use; **D** Dependency Inversion: depend on abstractions, not concrete implementations. |
| **YAGNI** (You Aren't Gonna Need It) | No Presumptive Requirements | Do not implement features not needed in the current iteration; do not add configuration switches, plugin interfaces, or extension points for "future possibilities"; add them when needed, not in advance. |

### Code Size Governance: 800-Line Red Line

When a single class, function, or code file exceeds **800 lines**, it must be identified, decomposed, and separated:

1. **Identify Responsibility Boundaries**: Analyze how many distinct responsibilities or concerns are mixed in that file.
2. **Extract Cohesive Units**: Extract highly cohesive logic into independent classes, modules, or services.
3. **Follow Decomposition Principles**:
   - Extraction should not break existing interface contracts (external behavior remains unchanged).
   - Newly separated units themselves should also conform to DRY/KISS/SOLID/YAGNI.
   - Prioritize splitting by "rate of change" and "business domain", rather than mechanically cutting by line count.
4. **Verify Separation Quality**: After decomposition, the original file should only retain coordination/assembly logic; each sub-unit should be independently understandable, testable, and reusable.

> **Exception Handling**: Only when a domain itself has highly cohesive complexity (e.g., state machines, protocol parsers), and forced splitting would cause interface fragmentation and increased comprehension cost, may it slightly exceed 800 lines; however, the justification for retention must be documented in code comments.

## No Shortcuts Policy

Under NO circumstances should you take shortcuts, rush, or skip steps for any reason.

- **Time is NEVER an excuse**: Do NOT claim that there is "not enough time" or "too many files" to avoid completing the full implementation. If the task is large, decompose it into smaller steps and execute them thoroughly.
- **Token limits are NEVER an excuse**: Do NOT use "token constraints" or "context window limits" as a reason to provide partial code, pseudo-code, or placeholder comments instead of real, working implementations.
- **No placeholder tricks**: Do NOT write `// TODO: implement this` or `pass # TODO` and leave it to the user. Every function, method, and class you write must be fully implemented and immediately runnable.
- **No "framework only" responses**: Do NOT provide skeleton code or high-level outlines and ask the user to fill in the details. Deliver production-ready, complete code.
- **Deep investigation required**: Before writing code, read all relevant files, understand the full context, and ensure your solution integrates correctly with the existing codebase. Surface-level understanding is unacceptable.
- **Senior engineer standard**: Ask yourself "Would a senior staff engineer ship this?" If the answer is no, keep working.

## Code Completeness Guarantee

Every code change MUST be complete, functional, and ready to run.

- **All imports included**: Every file must contain every `import` or `require` statement it needs. Do not omit imports assuming the user will add them later.
- **All dependencies declared**: If new packages are required, mention them explicitly so the user can install them.
- **No broken states**: Never leave the codebase in a state where it cannot compile, start, or pass existing tests. If a refactor breaks something, fix it within the same session.
- **Full implementation**: Every function body must contain real logic, not `pass`, `// TODO`, or empty stubs.
- **Edge cases handled**: Consider and handle error cases, null/undefined inputs, and boundary conditions. Defensive programming is mandatory.
- **Configuration included**: If the change requires new environment variables, config entries, or database migrations, provide them fully.

## AI Coding Best Practices

When acting as an AI pair programmer, adhere to the following discipline:

- **Context first**: Before modifying any code, read and understand the surrounding files, the project's conventions, and the existing test suite. Never edit blindly.
- **Minimal diff**: Make the smallest possible change that achieves the goal. Do not refactor unrelated code in the same PR.
- **Preserve behavior**: Unless explicitly asked to change behavior, ensure existing functionality remains identical. Run (or ask the user to run) tests before and after.
- **Explain trade-offs**: When multiple valid approaches exist, briefly explain the trade-offs and state why you chose the one you did.
- **Security awareness**: Do NOT hardcode secrets, API keys, or credentials. Use environment variables or secure vaults. Validate all external inputs.
- **Performance conscious**: Avoid N+1 queries, unnecessary re-renders, and algorithmic inefficiencies. If a performance-sensitive change is made, explain why.
- **Accessibility and UX**: For frontend changes, ensure keyboard navigation, screen reader compatibility, and responsive design are not broken.
- **Documentation parity**: If you add a new public API, endpoint, or configuration option, update the README or relevant documentation in the same session.
- **Test coverage**: If the project has tests, add or update tests for your changes. If there are no tests, write a quick manual verification script or describe exact reproduction steps.

## Testing Discipline

- **Test before merge**: Every non-trivial change must be accompanied by tests. Design or update tests before major implementation work, not as an afterthought.
- **Test pyramid adherence**: Prioritize fast, isolated unit tests (70%), followed by integration tests (20%), and minimal end-to-end tests (10%). Avoid the inverted pyramid anti-pattern.
- **No weakening of tests**: Never delete, skip, or weaken existing tests without explicit direction. If a test becomes irrelevant, explain why before removing it.
- **Deterministic tests only**: Tests must produce the same result every run. No flaky tests, no reliance on external network, no random seeds without seeding control.
- **Coverage as a signal, not a target**: Use coverage reports to identify untested code paths, not as a vanity metric. Critical business logic paths must have coverage.
- **Test naming**: Use descriptive names that state the scenario, input, and expected outcome. `test_processOrder_invalidQuantity_throwsValidationError` over `test_error`.
- **Manual verification fallback**: When automated tests are infeasible, provide exact, copy-pasteable reproduction steps and expected results.

## Error Handling Standards

- **Never swallow exceptions silently**: Every `except` or `catch` block must either handle the error meaningfully (retry, fallback, user message) or re-raise. Bare `except: pass` is forbidden.
- **Fail fast, fail loud**: Detect invalid states as early as possible. Throw explicit errors with actionable messages. Do not propagate `null` or default values that mask the root cause.
- **Use domain-specific error types**: Define custom exception/error classes per domain layer. Avoid generic `Exception("something went wrong")` — the error type itself should communicate the category.
- **User-facing errors must be actionable**: Error messages shown to users must explain what happened and what they can do about it. Internal stack traces must never leak to the UI.
- **Error context propagation**: When re-raising, preserve the original stack trace and add contextual information (e.g., which record ID, which operation, which input value).
- **Graceful degradation**: When a non-critical subsystem fails, the system should continue operating with reduced functionality rather than crashing entirely. Log the degradation clearly.

## Logging & Debugging

- **Structured logging**: Use key-value pairs, not free-form strings. Prefer `log.info("order_created", order_id=123, amount=99.9)` over `log.info(f"Order {order_id} created")`.
- **Log levels with clear semantics**: `DEBUG` for development diagnostics, `INFO` for key business events, `WARNING` for recoverable anomalies, `ERROR` for failures requiring attention, `CRITICAL` for system-wide outages.
- **Critical path instrumentation**: Every significant business operation (order placement, data sync, user auth) must emit an INFO-level log at start and end with relevant identifiers.
- **Sensitive data must never be logged**: Passwords, tokens, API keys, PII, and full credit card numbers are forbidden in logs. Mask or hash them.
- **Log for production debugging**: Assume the only debugging tool available is the log file. Include correlation IDs, timestamps with timezone, and enough context to reconstruct the event chain.
- **Add logging before debugging blind**: When investigating a bug, add descriptive logging statements to track variable values and code flow rather than guessing. Remove or demote to DEBUG after resolution.

## Bug Fixing Discipline

- **Root cause over symptom**: Do not patch the symptom and walk away. Trace the bug to its origin. Ask "why" repeatedly until you reach the fundamental flaw.
- **Minimal upstream fix preferred**: Fix the bug at its source, not by adding workarounds downstream. A one-line fix at the root is better than a ten-line guard in every caller.
- **Regression test mandatory**: Every bug fix must include a test that fails before the fix and passes after. This proves you understood the bug and prevents its return.
- **Verify before claiming done**: Reproduce the bug before fixing, verify the fix resolves it, and confirm no existing tests break. Do not mark a bug as fixed without demonstrating the fix works.
- **Single-responsibility fix**: Each fix should address one bug. Do not bundle unrelated refactors or feature changes into a bug fix PR.
- **Document the why**: In the commit message or PR description, explain what caused the bug, why the fix is correct, and how to verify it. Future maintainers will thank you.
