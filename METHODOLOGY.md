## Universal software project methodology (concise, project-agnostic)

### 1) Clarify the intent before coding

* State the **goal**, **constraints**, **assumptions**, and **unknowns**.
* If ambiguous, document a **safe default** and/or list options with trade-offs.

### 2) Deliver the smallest useful increment

* Build only what is required to meet the goal **now**.
* Avoid speculative features and unnecessary abstraction.
* Minimize dependencies; **justify** each new dependency or custom build by overall risk and maintenance cost.

### 3) Keep changes focused and consistent

* Change only what the goal requires, plus necessary integration points.
* Defer unrelated refactors unless they block delivery.
* Follow existing conventions; if none exist, define lightweight ones and apply consistently.

### 4) Make done measurable and verified

* Define **success criteria** (acceptance criteria, metrics, or observable outcomes).
* Validate with automated checks when feasible; otherwise provide clear manual verification steps.
* Record what was verified and what remains unverified.

### 5) Engineer robustness proportional to risk

* Handle realistic edge cases at **Input/Output (I/O)** boundaries and external interactions.
* Treat "impossible" cases as assumptions and document them.
* Address quality attributes as needed (security, privacy, reliability, performance, compliance), scaling rigor to impact.

### 6) Design for change without over-engineering

* Apply **KISS**: prefer the simplest architecture and design that fit the current scope and risk.
* Apply **YAGNI**: do not add extension points, layers, or abstractions before there is a concrete need.
* Separate concerns: keep business logic distinct from I/O boundaries such as HTTP, CLI, files, databases, and external APIs.
* Prefer high cohesion and low coupling: each module/component should have one primary responsibility and few reasons to change.
* Hide internal implementation details behind clear contracts so callers depend on behavior, not internals.
* Prefer composition and small focused interfaces over deep inheritance or large multi-purpose APIs.
* Design for testability: structure code so core behavior can be verified without heavy reliance on frameworks, networks, or global state.
* Add interfaces/seams at boundaries only when they materially reduce coupling or improve testability.
* Treat **SOLID** as a review heuristic, not a mandatory checklist. Apply it only when it improves correctness, readability, or ease of change.
