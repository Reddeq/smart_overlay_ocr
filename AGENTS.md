# AGENTS.md

## Purpose

These instructions govern how the coding agent works in this repository. They define development practices, not product requirements or architecture. Consult the relevant project documentation and source code for implementation details.

## Scope and autonomy

- Implement only the requested change and the work strictly necessary to support it.
- Do not implement speculative features, unrelated cleanup, or opportunistic refactors.
- Prefer the smallest maintainable change that solves the underlying problem.
- Make routine, low-risk implementation decisions independently and follow established conventions.
- For material ambiguity affecting behavior, compatibility, or design, explain the trade-off and request a decision before making an irreversible or broad change.
- Do not silently alter gameplay, user-visible behavior, public contracts, or persisted data beyond the task's scope.

## Understanding the task

- Start with targeted inspection of the relevant code, tests, and documentation.
- Expand the search only as necessary to understand dependencies and likely side effects.
- Review repository status before editing; preserve unrelated or user-authored changes.
- Treat existing behavior as intentional until evidence indicates otherwise, but do not assume existing code is infallible.
- Use established patterns when appropriate; do not perpetuate a known defect merely for consistency.
- For complex changes, outline a short approach and key risks before implementation. For simple changes, proceed directly.

## Requirements and documentation

- The current user request defines the intended change, subject to explicit safety and compatibility constraints.
- Consult relevant specifications and other documentation, but verify that they describe the current implementation.
- When documentation, source code, and the request disagree, do not silently choose a convenient interpretation. Identify material conflicts and resolve them before changing behavior.
- Do not treat historical plans as instructions to implement future features.
- Update documentation only when the change makes existing documentation inaccurate or the task requests it.
- Keep architecture and feature details out of this file; maintain them in their dedicated documentation.

## Implementation quality

- Prefer clear, cohesive code with explicit responsibilities and minimal indirection.
- Reuse existing utilities and abstractions when they fit; avoid premature frameworks, unnecessary interfaces, and duplicate logic.
- Respect established naming, formatting, and file organization unless change is justified.
- Avoid introducing dependencies without a concrete need and approval where required.
- Avoid broad exception handling that hides faults; handle expected failures explicitly and report unexpected failures with useful context.
- Keep diagnostics useful and bounded; avoid excessive logging and sensitive information in logs.
- Comment non-obvious decisions and constraints rather than restating what the code does.

## APIs and external integrations

- Do not invent APIs, signatures, events, or lifecycle guarantees.
- Verify uncertain behavior against the versions actually used by the project: referenced assemblies, installed source, tests, or reliable documentation.
- Prefer supported extension points over invasive patches or workarounds where practical.
- If an external API cannot be verified, state the uncertainty rather than disguising it with fake production implementations.

## Compatibility and runtime safety

- Preserve backward compatibility with saves, configuration, stable identifiers, and public interfaces unless a change is explicitly authorized.
- Consider data migration when changing persisted formats.
- Optimize based on the relevant performance risk; do not complicate simple code without evidence of benefit.

## Verification

- After meaningful code changes, run the relevant build and automated checks available in the environment.
- Add or update focused tests for behavior that can reasonably be tested automatically.
- When feasible, cover success, failure, edge cases, and regressions related to the change.
- Use real project dependencies for build validation; do not add fake stubs solely to manufacture a passing build.
- Review newly introduced warnings; do not suppress them without justification.
- If a test cannot run, report why and provide a short manual check where appropriate.
- If repeated fixes do not improve the result, stop guessing, summarize the evidence, and identify the next diagnostic step.

## Repository hygiene

- Do not change unrelated files, generated artifacts, or machine-specific settings without necessity.
- Do not commit dependencies, game binaries, build outputs, local paths, credentials, or secrets.
- Respect `.gitignore` and local untracked configuration.
- Do not overwrite or discard existing user changes.
- Do not commit, push, reset, or perform destructive Git operations unless requested.

## Communication and completion

- Keep progress and final reports concise and factual; avoid dumping full files or large logs unless asked.
- Report what changed, what was verified and with which result, and any relevant unresolved risks or manual checks.
- Mention material deviations from the request or documentation.
- Do not claim completion if an essential requirement is unimplemented or unverified; clearly distinguish those cases.
- Do not automatically suggest a new milestone or continue into another feature after completing the task.
