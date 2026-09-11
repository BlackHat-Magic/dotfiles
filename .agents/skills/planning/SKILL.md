---
name: planning
description: Distill an idea to shared understanding or produce a repository-informed implementation plan. Use for design clarification and non-trivial implementation planning.
---

# Planning

Choose the mode from the request. Planning is read-only unless the user explicitly asks to save the resulting plan.

## Common approach

1. Inspect the repository, governing documentation, current implementation, tests, and relevant history before asking questions.
2. Investigate facts available from code, configuration, documentation, or tools instead of asking the user to retrieve them.
3. Separate discovered facts from decisions that require user intent.
4. Resolve decisions in dependency order. Ask concise rounds: prerequisite decisions first, then only questions enabled by those answers. Group independent questions when that reduces delay.
5. Give a recommendation and rationale with each decision. Compare alternatives only when they differ meaningfully in behavior, cost, risk, or reversibility.
6. Turn the agreed behavior into checkable acceptance criteria, including important success, failure, boundary, and out-of-scope cases.
7. Keep depth proportional to the work. Prefer existing project patterns and avoid unrelated refactoring.

## Mode 1: Distillation only

Use when the user wants to clarify, stress-test, or reach agreement on an idea without requesting an implementation plan.

- Build a concise statement of purpose, users, scope, constraints, key decisions, risks, and acceptance criteria.
- Present the distilled understanding and ask for confirmation.
- Stop after shared-understanding confirmation. Do not write an implementation plan or implement the idea.

## Mode 2: Implementation planning

Use when the user requests a plan suitable for later implementation or compliance review.

After resolving necessary decisions, present a proportionate implementation-ready Markdown plan containing:

- objective and originating requirements;
- discovered current state and governing constraints;
- chosen approach and meaningful tradeoffs;
- affected components and files when known;
- ordered implementation steps with dependencies and data/control flow where relevant;
- error handling, compatibility, migration, and rollback considerations when applicable;
- testing and verification commands;
- checkable acceptance criteria and explicit non-goals.

Default to Markdown in the response. Save Markdown only when requested, using an existing project documentation location when available. HTML or browser output is optional and only produced when explicitly requested.

Stop after presenting the plan. Do not begin implementation automatically.
