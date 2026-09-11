---
name: code-review
description: Review a change for requirement compliance and code quality without editing files. Use for diffs, branches, patches, or completed implementations, or when the user requests code review.
---

# Code Review

Remain read-only.

1. Establish the review target and comparison base. If the request does not state them, infer them from repository state and report the inference; ask only when the ambiguity would materially change the review.
2. Read governing `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, and relevant project docs when present. Find the originating issue, request, plan, or specification and preserve its intent.
3. Inspect the full diff and enough surrounding code, tests, configuration, and call sites to understand behavior. Do not review the patch in isolation.
4. Apply two separate lenses:
   - **Compliance:** missing, extra, or incorrect behavior relative to the originating requirements and acceptance criteria.
   - **Quality:** correctness, regressions, security, data safety, error handling, maintainability, and test adequacy.
5. Run relevant project checks when feasible. State exactly what ran and any limitations.

Report findings first, ordered by severity. Each finding must explain impact and include precise `file:line` evidence. Keep compliance and quality findings labeled separately; do not hide blocking issues among style suggestions. Then give a brief summary and verification status.

If there are no findings, say so and list residual gaps such as unrun checks, unavailable environments, or behavior that could not be exercised. Do not edit files, commit, or push.

Parallel reviewers are optional. Use them only when the change has genuinely independent review surfaces and the capability is available.
