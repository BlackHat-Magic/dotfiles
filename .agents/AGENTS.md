# Interpreting Instructions

- Take instructions literally. Do not turn them into logic puzzles or infer unrequested work.
- Every stated requirement matters. If a material ambiguity remains after investigation, ask for clarification.
- Keep changes inside the requested scope and avoid destructive actions that were not explicitly authorized.

## Questions Are Read-Only

- A question asks for an answer; it does not authorize changes.
- Read-only investigation is allowed when needed to answer accurately, including reading files and running non-mutating commands.
- Ask before making changes that the user did not request.

# Coding Preferences

- Read `README.md` and `CONTRIBUTING.md` when they are present.
- Update existing documentation only when the requested change makes it stale. Do not create documentation ceremonially.
- Keep things simple. I absolutely hate complexity unless it is absolutely necessary, and you should, too.
- Prefer precise static types and minimize type ambiguity.
- Comments are good, but do not comment every line of code. Prefer to document when and why rather than what and how (the latter is in the function signature and implementation)

## TypeScript

- Avoid `any`; prefer specific types and designs that localize future changes.
- For greenfield work without contrary requirements, prefer Svelte, Bun, and Cloudflare.

## Python

- Use type annotations to make usage and boundaries clear.
- Prefer type-safe code even when a less explicit style would be more idiomatic.
- Where reasonable, treat Python as a statically typed language.
