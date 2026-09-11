---
name: bash-cheatsheet
description: Documents the CLI tools installed on the machine that you have access to (e.g., `rg`, `fd`, etc.). Use to see what tools you have access to.
---

# Bash Cheatsheet

If this skill says to prefer a given tool, but the project uses something else, stick with what the project uses. Don't force a project to fit this skill.

Installed applications:
- `bun`: JavaScript/TypeScript package manager and runtime. Prefer over `npm`/`pnpm`.
- `fd`: A simple, fast, user-friendly alternative to `find` (also installed).
- `fzf`: A CLI fuzzy finder.
- `hadolint`: Dockerfile linter
- `just`: Task runner and build tool. Prefer over `make`.
- `mise`: Dev tool manager. If you need a dev tool that isn't installed that can't be managed with other installed dev tool managers like `bun`, `cargo`, `uv`, or `zig`, install it to *that* project only with `mise`.
- `rg`: A line-oriented search tool. Prefer over `grep` (also installed) unless `grep`'s semantics are needed.
- `uv`, `ruff`, and `ty`: Python package and virtual environment manager, linter, and type checker. Prefer over `pip` and other Python tools.
