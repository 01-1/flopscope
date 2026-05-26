# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Gemini, etc.) and
human contributors working in this repo. Covers commit conventions, version
bumps, and the release process.

## Commit messages — Conventional Commits

Every commit on this repo MUST follow the
[Conventional Commits](https://www.conventionalcommits.org/) spec. The
title format is:

```
<type>[(scope)][!]: <subject>
```

### Allowed types

| Type       | When to use                                          |
|------------|------------------------------------------------------|
| `feat`     | A new user-visible feature                           |
| `fix`      | A bug fix                                            |
| `docs`     | Docs-only change                                     |
| `style`    | Formatting / whitespace; no logic change             |
| `refactor` | Code restructuring with no behavior change           |
| `perf`     | Performance improvement                              |
| `test`     | Adding or fixing tests                               |
| `build`    | Build system, deps, packaging                        |
| `ci`       | CI workflows                                         |
| `chore`    | Tooling / housekeeping that doesn't fit above        |
| `revert`   | Reverting a prior commit                             |

(Source of truth: `.gitlint`.)

### Rules

- Title MUST start with one of the types above
- Scope is optional, in parens: `fix(reduction-cost): handle 0-d arrays`
- Breaking changes: append `!` after the type/scope, OR put `BREAKING CHANGE:`
  in the footer
- Title MUST be ≤ 90 chars
- Imperative mood ("add X" not "added X")
- No trailing period

### Examples

GOOD:
```
fix(reduction-cost): handle 0-d arrays in _normalize_axis
test(numpy-compat): xfail TestSVDHermitian variants
chore: bump scipy floor to >=1.10
feat!: rewrite einsum cost model to direct-event α/M
```

BAD:
```
update stuff             ← missing type
FIX: broken thing        ← wrong case
Fix bug                  ← capital + missing colon
fix(): empty scope       ← empty scope
fix some bug.            ← trailing period
```

## Version bumps

Versions follow Semantic Versioning. While at `0.x`, the project is
API-unstable — breaking changes bump the **minor** digit, not the major.
Commit types map to bumps as follows:

| Commit type                       | `0.x` bump                  | `≥1.0` bump                |
|-----------------------------------|-----------------------------|----------------------------|
| `fix:`                            | patch (`0.2.0` → `0.2.1`)   | patch                      |
| `feat:`                           | minor (`0.2.0` → `0.3.0`)   | minor                      |
| `feat!:` / `BREAKING CHANGE:`     | minor (`0.2.0` → `0.3.0`)   | **major** (`1.4` → `2.0`)  |
| Other types (`chore`, `docs`, …)  | no release                  | no release                 |

Commitizen handles this automatically via `major_version_zero = true` in
the `[tool.commitizen]` block of `pyproject.toml`. Flip that to `false`
when we cut `v1.0.0`.

## Release process

We release with [commitizen](https://commitizen-tools.github.io/commitizen/).
End-to-end flow:

```bash
# 1. Make sure local main is at origin/main
git checkout main && git pull origin main

# 2. Preview the next version + changelog (no writes)
uv run cz bump --dry-run

# 3. Cut the release: bumps pyproject.toml version, prepends to
#    CHANGELOG.md, creates the v<x.y.z> git tag.
uv run cz bump

# 4. Push the commit + tag. Tag push triggers the PyPI publish workflow.
git push --follow-tags
```

Pre-releases: `uv run cz bump --prerelease alpha` → produces tags like
`v0.3.0a0`.

## Local commit hooks

The repo ships a `commit-msg` hook at `.githooks/commit-msg` that runs
gitlint on every `git commit`. To enable it (one-time per clone):

```bash
make install   # `uv sync --all-extras` + `git config core.hooksPath .githooks`
```

After that, any commit with a bad message is rejected at commit time,
instead of failing later in CI.

## For AI agents — validating messages programmatically

Before committing, validate the message:

```bash
# Validate a candidate message file
uv run gitlint --msg-filename path/to/message-file

# One-off validation of a candidate string
echo "fix: my proposed message" | uv run gitlint --ignore-stdin --staged

# Commitizen has a parallel check (same rules, slightly different errors)
uv run cz check --message "fix: my proposed message"
```

If `gitlint` exits non-zero, the message will also fail the `commit-msg`
hook and the CI `lint-commits` job. Fix the message before committing
rather than retrying via `git commit --amend`.

## Related references

- `.gitlint` — linter config (the conventional types live here)
- `[tool.commitizen]` in `pyproject.toml` — release tooling config
- `.githooks/commit-msg` — local enforcement at commit time
- `.githooks/pre-push` — runs `make ci` (includes commit lint) before push
- `Makefile` — `make lint-commits`, `make install`, `make ci`
- `.github/workflows/ci.yml` — CI gate (also runs `lint-commits`)
