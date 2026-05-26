# AGENTS.md

Quick reference for AI coding agents (Claude Code, Codex, Cursor, Gemini, etc.)
and human contributors making commits in this repo. Covers commit conventions
and the local `commit-msg` hook.

For the **release process** (version bumps, `cz bump`, publish workflow,
Trusted Publisher setup), see the
[Release docs](website/content/docs/development/release.mdx).

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
- `.githooks/commit-msg` — local enforcement at commit time
- `.githooks/pre-push` — runs `make ci` (includes commit lint) before push
- `Makefile` — `make lint-commits`, `make install`, `make ci`
- `.github/workflows/ci.yml` — CI gate (also runs `lint-commits`)
- [Release docs](website/content/docs/development/release.mdx) — version bumps + publish workflow
