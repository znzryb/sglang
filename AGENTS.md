# Project contribution instructions

## Commit attribution

- Do not add AI assistants, models, or vendors as commit co-authors. In
  particular, do not add `Co-Authored-By` trailers for Codex, OpenAI, Claude,
  or other AI tools, even if a general commit workflow recommends them.
- Use the contributor's existing Git identity; do not change Git author or
  committer configuration to an AI identity.
- Preserve existing copyright notices, license headers, and attribution to
  human contributors. Follow any upstream contribution and disclosure requirements.

## Branch names

- Branch names must not contain `codex`, including `codex/` and `codex-`.
- Use a concise name describing the change, such as `fix-parser-validation`.
  Use a user-specified name when it satisfies the rule above.
- Make contributions on a separate branch, never directly on `main`.

## Contribution workflow

- Read `docs/docs/developer_guide/contribution_guide.mdx` and applicable
  instructions in `.claude/rules/` before changing implementation code.
- Follow nested `AGENTS.md` instructions for files in their scope.
- Before starting an issue, check its recent discussion and related pull
  requests to avoid duplicating ongoing work.
- Keep each contribution focused and validate the behavior it changes.
  Report the checks actually run and any remaining validation limits.
