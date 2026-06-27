---
name: release-version
description: Release workflow for roctop. Use when the user asks to release a version such as "release version v0.3.4", "release vX.Y.Z", or asks Codex to bump the package version, commit it, create the matching git tag, and write release notes for a roctop release.
---

# Release Version

Use this skill for the roctop repository release flow. Treat `vX.Y.Z` as the requested git tag and `X.Y.Z` as the package version.

## Workflow

1. Parse the requested version.
   - Accept `vX.Y.Z` or `X.Y.Z`.
   - Use `vX.Y.Z` for the tag and `X.Y.Z` inside files.
   - Stop and ask if the version is missing or not SemVer-like.

2. Inspect the repo before editing.
   - Run `git status --short`.
   - If unrelated user changes exist, leave them alone and commit only release-version files.
   - If the requested tag already exists, report where it points and ask before replacing it.

3. Bump roctop version files.
   - Update `pyproject.toml`: `[project] version = "X.Y.Z"`.
   - Update `src/roctop/__init__.py`: `__version__ = "X.Y.Z"`.
   - Update the README demo image tag from `/vOLD/docs/demo.svg` to `/vX.Y.Z/docs/demo.svg` when present.
   - Search for the previous version string and decide whether any remaining occurrence is release-specific or historical before editing it.

4. Verify the bump.
   - Prefer the project test command from `AGENTS.md`: `.venv/bin/python -m unittest discover -s tests`.
   - If a full suite is too expensive or the environment is missing dependencies, run the narrow version/CLI tests and report the limitation.
   - Confirm no old active version string remains in package metadata.

5. Commit the version bump.
   - Stage only files changed for the release bump.
   - Commit with `Bump version to X.Y.Z`.
   - Do not include unrelated local changes.

6. Create the git tag.
   - Create lightweight tag `vX.Y.Z` on the version bump commit.
   - Verify with `git log --oneline --decorate --max-count=3` or `git rev-parse --short vX.Y.Z`.
   - Do not push commits or tags unless the user explicitly asks.

7. Write the release note.
   - Determine the previous release tag with `git tag --list 'v*' --sort=-version:refname`, excluding the new tag.
   - Summarize `git log --oneline PREVIOUS_TAG..vX.Y.Z`.
   - Include a compare link: `https://github.com/nrhevu/roctop/compare/PREVIOUS_TAG...vX.Y.Z`.
   - Put the release note in the final response unless the user asks to write it to a file or GitHub release.

## Release Note Shape

Use this concise format:

```markdown
## roctop vX.Y.Z

One short overview sentence.

### Highlights

- User-facing change.
- User-facing change.

### Fixes

- Bug fix or stability improvement.

### Full Changelog

https://github.com/nrhevu/roctop/compare/vOLD...vX.Y.Z
```

Skip empty sections. Keep the wording user-facing; do not just copy commit subjects if a clearer summary is available.
