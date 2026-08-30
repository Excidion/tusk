# Development and release tasks. CI runs these same recipes.

# List the available recipes.
default:
    @just --list

# Run every pre-commit hook across the whole repository.
# The file list is passed explicitly because pre-commit's own --all-files
# needs git 2.31 for `git ls-files --deduplicate`.
lint:
    git ls-files -z | xargs -0 uv run pre-commit run --files

# Run the default test suite.
test:
    uv run pytest

# Cross-check computed values against featuretools.
test-differential:
    uv run --group validation pytest -m differential

# Measure performance against relbench datasets.
benchmark:
    uv run --group benchmark pytest -m benchmark

# Build the documentation site into site/.
docs:
    uv run --group docs zensical build --clean

# Build the wheel and the source distribution into dist/.
build:
    uv build

# Everything a release has to pass. CI gates pull requests on a subset,
# leaving the featuretools cross-check to its nightly run.
check: lint test test-differential

# Bump the version, tag it, and push, which starts the release pipeline.
release bump="patch":
    #!/usr/bin/env bash
    set -euo pipefail
    branch="$(git branch --show-current)"
    if [ "$branch" != "main" ]; then
        echo "Releases are cut from main, but you are on $branch." >&2
        exit 1
    fi
    if ! git diff --quiet HEAD; then
        echo "The working tree has uncommitted changes. Commit or stash them, then retry." >&2
        exit 1
    fi
    git fetch --quiet origin main
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        echo "Local main differs from origin/main. Pull or push, then retry." >&2
        exit 1
    fi
    just check
    uv version --bump {{bump}}
    version="$(uv version --short)"
    git commit --all --message "release: v$version"
    git tag "v$version"
    git push --follow-tags
    echo "Pushed v$version. The release pipeline takes it from here."
