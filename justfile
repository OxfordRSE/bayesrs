# Project task runner: `just --list` shows everything.
# CI runs these exact recipes, so local and CI behaviour never drift.

# List available recipes
default:
    @just --list

# Format all Rust code
fmt:
    cargo fmt --all

# Check formatting without modifying anything
fmt-check:
    cargo fmt --all --check

# Lint with clippy, treating warnings as errors
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Run the Rust test suite
test:
    cargo test --workspace

# Everything the CI rust job runs
ci-rust: fmt-check lint test
