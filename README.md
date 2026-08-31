# bayesrs
Rust library for Bayesian inference with an ask/tell interface

## Development

bayesrs is a Rust core (`crates/bayesrs-core`) with Python (PyO3/maturin), C (cbindgen), and R (extendr) bindings.
Working on the full project needs:

| Tool | What for |
|---|---|
| rustup | Rust toolchain manager; auto-installs the version pinned in `rust-toolchain.toml` |
| cbindgen | regenerating the committed C header |
| just | task runner — `just --list` shows all project recipes |
| uv | Python side: environments, maturin, pytest |
| build-essential | C compiler and make, for the C smoke tests and the R package |
| R + headers | building and checking the R package |

### Install (Ubuntu / WSL)

```sh
# Rust — rustup fetches the repo-pinned toolchain automatically on first build
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# Cargo-installed dev tools
cargo install cbindgen just

# uv — manages Python versions and runs maturin/pytest (no manual venvs needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# System packages: C toolchain, R, and headers needed to compile the R
# development packages below
sudo apt-get update
sudo apt-get install -y build-essential r-base r-base-dev \
  libuv1-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
  libfontconfig1-dev libfreetype6-dev libpng-dev libtiff-dev libjpeg-dev \
  libharfbuzz-dev libfribidi-dev libgit2-dev

# R development packages, into a personal library (no sudo; the system
# site-library is root-owned, and non-interactive R can't offer to create
# the personal one itself)
Rscript -e 'dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)'
Rscript -e 'install.packages(c("rextendr", "devtools", "testthat"), repos = "https://cloud.r-project.org")'
```

On macOS, replace the apt lines with `brew install r` (Xcode command line tools provide the C toolchain).
