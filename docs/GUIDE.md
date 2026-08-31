# bayesrs: an introduction, with worked examples

This is the friendly companion to `INTERFACE.md` (the precise specification) and `DECISIONS.md` (the design rationale).
It is written for statisticians and students: it assumes you know what MCMC is, and assumes nothing about software engineering or about any particular programming language.
The same worked example — a simple Bayesian linear regression — appears in Python, R, C, and Rust itself, and a second example (Section 5) adds a discrete parameter: was a signal present in a noisy recording, or not?

---

## 1. What bayesrs is

bayesrs is a library of MCMC samplers.
The samplers are written once, in a fast compiled language called Rust, and you use them from whichever language you already work in — Python, R, or C. You never need to write or read any Rust — though if you want the cost of talking to the library to be exactly zero, you can also use it from Rust directly (Section 4.4).

You bring one thing: a function that computes your **log-posterior** — the log of prior × likelihood, up to an additive constant (normalising constants can be dropped, as usual in MCMC).
The library brings everything else: proposals, accept/reject, storage of the chains, and convergence diagnostics.

---

## 2. The one idea to understand: *you* run the loop

Most MCMC software asks for your log-posterior function and then disappears to run the whole chain, calling your function internally whenever it needs it. bayesrs turns that inside out.
The sampler is a small machine that you turn by hand, two verbs at a time:

- **`ask()`** — the sampler hands you a batch of proposed parameter values, one row per chain.
- You compute the log-posterior of each row, any way you like, in your own language.
- **`tell(logp)`** — you hand the log-densities back.
  The sampler accepts or rejects each proposal, stores the result, and gets ready for the next round.

So your program contains a loop that looks like this (in any language):

```
while the sampler has fewer draws than you want:
    proposals = ask the sampler
    logp      = your log-posterior, evaluated at each proposed row
    tell the sampler logp
```

Why this way round?
Because the library then never needs to call into your language — which is the thing that makes cross-language software fragile and slow.
Everything that crosses between your code and the library is a plain block of numbers.
A side benefit: if your log-posterior has a bug and crashes, it crashes in *your* code, in your debugger, with your error messages.

Three details of the loop, all designed to remove surprises:

1. **The first `ask()` returns your starting points.**
   You provide starting values for every chain when you create the sampler (the library cannot invent them — it doesn't know your prior).
   The very first `ask()` simply hands those back so you can evaluate them, and the first `tell` records their densities.
   Your loop is identical from the first iteration.
2. **Count draws, not iterations.**
   The condition is "while the sampler has fewer than N draws", not "repeat N times".
   For a single basic sampler the two are the same, but a sampler that updates parameters in blocks (Section 5) needs one round *per block* per draw, and future samplers (e.g. NUTS) will need a variable number.
   A loop written this way keeps working unchanged in every case.
3. **Rows are evaluated independently.**
   Your job is only ever: for each row of the batch, return one number.
   You never need to know which row belongs to which chain.

### Infinities and errors

- Return **`-inf`** (minus infinity) for a proposal outside the support of your posterior.
  That is always legal: the sampler rejects it and moves on.
  (The one exception: a *starting point* with density zero is an error, because the chain would have nowhere to stand.)
- **`NaN`** ("not a number", the result of things like 0/0) is treated as an error, not a rejection.
  A NaN almost always means a bug in your log-posterior, and silently rejecting it could hide that bug for hours.
- Calling `ask()` twice without a `tell` in between, or `tell` without an `ask`, gives a clear error message.
  You cannot corrupt a chain by calling things in the wrong order.

---

## 3. Describing your parameters

Before creating a sampler you describe your parameters: their names, and what kind of quantity each one is.
This description is called the **parameter space**.
For the regression example:

| name | kind | meaning |
|---|---|---|
| `alpha` | `Real` | intercept — any real number |
| `beta` | `Real` | slope — any real number |
| `sigma` | `Bounded(0, ∞)` | noise standard deviation — must be positive |

The first release supports five kinds:

| kind | use it for |
|---|---|
| `Real` | unconstrained quantities (means, slopes, log-rates) |
| `Bounded(lo, hi)` | quantities in an interval; either end may be infinite |
| `Simplex(k)` | k non-negative numbers that sum to 1 (mixture weights, probabilities) |
| `CovMatrix(n)` | an n×n covariance matrix (symmetric, positive-definite) |
| `Categorical(k)` | one of k unordered states (a label, a model index, "is the signal there?") |

A categorical parameter's value travels as a whole number, 0 to k−1 — exactly representable, never rounded — and you can attach text labels for readability.
Discrete parameters need their own sampler, and usually appear alongside continuous ones; Section 5 shows how the two are combined.
(Ordered integer parameters — counts, where "current ± 1" is a sensible move — and more matrix types are planned for a later release.)

### Why the *kind* matters, not just the name

A random-walk sampler proposes by adding random steps.
If it stepped `sigma` directly it would constantly propose negative standard deviations, and every such proposal would be wasted.
Instead, the library internally works with transformed, unconstrained versions of your parameters (for `sigma`, its logarithm; for a covariance matrix, a clever factorisation), proposes freely there, and transforms back before showing you anything.

**You never see any of this.**
Two guarantees make it invisible:

1. Every value the sampler hands you is valid: `sigma` is always *strictly* positive — never exactly zero, even at the edge of floating-point precision — simplex weights always sum to one, covariance matrices are always genuinely positive-definite.
   You never need to check, and you never need `-inf` guards for these constraints.
   (Bounds are open intervals; endpoints carry no probability for a continuous parameter, so nothing is lost.
   Starting values must respect the bounds strictly too — `sigma = 0` is not a legal place to start a chain.)
2. Changing variables mathematically requires a **Jacobian correction** to the density.
   The library applies it internally.
   You write the log-posterior of your *actual* parameters — the density you would write on paper — and nothing else.
   Hand-deriving Jacobians is a rich source of silent errors, which is exactly why it's the library's job.

### One flat row per chain

The batch that `ask()` returns is a plain matrix: one row per chain, one column per parameter value, in the order you declared them.
For the regression that is 3 columns: `alpha, beta, sigma`.
So you *could* write `theta[0]`, `theta[1]`, `theta[2]`.

But numbered columns are fragile — add a parameter and every number shifts.
So the parameter space provides **`unpack`**, which wraps the same matrix in named pieces (`p.alpha`, `p.beta`, `p.sigma`), each correctly shaped, at no cost.
Its mirror image **`pack`** builds a valid starting-value matrix from named pieces, which is how you'll set up starting points without counting columns.
The examples below use both.

---

## 4. The worked example

The model, throughout, is straight-line regression with unknown noise:

$$y_i = \alpha + \beta x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathrm{N}(0, \sigma^2), \qquad i = 1, \dots, n$$

with priors

$$\alpha \sim \mathrm{N}(0, 10^2), \qquad \beta \sim \mathrm{N}(0, 10^2), \qquad \sigma \sim \text{Half-Normal}(5).$$

Dropping additive constants, the log-posterior is

$$\log p(\alpha, \beta, \sigma \mid y) \;=\; -\,n \log \sigma \;-\; \sum_i \frac{(y_i - \alpha - \beta x_i)^2}{2\sigma^2} \;-\; \frac{\alpha^2}{200} \;-\; \frac{\beta^2}{200} \;-\; \frac{\sigma^2}{50}.$$

All four programs use the same ten hard-coded data points (true values roughly α = 1, β = 2), run **4 chains** of the plain random-walk Metropolis sampler for **20,000 draws** each, discard the first 5,000 as burn-in, and report posterior means and the R̂ convergence diagnostic.

```
x:  0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5
y:  1.1  1.8  3.2  4.1  4.9  6.2  7.1  7.9  9.2 10.1
```

> **Note on the sampler choice.**
> Plain `RandomWalkMetropolis` uses a fixed proposal size and can mix slowly if your parameters have very different scales.
> `AdaptiveMetropolis` tunes its own proposals as it runs and is a drop-in replacement — change one word in any of the programs below.
> We use the plain version here because it has nothing up its sleeve.

---

### 4.1 Python

Install with `pip install bayesrs`.
If you know a little NumPy (Python's array library) you know everything needed here.
One idiom to explain up front: `p.alpha` is a vector with one entry per chain (length 4), and `x` is the data vector (length 10).
Writing `p.alpha[:, None]` turns the chain vector into a column, so that adding it to `x` produces a 4×10 table — every chain's predictions computed in one line, no loop over chains needed.

```python
import numpy as np
from bayesrs import ParamSpace, Real, Bounded, RandomWalkMetropolis, diagnostics

# --- data -------------------------------------------------------------
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
y = np.array([1.1, 1.8, 3.2, 4.1, 4.9, 6.2, 7.1, 7.9, 9.2, 10.1])
n = len(y)

# --- the parameter space ---------------------------------------------
space = ParamSpace(
    alpha = Real(),
    beta  = Real(),
    sigma = Bounded(0, None),      # None means "no upper bound"
)

# --- the log-posterior ------------------------------------------------
# theta is a matrix: one row per chain, columns alpha, beta, sigma.
# unpack() gives us the columns by name. Every quantity below is
# computed for all 4 chains at once; the function returns 4 numbers.
def log_post(theta):
    p = space.unpack(theta)
    resid = y - (p.alpha[:, None] + p.beta[:, None] * x)      # 4 x 10 table
    log_lik   = -n * np.log(p.sigma) - (resid**2).sum(axis=1) / (2 * p.sigma**2)
    log_prior = -p.alpha**2 / 200 - p.beta**2 / 200 - p.sigma**2 / 50
    return log_lik + log_prior
# Note: no check that sigma > 0 — the library guarantees it (Section 3).

# --- starting points: one value per chain, spread out ----------------
x0 = space.pack(
    alpha = np.array([ 0.0, 1.0, -1.0, 2.0]),
    beta  = np.array([ 1.0, 2.0,  0.0, 3.0]),
    sigma = np.array([ 1.0, 0.5,  2.0, 1.5]),
)

# --- create the sampler and run the loop -----------------------------
s = RandomWalkMetropolis(space, x0, seed=42)

while s.n_draws < 20_000:
    s.tell(log_post(s.ask()))

# --- results ----------------------------------------------------------
d = s.draws()                      # array: 4 chains x 20,000 draws x 3 parameters
print(space.names())               # ['alpha', 'beta', 'sigma']
print(diagnostics.rhat(d))         # should all be close to 1.00

kept = d[:, 5_000:, :]             # drop the first 5,000 draws of each chain
print(kept.reshape(-1, 3).mean(axis=0))   # posterior means: ~ [1.0, 2.0, 0.5]
print(s.stats()["acceptance_rate"])       # fraction of proposals accepted, per chain
```

The loop body really is one line: ask, evaluate, tell.
Everything before the loop runs once; everything after it is reading out results.

---

### 4.2 R

Same model, same numbers, same structure.
If you have used R for data analysis you have everything needed.
The one non-obvious line is explained in its comment: `outer(p$beta, x)` builds the 4×10 table whose (c, i) entry is `beta[c] * x[i]` — every chain's predictions at every data point at once.

```r
library(bayesrs)

# --- data -------------------------------------------------------------
x <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
y <- c(1.1, 1.8, 3.2, 4.1, 4.9, 6.2, 7.1, 7.9, 9.2, 10.1)
n <- length(y)

# --- the parameter space ---------------------------------------------
space <- param_space(
  alpha = real(),
  beta  = real(),
  sigma = bounded(0, Inf)
)

# --- the log-posterior ------------------------------------------------
# theta is a matrix: one row per chain, columns alpha, beta, sigma.
# unpack() gives a named list; p$alpha etc. each have one entry per chain.
log_post <- function(theta) {
  p <- unpack(space, theta)
  pred  <- p$alpha + outer(p$beta, x)      # 4 x 10 table of fitted values
  resid <- sweep(pred, 2, y)               # subtract y from each column: pred - y
  log_lik   <- -n * log(p$sigma) - rowSums(resid^2) / (2 * p$sigma^2)
  log_prior <- -p$alpha^2/200 - p$beta^2/200 - p$sigma^2/50
  log_lik + log_prior
}
# Note: no check that sigma > 0 — the library guarantees it (Section 3).

# --- starting points: one value per chain, spread out ----------------
x0 <- pack(space,
  alpha = c( 0.0, 1.0, -1.0, 2.0),
  beta  = c( 1.0, 2.0,  0.0, 3.0),
  sigma = c( 1.0, 0.5,  2.0, 1.5)
)

# --- create the sampler and run the loop -----------------------------
s <- random_walk_metropolis(space, x0, seed = 42)

while (n_draws(s) < 20000) {
  tell(s, log_post(ask(s)))
}

# --- results ----------------------------------------------------------
d <- draws(s)              # array: 20,000 draws x 4 chains x 3 parameters,
                           # named, and ready for the 'posterior' package
rhat(d)                    # should all be close to 1.00

kept <- d[5001:20000, , ]                 # drop the first 5,000 draws
apply(kept, 3, mean)                      # posterior means: ~ c(1.0, 2.0, 0.5)
```

The draws array is laid out the way R's MCMC ecosystem (the `posterior` and `bayesplot` packages) expects, so `posterior::as_draws_array(d)` hands you straight to those tools for richer summaries and plots.

---

### 4.3 C

C is the low-level option — it is what you would use to drive bayesrs from a large simulation code, or as the bridge from another language entirely.
It has no vectors or named lists built in, so the program is longer and more manual, but the shape is identical: build the space, provide starting values, loop ask/evaluate/tell, read out the draws.

Three pieces of C background, in case it is new to you:

- `const double *theta` declares a *pointer* — the memory address where a sequence of numbers begins.
  `theta[k]` reads the k-th number from that address.
  `ask` gives us a pointer to the batch rather than a copy of it.
- Every bayesrs function that can fail returns a status code; `BRS_OK` means success, and `brs_last_error()` retrieves a human-readable message.
- Memory you allocate with `malloc` you must later release with `free`, and every bayesrs object gets released with its matching `_free` function.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bayesrs.h"

/* --- data ---------------------------------------------------------- */
#define N_DATA   10
#define N_CHAINS 4
#define N_PARAMS 3                       /* alpha, beta, sigma           */

static const double xs[N_DATA] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
static const double ys[N_DATA] = {1.1, 1.8, 3.2, 4.1, 4.9, 6.2, 7.1, 7.9, 9.2, 10.1};

/* --- the log-posterior for ONE row (one chain's proposal) ---------- */
/* t points at 3 numbers, in declaration order: alpha, beta, sigma.    */
static double log_post(const double *t) {
    double alpha = t[0], beta = t[1], sigma = t[2];
    double ll = -N_DATA * log(sigma);   /* sigma > 0 is guaranteed      */
    for (int i = 0; i < N_DATA; i++) {
        double r = ys[i] - (alpha + beta * xs[i]);
        ll -= r * r / (2.0 * sigma * sigma);
    }
    return ll - alpha*alpha/200.0 - beta*beta/200.0 - sigma*sigma/50.0;
}

int main(void) {
    /* --- the parameter space, declared one block at a time --------- */
    brs_space *sp = brs_space_new();
    brs_space_add_real   (sp, "alpha", NULL, 0);          /* NULL,0 = scalar */
    brs_space_add_real   (sp, "beta",  NULL, 0);
    brs_space_add_bounded(sp, "sigma", NULL, 0, 0.0, INFINITY);

    /* --- starting points: one row of (alpha, beta, sigma) per chain */
    double x0[N_CHAINS * N_PARAMS] = {
         0.0, 1.0, 1.0,
         1.0, 2.0, 0.5,
        -1.0, 0.0, 2.0,
         2.0, 3.0, 1.5,
    };

    /* --- create the sampler ---------------------------------------- */
    brs_sampler *s = NULL;
    if (brs_random_walk_metropolis_new(sp, x0, N_CHAINS, 42, &s) != BRS_OK) {
        fprintf(stderr, "error: %s\n", brs_last_error());
        return 1;
    }

    /* --- the loop --------------------------------------------------- */
    const double *theta;                 /* points at the current batch  */
    size_t m, d;                         /* batch rows, columns per row  */
    double logp[N_CHAINS];

    while (brs_n_draws(s) < 20000) {
        brs_ask(s, &theta, &m, &d);      /* here m = 4 rows, d = 3 cols  */
        for (size_t c = 0; c < m; c++)
            logp[c] = log_post(&theta[c * d]);   /* &theta[c*d] = row c  */
        if (brs_tell(s, logp, m) != BRS_OK) {
            fprintf(stderr, "error: %s\n", brs_last_error());
            return 1;
        }
    }

    /* --- copy the draws out ----------------------------------------- */
    /* Layout: chain by chain, each chain draw by draw, each draw is    */
    /* 3 numbers. So draw i of chain c starts at (c*n_draws + i) * 3.   */
    size_t n_draws = brs_n_draws(s);
    size_t len = (size_t)N_CHAINS * n_draws * N_PARAMS;
    double *out = malloc(len * sizeof *out);
    brs_draws(s, out, len);

    /* --- posterior means, discarding the first 5,000 of each chain -- */
    double mean[N_PARAMS] = {0, 0, 0};
    size_t kept = 0;
    for (size_t c = 0; c < N_CHAINS; c++)
        for (size_t i = 5000; i < n_draws; i++) {
            const double *draw = &out[(c * n_draws + i) * N_PARAMS];
            for (int k = 0; k < N_PARAMS; k++) mean[k] += draw[k];
            kept++;
        }
    printf("posterior means: alpha %.3f  beta %.3f  sigma %.3f\n",
           mean[0]/kept, mean[1]/kept, mean[2]/kept);   /* ~ 1.0, 2.0, 0.5 */

    /* --- tidy up ----------------------------------------------------- */
    free(out);
    brs_sampler_free(s);
    brs_space_free(sp);
    return 0;
}
```

To compile and run, assuming the library is installed:

```
cc regression.c -lbayesrs -lm -o regression
./regression
```

(`cc` is the C compiler; `-lbayesrs -lm` link in the bayesrs library and the maths library; `-o regression` names the resulting program.)

---

### 4.4 Rust

Python, R, and C each pay a small — usually negligible — cost every time your loop talks to the library.
There is one way to pay exactly nothing: use the library from Rust, the language it is written in.
Your loop and the sampler then compile together into a single program, with no boundary left to cross.
This is the right choice if you are already writing a simulation in Rust, or if your log-posterior is so cheap (microseconds) and your run so long (tens of millions of iterations) that even a small per-iteration cost would add up.

Some background if Rust is new to you:

- Rust is installed with `rustup` (from rustup.rs).
  Its build tool, **cargo**, does everything: `cargo new regression` creates a project, two lines in the `Cargo.toml` file declare what the project depends on, and `cargo run --release` downloads dependencies, compiles with optimisations, and runs.
- `let` introduces a variable; `let mut`, one that is allowed to change.
  The sampler is `mut` because every `ask`/`tell` changes its internal state.
- The `?` after a call means "if this failed, stop and pass the error along" — Rust's tidy version of the status codes we checked by hand in C.
- Arrays come from the `ndarray` crate, Rust's equivalent of NumPy.
  `ask()` returns a view of the batch, and `.rows()` walks it row by row.

```toml
# Cargo.toml
[dependencies]
bayesrs = "0.1"
ndarray = "0.16"
```

```rust
// src/main.rs
use bayesrs::{diagnostics, ParamSpace, RandomWalkMetropolis, Sampler, Tell};
use ndarray::{s, Axis};

// --- data --------------------------------------------------------------
const XS: [f64; 10] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
const YS: [f64; 10] = [1.1, 1.8, 3.2, 4.1, 4.9, 6.2, 7.1, 7.9, 9.2, 10.1];

/// The log-posterior for one row: t holds (alpha, beta, sigma).
fn log_post(t: &[f64]) -> f64 {
    let (alpha, beta, sigma) = (t[0], t[1], t[2]);
    let mut ll = -(XS.len() as f64) * sigma.ln(); // sigma > 0 is guaranteed
    for (x, y) in XS.iter().zip(YS.iter()) {
        let r = y - (alpha + beta * x);
        ll -= r * r / (2.0 * sigma * sigma);
    }
    ll - alpha * alpha / 200.0 - beta * beta / 200.0 - sigma * sigma / 50.0
}

fn main() -> Result<(), bayesrs::Error> {
    // --- the parameter space -------------------------------------------
    let space = ParamSpace::builder()
        .real("alpha")
        .real("beta")
        .bounded("sigma", 0.0, f64::INFINITY)
        .build()?;

    // --- starting points: one row of (alpha, beta, sigma) per chain ----
    let x0 = [
         0.0, 1.0, 1.0,
         1.0, 2.0, 0.5,
        -1.0, 0.0, 2.0,
         2.0, 3.0, 1.5,
    ];

    // --- create the sampler and run the loop ---------------------------
    let mut sampler = RandomWalkMetropolis::new(&space, &x0, 4, 42)?;

    while sampler.n_draws() < 20_000 {
        let theta = sampler.ask()?;              // a 4 x 3 view of the batch
        let logp: Vec<f64> = theta
            .rows()
            .into_iter()
            .map(|row| log_post(row.as_slice().unwrap()))
            .collect();
        sampler.tell(Tell::new(&logp))?;
    }

    // --- results --------------------------------------------------------
    let d = sampler.draws();                     // 4 chains x 20,000 x 3
    println!("rhat: {}", diagnostics::rhat(&d)); // should all be near 1.00

    let kept = d.slice(s![.., 5_000.., ..]);     // drop the first 5,000
    let means = kept
        .mean_axis(Axis(0)).unwrap()             // average over chains...
        .mean_axis(Axis(0)).unwrap();            // ...then over draws
    println!("posterior means: {}", means);      // ~ [1.0, 2.0, 0.5]
    Ok(())
}
```

One pleasing detail.
In the other languages, calling `ask` and `tell` in the wrong order gives a clear error message *when the program runs*.
In Rust the discipline is enforced by the compiler: `theta` borrows the batch from the sampler, and a program that tries to keep using that batch after `tell` has moved the sampler on will simply refuse to compile.
The rule "evaluate what you were given, then hand the results back" is checked before the program ever runs.

---

## 5. A discrete parameter: is the signal there?

Everything so far had continuous parameters only.
This section adds one discrete unknown, using the model from `signal_detection_demo.ipynb`: a detector records 60 noisy time points, and a pulse of **known** shape may or may not be present,

$$y_t = \mu + z\,s_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathrm{N}(0, \sigma^2),$$

with unknown baseline $\mu \sim \mathrm{N}(0, 5^2)$, noise level $\sigma \sim \text{Half-Normal}(2)$, and $z \in \{0, 1\}$ — pulse absent or present — with a 50:50 prior.
The posterior probability that $z = 1$ *is* the detection probability, as direct as a posterior quantity gets.

A discrete parameter cannot share a sampler with continuous ones: a Gaussian step lands on 0.37, which is not a state.
(For the same reason, gradient-based samplers like NUTS are meaningless for it — there is no slope between categories.)
So bayesrs makes you split the parameters into **blocks**, each owned by a kernel that suits it, and composes them with `Gibbs`: each draw, the continuous block takes its step with $z$ frozen, then $z$ takes its step with the continuous values frozen.
You write **one** log-posterior over the full parameter set, exactly as before, and your loop does not change at all — each draw simply takes two trips through it (one per block) instead of one.

```python
import numpy as np
from bayesrs import (ParamSpace, Real, Bounded, Categorical,
                     RandomWalkMetropolis, DiscreteMetropolis, Gibbs, Block)

# --- the known pulse shape, and one synthetic recording (pulse present) --
t = np.arange(60)
pulse = 0.5 * np.exp(-(t - 30.0)**2 / (2 * 5.0**2))
rng = np.random.default_rng(1)
y = 1.0 + pulse + rng.normal(0.0, 1.0, 60)        # mu=1, sigma=1, z=1

# --- the parameter space -------------------------------------------------
space = ParamSpace(
    mu    = Real(),
    sigma = Bounded(0, None),
    z     = Categorical(2, labels=["absent", "present"]),
)

# --- one log-posterior over ALL parameters, discrete included -----------
# p.z arrives as exactly 0.0 or 1.0, so it can be used in arithmetic
# directly. The flat prior on z adds a constant, so it does not appear.
def log_post(theta):
    p = space.unpack(theta)
    mean  = p.mu[:, None] + p.z[:, None] * pulse           # 4 x 60 table
    ll    = -60 * np.log(p.sigma) - ((y - mean)**2).sum(axis=1) / (2 * p.sigma**2)
    return ll - p.mu**2 / 50 - p.sigma**2 / 8

# --- starting points, spread across both hypotheses ---------------------
x0 = space.pack(
    mu    = np.array([0.0,  1.0, -1.0, 2.0]),
    sigma = np.array([1.0,  2.0,  0.5, 1.5]),
    z     = np.array([0.0,  1.0,  0.0, 1.0]),
)

# --- two blocks, two kernels, one sampler -------------------------------
s = Gibbs(space, x0, seed=42, blocks=[
    Block(["mu", "sigma"], RandomWalkMetropolis),
    Block(["z"],           DiscreteMetropolis),
])

while s.n_draws < 20_000:          # the same loop as every other example
    s.tell(log_post(s.ask()))

# --- results -------------------------------------------------------------
p = space.unpack(s.draws())
print("P(pulse present | data):", p.z[:, 5_000:].mean())
print(s.stats()["blocks"][1]["acceptance_rate"])   # the discrete block's own rate
```

What `DiscreteMetropolis` does is ordinary Metropolis with a proposal suited to states rather than steps: it proposes one of the *other* states uniformly (for two states, simply the flip) and accepts or rejects by the usual rule.
The chains for $\mu$ and $z$ are coupled in an intuitive way — during stretches where the chain believes $z = 0$, the baseline $\mu$ shifts up to absorb the bump — and the split of the draws by `p.z` shows it.

Three things worth knowing:

- **The same loop, more trips.**
  "Count draws, not iterations" (Section 2) is doing real work here: a draw is one full sweep over the blocks, so the loop goes round twice per draw.
  Your code never notices.
- **If you *can* sum the discrete parameter out of your likelihood, do.**
  For this model $p(y \mid \mu, \sigma) = \frac12 p(y \mid \mu, \sigma, z{=}0)
  + \frac12 p(y \mid \mu, \sigma, z{=}1)$ is two evaluations and an average — and then every parameter is continuous, any sampler works, and mixing is typically better.
    Sample a discrete parameter when marginalising is genuinely unavailable (many states, or states that change the likelihood's structure), not by default.
- **Blocks are not just for discrete parameters.**
  The same mechanism handles any model where different parameters want different kernels — when gradient-based samplers arrive, `Gibbs` will run NUTS on the smooth block and `DiscreteMetropolis` on the discrete one, with this exact loop.

The R, C, and Rust versions follow the same shape as their Section 4 counterparts: declare a categorical block, list the blocks, keep the loop.

---

## 6. Reading the results

**`draws()`** returns every stored draw of every chain, always as the actual parameters (a real `sigma`, never its logarithm).
In Python the array is arranged (chain, draw, parameter); in R, (draw, chain, parameter) — each matching what that language's ecosystem expects.
Columns are in declaration order, and `space.names()` (Python) / `names(space)` (R) lists them.

**R̂ ("R-hat")** compares variation within chains to variation between them.
Values near 1.00 for every parameter say the chains agree about where they've converged to; values above about 1.01 mean run longer or look for problems.
**Effective sample size** (`ess_bulk`, `ess_tail`) estimates how many *independent* draws your correlated draws are worth.
Both are built in, using the modern rank-normalised definitions.

**Acceptance rate** (from `stats()`) is the fraction of proposals accepted.
For random-walk samplers, somewhere between roughly 0.2 and 0.5 is healthy; near 0 means steps are too large, near 1 means too small.

**Burn-in** is your call: draws are stored from the very start (draw 0 is your starting point), so discard an initial stretch, as the examples do.

---

## 7. Common questions

**How fast is the ask/tell round trip?**
Crossing between your language and the library costs well under a microsecond per iteration — for any realistic log-posterior, a negligible fraction of the total.
The design rule of thumb: your posterior evaluation is the cost of your MCMC; the library is free.
And from Rust (Section 4.4) the cost is literally zero: there is no boundary, because your loop and the sampler compile into one program.

**Why do all the examples run 4 chains?**
You want multiple chains anyway for R̂ — and because every ask/tell round carries all chains at once, the fixed cost of the round trip is shared between them.
Batch evaluation (as in the Python and R examples) often makes 4 chains barely more expensive than 1.

**Can I keep a proposal for later?**
In Python and R, yes — the batch `ask()` returns is an ordinary array that belongs to you; the library keeps no strings attached.
In C the batch pointer is only valid until the next `ask`, so copy anything you want to keep.
In Rust the batch is a borrowed view and the compiler will insist you copy it (`theta.to_owned()`) before moving on — same rule as C, but enforced automatically.

**Is it reproducible?**
Every random number comes from the library's own generator, controlled entirely by `seed`.
Same seed, same model, same data ⇒ the same chains, every run — and the design goal is bit-for-bit identical chains across Python, R, and C when the log-posterior values match exactly.

**What if my likelihood is expensive?**
Parallelise it yourself — you own the loop, so the batch of rows can be farmed out to multiple cores with your language's standard tools before you `tell`.
The library neither knows nor cares how the numbers were computed.

**What's coming later?**
The first release ships random-walk Metropolis, its adaptive variant, the discrete kernel, and the `Gibbs` compositor.
Planned next are gradient-based samplers (MALA, HMC, NUTS — which will slot into `Gibbs` blocks alongside the discrete kernel), ordered integer parameters, more matrix types, and checkpointing.
The interface in this guide is designed so those arrive without changing any code you write today.

---

## Glossary

**Array / matrix** — a block of numbers arranged in rows and columns.
The ask/tell exchange is always one of these: rows are chains, columns are parameter values in declaration order.

**Batch** — all chains' proposals delivered together in one `ask()`.

**Block** — a named subset of the parameters owned by one kernel inside `Gibbs`; each draw updates every block in turn (one **sweep**).

**Burn-in** — the initial stretch of a chain, discarded because it reflects the starting point rather than the posterior.

**Draw** — one stored sample of the full parameter vector, per chain.

**Jacobian correction** — the adjustment a probability density needs when you change variables.
Applied internally by the library; you never write one.

**Log-posterior** — the logarithm of prior × likelihood, up to an additive constant.
The one function you write.

**Parameter space (schema)** — your declaration of parameter names and kinds, made once before sampling.

**Seed** — the number that determines every random choice the sampler makes; fixing it makes runs exactly repeatable.
