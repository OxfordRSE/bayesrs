# bayesrs v1 interface specification

This is the normative interface for the first release; `DECISIONS.md` records the
rationale behind it.

Settled here (2026-08-11): verbs are **`ask`/`tell`**; proposals are **copied out**
(no live views), with named access via **`space.unpack()`**; v1 ships
**RandomWalkMetropolis** and **AdaptiveMetropolis**; discrete parameters are
**deferred to v2** with their wire format reserved.

---

## 1. Scope of v1

**In:**

- Schema (`ParamSpace`) with kinds `Real`, `Bounded`, `Simplex`, `CovMatrix`
- Samplers: `RandomWalkMetropolis`, `AdaptiveMetropolis` (Haario)
- The `ask`/`tell` protocol, draw storage, per-chain acceptance statistics
- Diagnostics: rank-normalised split-R̂, bulk ESS, tail ESS (Vehtari et al. 2021)
- Bindings: Python (PyO3/maturin), C (cbindgen header), R (extendr)
- Cross-language bit-reproducibility of chains given identical `tell` values

**Out, with names/wire format reserved (§10):** discrete kinds (`Integer`,
`Categorical`, `Binary`), `CorrMatrix`, `Cholesky`, `Custom`, gradient-based samplers
(MALA/HMC/NUTS), DE-MCMC, parallel tempering, SMC, `run_compiled`, live views,
checkpointing.

---

## 2. The core protocol

A sampler is a state machine with two states:

```
             ask() ──────────────►
  READY                              WAITING_FOR_TELL
             ◄────────── tell(...)
```

- `ask()` in `READY` → returns an `(m, d_constr)` array of points in **constrained
  space** whose log densities the caller must evaluate; sampler moves to
  `WAITING_FOR_TELL`.
- `tell(logp)` in `WAITING_FOR_TELL` → consumes exactly `m` log-density values
  (ordered to match the rows of the last `ask`); sampler moves to `READY`.
- `ask()` in `WAITING_FOR_TELL` is an **error**: *"ask() called again before tell();
  tell() the results of the previous ask() first."*
- `tell()` in `READY` is an **error**: *"tell() called with no outstanding ask()."*

**Draw counting, not iteration counting.** Callers must not assume one
`ask`/`tell` exchange produces one draw. The contract is: draws accumulate as the
sampler completes them, and `n_draws` reports the count of stored draws per chain.
The canonical loop is:

```python
while s.n_draws < N:
    theta = s.ask()          # (m, d_constr)
    s.tell(logp(theta))
```

For both v1 samplers, `m == n_chains` and every `tell` after the first completes
exactly one draw per chain — so a plain `for` loop also works today. The
`while`-loop form is what the documentation teaches, because future samplers
(NUTS tree-building, within-Gibbs sweeps) will take several exchanges per draw
and may return `m != n_chains`.

**Initialisation is the first exchange.** The constructor takes starting positions
`x0`; the sampler never invents them (it does not know the prior). The **first**
`ask()` returns `x0` itself, and the first `tell` records those densities without
an accept/reject step and stores `x0` as draw 0 (`n_draws` becomes 1). Every
subsequent `tell` performs the Metropolis step. The user's loop is therefore
identical from iteration zero.

**`tell` payload.** Each sampler declares which fields it requires. Both v1
samplers require exactly `logp`, shape `(m,)`, float64. The field names `grad`,
`log_lik`, `log_prior`, `metric` are reserved (§10), and future samplers may
require different combinations (NUTS: `logp` + `grad`; parallel tempering:
`log_lik` + `log_prior`, with `logp` derived) — `logp` is required by every v1
sampler, not by the protocol itself. Passing a field the sampler does not use
is an **error** naming the sampler and the field — it means the caller is
computing something that is being thrown away.

**Return value of `tell`:** nothing. Monitoring goes through `stats()` (§6).

---

## 3. Special values and validation

| Value in `tell(logp)` | Meaning | Behaviour |
|---|---|---|
| finite | ordinary density | accept/reject as usual |
| `-inf` | proposal outside support | always legal; reject and move on |
| `-inf` on the **first** tell | chain starts outside support | **error**, naming the chain index — the chain has nowhere to stand |
| `+inf` | broken likelihood | **error**, naming the chain index |
| `NaN` | broken likelihood | **error**, naming the chain index |
| wrong length / dtype not castable to float64 | caller bug | **error** stating expected shape |

Errors from `tell` leave the sampler in `WAITING_FOR_TELL` with its chain state
untouched: the caller can fix the problem and `tell` again.

`x0` is validated at construction against every block's constraints (bounds
**strictly** respected — a boundary value has no free-space image, see §4.2 —
simplexes sum to 1 within tolerance with strictly positive entries, covariance
blocks symmetric positive-definite), with errors naming the block and chain.

---

## 4. Schema: `ParamSpace`

### 4.1 Construction

Names map to kinds; each kind carries its own shape. Declaration order fixes the
column order of every array in the system.

```python
from bayesrs import ParamSpace, Real, Bounded, Simplex, CovMatrix

space = ParamSpace(
    mu      = Real(shape=3),        # shape=() scalar default; any shape allowed
    sigma   = Bounded(0, None),     # scalar; either bound may be None/inf
    Sigma   = CovMatrix(3),         # 3×3 symmetric positive-definite
    weights = Simplex(4),           # 4 non-negative entries summing to 1
)
space.d_free     # 3 + 1 + 6 + 3 = 13
space.d_constr   # 3 + 1 + 9 + 4 = 17
```

`ParamSpace` also accepts an ordered mapping of name → kind
(`ParamSpace({"mu": Real(shape=3), ...})`). The keyword form is sugar; because
its keywords are all parameter names, any future constructor *option* goes on
the mapping form (or a classmethod) — options never compete with parameter
names for keyword slots.

Names must be valid identifiers in the host language (they become attributes /
list names). Shape rules: `Real` and `Bounded` take any shape (constraints
elementwise); `Simplex(k)` is a length-`k` vector; `CovMatrix(n)` is `n × n`.
Matrix-valued blocks are flattened row-major into their columns of the
constrained vector.

### 4.2 Transforms (frozen, Stan-compatible)

Each kind defines the free↔constrained map and its log-Jacobian. These follow the
Stan Reference Manual's constraint transforms exactly, so they are documented,
battle-tested, and cross-checkable:

| Kind | d_free | Transform |
|---|---|---|
| `Real` | = d_constr | identity |
| `Bounded(lo, ∞)` | = d_constr | `x = lo + exp(z)` |
| `Bounded(−∞, hi)` | = d_constr | `x = hi − exp(z)` |
| `Bounded(lo, hi)` | = d_constr | scaled logistic |
| `Simplex(k)` | k − 1 | stick-breaking |
| `CovMatrix(n)` | n(n+1)/2 | Cholesky factor, log-transformed diagonal |

**Boundary semantics (settled).** `Bounded(lo, hi)` and `Simplex` intervals
are **open**: endpoints carry no probability mass for a continuous parameter,
so no open/closed API distinction exists or is needed (mass *at* a boundary is
a mixed distribution — a different model, out of scope). The transforms are
open-interval by construction, but in floating point the endpoints are
reachable by rounding (`exp(z)` underflows; `sigmoid(z)` rounds to 0 or 1 for
`|z| ≳ 37`). Therefore `to_constrained` **clamps boundary-rounded values to
the nearest representable interior value** (one `nextafter`), so user-visible
values satisfy their strict inequalities unconditionally — the "no guards
needed" guarantee has no floating-point asterisk. Symmetrically, `x0`
validation and `to_free` use strict inequalities: a value exactly on a
boundary is invalid input (the free-space image does not exist).

**Jacobian convention (settled).** The caller's `logp` is the log density of the
**constrained** parameters with respect to Lebesgue measure on the constrained
space — the density you would write on paper. The library adds
`log |det J(free → constrained)|` internally when it compares densities in free
space. The caller never sees or applies a Jacobian.

For testing and debugging, the transforms are public pure functions:

```python
space.to_free(theta_constr)              # (..., d_constr) → (..., d_free); errors if invalid
space.to_constrained(z_free)             # (..., d_free) → (..., d_constr)
space.log_abs_det_jacobian(z_free)       # (..., d_free) → (...)
```

These three signatures are **final**, including under future discrete kinds:
discrete columns will ride through the free vector unchanged (identity map,
zero log-Jacobian, counted in `d_free`), so the free vector always determines
the constrained vector completely. Samplers learn which free columns are
discrete from the schema, not from the wire format. (See §10.)

### 4.3 `pack` / `unpack`

Naming lives on the schema, not the sampler. `unpack` wraps a constrained array
in named, correctly-shaped **views into that array** (zero-copy, no lifetime
hazard — the caller owns the array). `pack` is its inverse, and is how `x0` is
built ergonomically.

```python
theta = s.ask()                     # (m, 17), caller-owned
p = space.unpack(theta)             # p.mu (m,3) · p.sigma (m,) · p.Sigma (m,3,3) · p.weights (m,4)

x0 = space.pack(                    # broadcasting: scalars/single values repeat across chains
    mu      = rng.normal(size=(8, 3)),
    sigma   = np.full(8, 1.0),
    Sigma   = np.broadcast_to(np.eye(3), (8, 3, 3)),
    weights = np.full((8, 4), 0.25),
)                                   # → (8, 17)
```

Both accept arbitrary leading batch dimensions: `space.unpack(s.draws())` gives
`p.mu` with shape `(n_chains, n_draws, 3)`.

The unpacked object is also a mapping (`p["mu"]`), and every non-parameter
attribute of it is underscore-prefixed, so parameter names can never collide
with the object's own API as it grows. For the same reason the design note's
`p.n_comp_i` suffix convention is dropped: integer access for future discrete
kinds will be a method (e.g. `p.as_int("n_comp")`), never a name suffix that a
parameter could legally claim today.

`space.names()` returns the flattened per-column names in declaration order
(`"mu[0]", …, "Sigma[0,0]", …`); each binding uses its language's index base
(0-based Python/C, 1-based R) but the **order** is identical everywhere.

---

## 5. Samplers

Both are constructed from `(space, x0, seed)`; `n_chains` is inferred from
`x0.shape[0]`. Tuning parameter *names* below are frozen; their default *values*
are initial choices from the literature, tunable until 1.0.

A sampler validates at construction that it supports every kind in the space,
and errors otherwise naming the sampler and the block. (Vacuous in v1 — both
samplers support all four kinds — but the rule is what makes v2 kinds safe:
when `Integer` arrives, existing samplers reject it loudly at construction
rather than proposing Gaussian steps on it.)

```python
RandomWalkMetropolis(space, x0, *, seed,
    proposal_cov=None)     # free-space (d_free, d_free); default (2.38²/d_free)·I

AdaptiveMetropolis(space, x0, *, seed,   # Haario et al. (2001), diminishing adaptation
    initial_cov=None,      # free-space proposal cov before adaptation starts; default as RWM
    adapt_start=100,       # draws before the empirical covariance takes over
    epsilon=1e-6,          # regularisation: proposal = sd·(emp_cov + ε·I)
    sd=None)               # scale factor; default 2.38²/d_free
```

Proposal covariances are in **free space** — that is where proposing happens, and
it is documented as such. Adaptation uses diminishing step sizes and runs for the
whole chain (ergodic; no separate warmup phase to configure in v1).

**RNG (frozen for bit-reproducibility).** All randomness comes from **ChaCha8**
streams derived deterministically from `(seed: u64, stream_id: u64)` via a
documented SplitMix64 expansion. Chains use stream ids `0..n_chains`; higher
ids are reserved for future sampler-level randomness (cross-chain moves,
tempering swaps), so adding such features never perturbs the per-chain streams.
No binding ever touches its host language's RNG. The derivation is part of the
interface: same seed, same schema, same `tell` values ⇒ bit-identical chains in
Python, R, and C.

---

## 6. Outputs

```python
s.n_draws          # int: stored draws per chain (draw 0 is x0)
s.n_chains         # int
s.draws()          # fresh float64 array, (n_chains, n_draws, d_constr), constrained space
s.position()       # (n_chains, d_constr): current state, for inspection or restarts
s.stats()          # {"n_draws": int, "acceptance_rate": (n_chains,), ...}
s.proposal_cov()   # (d_free, d_free) copy — AdaptiveMetropolis only
```

`stats()` may gain keys in later versions; callers must ignore keys they do not
recognise. The `n_draws` invariant is: `n_draws` always equals the number of
draws `draws()` returns — a future sampler with a configurable warmup phase
counts only stored post-warmup draws. **Definition for asynchronous samplers
(settled now):** a future sampler may complete draws at different rates per
chain (NUTS trees terminate independently, and the efficient schedule lets
each chain start its next trajectory immediately rather than idle). `n_draws`
is then the number of complete draws available from **every** chain — the
minimum across chains — so `while n_draws < N` still means "until every chain
has N draws", and `draws()` returns exactly that rectangle. A chain's surplus
draws beyond the minimum are retained, not deleted: they enter the rectangle
as slower chains catch up, and are only left unsurfaced if sampling stops
first. (Whether such a sampler caps how far a chain may run ahead — pausing
it, shrinking `m` — is internal scheduling policy, free to tune.) For v1
samplers chains advance in lockstep and this reduces to the obvious count.

Internal chain storage is in **free space** (smaller: `d_free` ≤ `d_constr`);
`draws()` maps to constrained space on extraction. `thin=k` (constructor keyword,
default 1, on both samplers) stores every k-th draw; `n_draws` counts *stored*
draws; draw 0 is always stored.

Memory rule of thumb documented prominently: `n_chains × n_draws × d_free × 8`
bytes — 8 chains × 1M draws × 13 free dims ≈ 0.8 GB. `thin` is the lever.
(`draws()` may be called mid-run at any time — checking diagnostics every N
draws is an intended workflow; a reserved `draws(since=k)` makes that cheap
for very long runs, see §10.)

**Diagnostics** (implemented in the core, exposed in every binding):

```python
bayesrs.diagnostics.rhat(draws)       # rank-normalised split-R̂  → (d_constr,)
bayesrs.diagnostics.ess_bulk(draws)   # → (d_constr,)
bayesrs.diagnostics.ess_tail(draws)   # → (d_constr,)
```

These take `(n_chains, n_draws, d)` (or `(n_chains, n_draws)` for one
dimension). Python users are pointed at ArviZ for everything richer; the draws
array plus `space.names()` is enough to build an `InferenceData`.

---

## 7. Python binding

Package `bayesrs`; abi3 wheels via maturin, CPython ≥ 3.10.

```python
import numpy as np
from bayesrs import ParamSpace, Real, Bounded, AdaptiveMetropolis, diagnostics

data = np.loadtxt("data.csv")

space = ParamSpace(mu=Real(), sigma=Bounded(0, None))

rng = np.random.default_rng(0)                     # only for making x0
x0 = space.pack(mu=rng.normal(size=8), sigma=np.exp(rng.normal(size=8)))

s = AdaptiveMetropolis(space, x0, seed=42)

def log_post(theta):
    p = space.unpack(theta)
    ll = -0.5 * np.sum((data - p.mu[:, None])**2 / p.sigma[:, None]**2, axis=1) \
         - len(data) * np.log(p.sigma)
    return ll - 0.5 * p.mu**2 - np.log(p.sigma)    # log-normal prior on sigma

while s.n_draws < 50_000:
    s.tell(log_post(s.ask()))

draws = s.draws()                                  # (8, 50000, 2)
print(diagnostics.rhat(draws))                     # column order = space.names()
```

- `ask()` returns a **fresh, caller-owned, C-contiguous** float64 array. Nothing
  the caller does to it affects the sampler.
- `tell` accepts anything castable to a float64 `(m,)` array.
- A three-line pure-Python convenience `bayesrs.run(sampler, log_post, n_draws)`
  wraps the canonical loop; it is sugar, not a separate code path.

---

## 8. C binding

Second to be built, immediately after Python — it disciplines the core. Header
generated by cbindgen; opaque handles; caller allocates what caller frees;
every fallible function returns a status, message via thread-local
`brs_last_error()`.

```c
#include "bayesrs.h"

brs_space *sp = brs_space_new();
brs_space_add_real(sp, "mu", NULL, 0);                    /* scalar */
brs_space_add_bounded(sp, "sigma", NULL, 0, 0.0, INFINITY);

double x0[8 * 2] = { /* row per chain, constrained space */ };
brs_sampler *s = NULL;
if (brs_adaptive_metropolis_new(sp, x0, 8, 42, &s) != BRS_OK) {
    fprintf(stderr, "%s\n", brs_last_error());
    return 1;
}

const double *theta;  size_t m, d;
double logp[8];
while (brs_n_draws(s) < 50000) {
    brs_ask(s, &theta, &m, &d);              /* theta: library-owned, row-major,   */
    for (size_t c = 0; c < m; c++)           /* valid only until the next brs_ask  */
        logp[c] = my_log_post(&theta[c * d], d);
    if (brs_tell(s, logp, m) != BRS_OK) { /* handle */ }
}

size_t len = brs_n_chains(s) * brs_n_draws(s) * brs_space_d_constr(sp);
double *out = malloc(len * sizeof *out);
brs_draws(s, out, len);                      /* row-major (chain, draw, dim) */
```

C is the one binding where `ask` exposes the library-owned buffer directly
(valid until the next `brs_ask`) — C callers manage lifetimes anyway, and it
keeps the C path allocation-free. `brs_space_offset(sp, "Sigma", &off, &len)`
is `unpack` for C: compute offsets once, index forever.

Forward compatibility: future samplers may return a different `m` on every
call, so portable callers size buffers from the returned `m`, not from
`n_chains` (the fixed `logp[8]` above is valid for v1 samplers only). The C ABI
evolves only by **adding** functions — e.g. `brs_tell_grad(s, logp, grad, m)`
when gradient samplers arrive — never by changing existing signatures.

Also in the header: `brs_space_d_free`, `brs_space_names` (flattened, caller
iterates), `brs_stats` (acceptance counts), `brs_space_free`, `brs_sampler_free`,
`brs_random_walk_metropolis_new`, and setter-style tuning options
(`brs_am_set_adapt_start`, …) called between `new` and the first `ask`.

---

## 9. R binding

Built on extendr, released via r-universe first, CRAN once the interface has
been stable for a cycle.

```r
library(bayesrs)

space <- param_space(mu = real(), sigma = bounded(0, Inf))
x0 <- pack(space, mu = rnorm(8), sigma = exp(rnorm(8)))   # 8 × d_constr matrix

s <- adaptive_metropolis(space, x0, seed = 42)

log_post <- function(theta) {
  p <- unpack(space, theta)          # named list: p$mu (8), p$sigma (8)
  ...
}

while (n_draws(s) < 50000) tell(s, log_post(ask(s)))

d <- draws(s)                        # n_draws × n_chains × d_constr
rhat(d)
```

Settled R-specific decisions:

- Verbs `ask`/`tell` — no collision with any base/stats generic.
- Per-exchange matrices (`ask`, `x0`) are **`n_chains × d_constr`**, row per
  chain, matching Python semantically.
- `draws()` returns **`(draw, chain, dim)`** with dimnames from `names(space)` —
  the orientation the `posterior` package's `as_draws_array()` expects, so
  interop is one function call. (The core's native layout is `(chain, draw,
  dim)`; the R binding permutes on extraction, once.)
- 1-based flattened names: `"mu[1]"`, `"Sigma[1,1]"`.

---

## 10. Reserved for later versions

These are *named now* so that adding them breaks nobody:

| Reservation | Where |
|---|---|
| `tell` field `grad`, shape `(m, d_constr)`: the partial derivative of `logp` w.r.t. **each flattened constrained coordinate as delivered by `ask`**, entries treated as independent — exactly what autodiff of the caller's code yields. Symmetric-matrix entries appear twice and each carries its own partial; the transform's Jacobian sums the duplicates correctly when the library computes `∇_free = Jᵀ∇_θ + ∇_free log\|det J\|`. The caller never applies a chain rule. | §2 |
| `tell` fields `log_lik`, `log_prior` (tempering, SMC — may be required *instead of* `logp`) and `metric` (Riemannian; shape pinned when first used) | §2 |
| Kinds `Integer(lo, hi)`, `Categorical(k)`, `Binary()`: whole-valued float64 in the same constrained array, **identity map through the free vector** (zero log-Jacobian, counted in `d_free`) so the §4.2 transform signatures never change; samplers get discreteness from the schema | §4 |
| Kinds `CorrMatrix()`, `Cholesky()`, and transform *combinators* (ordered, offset/scale, composition — Stan-style), all implemented in the core: fast, serialisable, identical in every language | §4 |
| `Custom(...)`: user-supplied `to_constrained` / `log_abs_det_jacobian` / `to_free` (plus a Jacobian-vector product for gradient samplers), stored by the binding and invoked **synchronously and re-entrantly on the caller's thread during the caller's own `ask`/`tell`/`draws` call** — categorically unlike the rejected library-owned-loop callbacks (no foreign threads, exceptions propagate normally, one vectorised call per exchange). Documented costs: the space is binding-local (unusable from other languages), checkpoint restore requires re-supplying the callables, and output validity becomes the user's promise. A compiled-function-pointer variant (Numba `@cfunc` etc.) recovers speed and composes with `run_compiled`. Not a capability gate: declaring the block `Real` and applying the change of variables (with its log-Jacobian) inside your own logp works today in any language | §4 |
| `m != n_chains` and multi-exchange draws (NUTS tree-building, within-Gibbs sweeps). NUTS scheduling: leapfrog steps are sequential *within* a chain, so each exchange carries each active chain's single next evaluation point; chains complete draws asynchronously (ragged), with `n_draws` = min across chains per §6. Steady-state `m == n_chains`; `m` shrinks only when chains pause (e.g. at a draw limit) | §2, §6 |
| `warmup=` constructor keyword on samplers that need a distinct adaptation phase; `n_draws` counts stored draws only | §6 |
| Incremental extraction: `draws(since=k)` returns draws `k..n_draws` only, so a monitor-every-block loop avoids re-copying the whole history (which is quadratic over a long run). Any "since last time I asked" convenience is a caller-held reader object with its own cursor (`s.draw_reader()`, one per consumer) — never hidden state in the sampler, so `draws()` and friends stay idempotent and independent consumers cannot steal each other's draws | §6 |
| `run_compiled(fn_ptr, n)` escape hatch for compiled likelihoods | design note §7.4 |
| Opt-in live views (`params(live=True)`) if profiling ever justifies them | design note §4.1 |
| Checkpoint/restore of full sampler state (serde) — state is designed to be serialisable from day one (ChaCha8 is counter-based) | §6 `position()` covers crude restarts meanwhile |

**Explicit non-goal:** trans-dimensional (reversible-jump) sampling. A fixed
`d_constr` for the lifetime of a sampler is load-bearing for the entire wire
format, and we accept that constraint permanently. Model choice remains
expressible in the standard fixed-dimension way: a `Categorical` model index
plus padded per-model parameter blocks.

---

## 11. Rust core shape

`bayesrs-core` is pure Rust, no binding code, no allocation in the hot path.
It is also itself the fourth public interface: Rust users depend on the crate
directly, with zero boundary cost, so its user-facing API follows the same
contract and stability rules as the bindings.

```rust
pub struct ParamSpace { blocks: Vec<Block>, d_free: usize, d_constr: usize }

impl ParamSpace {                       // Rust-facing construction mirrors the C API:
    pub fn builder() -> ParamSpaceBuilder;
    // ParamSpace::builder().real("alpha").bounded("sigma", 0.0, f64::INFINITY).build()?
}

pub trait Transform {                    // one impl per kind; adding a kind = this + nothing
    fn d_free(&self) -> usize;
    fn d_constr(&self) -> usize;
    fn to_constrained(&self, free: &[f64], out: &mut [f64]);
    fn to_free(&self, constr: &[f64], out: &mut [f64]) -> Result<(), Error>;
    fn log_abs_det_jacobian(&self, free: &[f64]) -> f64;
    // reserved: gradient chain rule
}

pub trait Sampler {
    fn ask(&mut self) -> Result<ArrayView2<'_, f64>, Error>;   // (m, d_constr)
    fn tell(&mut self, t: Tell<'_>) -> Result<(), Error>;
    fn n_draws(&self) -> usize;
    fn draws(&self) -> Array3<f64>;                            // (chain, draw, dim), constrained
}

#[non_exhaustive]                       // fields will be added (metric, …);
pub struct Tell<'a> {                   // constructed via builder, never literally:
    pub logp: Option<&'a [f64]>,        //   Tell::new(logp).with_grad(g)
    pub grad: Option<&'a [f64]>,        // reserved; v1 samplers error if Some
    pub log_lik: Option<&'a [f64]>,     // reserved
    pub log_prior: Option<&'a [f64]>,   // reserved
}
// Error is #[non_exhaustive] too, and sampler tuning options live in
// per-sampler option structs with Default — new knobs are never breaking.
```

Workspace layout:

```
bayesrs/
├── crates/
│   ├── bayesrs-core/     samplers, transforms, diagnostics, RNG streams
│   ├── bayesrs-py/       PyO3 + rust-numpy → maturin wheels
│   └── bayesrs-capi/     #[no_mangle] extern "C" + cbindgen → bayesrs.h
└── r/bayesrs/            extendr package, links bayesrs-core
```

---

## 12. Verification plan

The spec is only credible if these tests exist from the start:

1. **Transform properties** — `to_free ∘ to_constrained = id` (round trip to
   float tolerance); `log_abs_det_jacobian` vs. finite differences; outputs of
   `to_constrained` always satisfy the constraints **strictly, including at
   extreme free values** (`z = ±40, ±1000`, where naive transforms round onto
   the boundary — this pins the §4.2 nextafter clamp).
2. **`pack`/`unpack` round trip** — including batch dimensions and broadcasting.
3. **Protocol errors** — every error path in §2–§3 has a test asserting the
   message names the operation, chain, and/or block.
4. **Statistical correctness** — sample analytically known posteriors
   (multivariate Gaussian; a bounded and a simplex parameter with conjugate
   structure) and check moments/quantiles within Monte-Carlo tolerance;
   acceptance rates in sane ranges.
5. **Cross-language bit-reproducibility (CI)** — one fixed problem whose `logp`
   uses only arithmetic that is bit-identical across languages (e.g.
   `-0.5 * Σ θᵢ²` in a fixed evaluation order), run through Python, C, and R;
   assert the three `draws()` arrays are byte-identical.
6. **Determinism under `thin`** — thinned chain equals unthinned chain
   subsampled.
7. **Protocol-generality conformance** — a mock sampler in the test suite that
   deliberately returns `m != n_chains`, varies `m` between calls, takes
   several exchanges to complete a draw, and completes draws at different
   rates across chains (exercising the min-across-chains `n_draws` rule and
   `draws()` truncation), driven through **every binding's** tests. No shipped v1 sampler exercises the general loop contract, so this
   mock is what keeps the contract executable rather than aspirational — it is
   the pre-paid compatibility test for NUTS and within-Gibbs sweeps.

---

## 13. Remaining open items (non-blocking, decide during implementation)

- Final default tuning constants for AM (`adapt_start`, `epsilon`, diminishing
  schedule) — names frozen, values tunable until 1.0.
- Exact ChaCha8 stream-derivation constants — frozen the day the first
  cross-language test passes, documented in the book.
- Whether `stats()` grows windowed (recent-history) acceptance rates alongside
  cumulative ones.
