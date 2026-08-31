# bayesrs: the design decisions, and the doors left open

This document explains *why* bayesrs is the way it is.
Its companion `GUIDE.md` shows you how to use the library; `INTERFACE.md` pins down every detail precisely for the people building it.
This one is for anyone — statistician, student, reviewer — who wants to understand the reasoning, including what we deliberately left out and where future features will slot in without breaking anything you write today.

A theme runs through everything here: **an interface is easy to add to and almost impossible to change.**
Once people have written code against a library, renaming a function or changing what a number means breaks their work.
So every decision below was made by asking not just "is this good now?" but "will this still be right when the library has grown?" — and every planned extension was checked against the interface *before* the first release, so that adding it later is guaranteed to break nothing.

---

## 1. The shape of the library

### One engine, written once

All the actual MCMC machinery — proposals, accept/reject, adaptation, random numbers, transforms, diagnostics — is implemented exactly once, in Rust, a fast compiled language.
Python, R, and C users all drive that same engine through thin connecting layers.

**Why.**
Correct MCMC code is subtle: adaptation bookkeeping, Jacobian corrections, and random-number handling are all places where quiet mistakes produce plausible-looking wrong answers.
Implementing the machinery once means testing it once, fixing bugs once, and knowing that a Python user and an R user who run the same model are running *literally the same code* — not two implementations that hopefully agree.

### Four ways in — and the plain-C door is built early

Python and R are the main audiences.
But the C interface is built second, not last, and Rust users can use the engine directly with no connecting layer at all.

**Why C early.**
A C interface only permits simple, explicit things: blocks of numbers, clear ownership of memory, no language-specific conveniences.
Building it early forces the whole design to stay that simple — which is exactly what keeps the per-iteration cost tiny and makes the library attachable from almost any other language later (Julia, MATLAB, and others can all connect to a C interface).
If Python were the only door for a year, the design would quietly grow Python-shaped habits that C could no longer accommodate.

**Why Rust directly.**
Every other language pays a small cost each time it talks to the library.
Using the engine from Rust, the cost is exactly zero — the user's loop and the sampler compile into one program.
That is the right tool for microsecond-scale likelihoods run tens of millions of times.

---

## 2. The conversation between you and the sampler

### You keep the loop

This is the single biggest decision.
Most MCMC software takes your log-posterior function and runs the whole chain itself, calling your function internally. bayesrs inverts that: the sampler proposes (`ask`), *you* evaluate, you report back (`tell`), and the sampler takes its step.

**Why.**
If the library ran the loop, Rust would have to call back into Python or R millions of times — and those languages have strict rules about who may call them, when, and from which thread.
Whole categories of crashes, deadlocks and platform-specific bugs live in that pattern.
Handing you the loop removes the entire category: the library never calls your code, so nothing about your language needs to live inside the library.

Three practical benefits follow for free.
A bug in your log-posterior crashes *your* program, in your debugger, with your error messages — not somewhere inside a foreign library.
Parallelising the likelihood is your choice, with your language's ordinary tools, because you own the evaluation step.
And the sampler hands you all chains' proposals in one batch, so a vectorised likelihood evaluates every chain at nearly the cost of one.

**The cost.**
Two boundary crossings per iteration, each carrying a small block of numbers — well under a microsecond, which is negligible next to any realistic likelihood.
For the rare case where it is not negligible (a compiled, microsecond-scale likelihood), there is a planned escape hatch (§7.6) and the zero-cost Rust route.

### The verbs are `ask` and `tell`

**Why.**
This is not our coinage: "ask/tell" is the established name for exactly this pattern in black-box optimisation software (pycma, scikit-optimize, Nevergrad, Optuna) and in the MCMC library PINTS.
People who have used any of those recognise the contract instantly.
The names are also short, and — unlike our early draft's `update` — collide with nothing in R.

### Count draws, not iterations

The recommended loop is "while the sampler has fewer than N draws", not "repeat N times".

**Why.**
For a single basic sampler the two are identical — one ask/tell round produces one draw per chain.
But that arithmetic is not the contract, and the first release already breaks it: the `Gibbs` compositor (§5) takes one round per *block* per draw, and NUTS will need a *variable* number of evaluations per draw.
Because the contract from day one is "keep going until you have enough draws", such samplers drop into existing programs without changing a single line of user code.
Had we promised "one round = one draw", adding any of them would have broken every user's loop.

This promise is kept honest by a test, not by good intentions: the test suite contains a deliberately awkward mock sampler (variable batch sizes, several rounds per draw) that is driven through every language's connecting layer, so no code can quietly assume the simple arithmetic.

### You provide the starting points

The sampler requires explicit starting values for every chain, and the very first `ask()` simply hands them back for evaluation.

**Why.**
The library never sees your prior — you only ever send it numbers — so it *cannot* invent sensible starting points, and pretending otherwise (zeros? random guesses?) would fail silently on real problems.
Making `x0` explicit also makes the loop uniform from the first iteration: no special initialisation step to forget.

### `-inf` is an answer; `NaN` is an error

Returning minus infinity ("this proposal is outside my posterior's support") is always legal: the sampler rejects and moves on.
Returning NaN stops the program with an error naming the offending chain.

**Why.**
These two look similar but mean opposite things.
`-inf` is a correct statement about your model.
NaN is almost always a bug — a 0/0, a log of a negative number — and an MCMC run that silently treats bugs as rejections can produce a full set of plausible-looking, wrong results.
Loud and early beats quiet and wrong.

For the same reason, calling `ask` twice in a row, or `tell` without an outstanding `ask`, gives an immediate clear error — it is impossible to corrupt a chain by calling things in the wrong order.
And if you `tell` the sampler a quantity it does not use (say, gradients to a random-walk sampler), that is an error too: it means you are paying to compute something that is being thrown away, and you would want to know.

---

## 3. Describing parameters

### Declare everything once, up front

Before sampling you declare your parameters — names, kinds, shapes — in a **parameter space**.
That declaration is made once; after that, everything that passes between you and the library is plain numbers.

**Why.**
None of that information ever changes during a run, so there is no reason to send it more than once.
This one split — rich description at setup, raw numbers forever after — is what allows the library to support genuinely structured parameters (covariance matrices, simplexes) *and* keep the per-iteration exchange down to a single small matrix.

### Kinds, not just names

You declare not just that a parameter exists but what *kind* of quantity it is: unconstrained real, bounded, simplex, covariance matrix.

**Why.**
The kind determines what a sensible proposal even looks like.
A random step is fine for a real number; it is nonsense for a covariance matrix, where naive stepping produces invalid (non-positive-definite) matrices almost every time.
The sampler can only choose the right behaviour if you tell it what it is dealing with.
This matters even more for planned future kinds: an ordered integer invites "current value ± 1" proposals, while an unordered category makes ±1 meaningless — a statistical distinction, not a cosmetic one, and one the library can only respect if the declaration captures it.

### The library owns the transforms — and the Jacobians

Internally, the sampler works with unconstrained versions of your parameters (the log of a standard deviation; a factorisation of a covariance matrix), proposes freely there, and maps back before you see anything.
Changing variables like this requires a Jacobian correction to the density.
The library applies it.
You write the log-posterior of your actual parameters — the density you would put on paper — and nothing else.

**Why.**
Hand-derived Jacobians for matrix transforms are precisely the kind of mathematics that produces subtle, silent, hard-to-detect errors, in a place where users have no easy way to check themselves.
Centralising them means they are derived once, tested once (including against numerical differentiation, in the automated test suite), and correct for everyone.
The transforms follow Stan's, which have a decade of scrutiny behind them.

Two guarantees make the machinery invisible: every value you are handed is valid (a covariance matrix is genuinely positive-definite; a standard deviation is strictly positive), and you never write a Jacobian.

### Bounds are open intervals — and that guarantee has no asterisk

`Bounded(0, ∞)` means the open interval: there is no way, and no need, to say whether an endpoint is "included".
For a continuous parameter the endpoint has probability zero, so the distinction is statistically meaningless.
(Wanting actual probability mass *at* zero is a different model — a mixture of a spike and a continuous part — not a bracket choice.)

One subtlety was worth engineering away: computers' floating-point arithmetic can round a value that is mathematically strictly inside the interval onto the boundary exactly.
The library nudges such values back inside by the smallest possible amount, so "strictly positive" is unconditionally true — your unguarded `log(sigma)` can never receive a zero.
Symmetrically, starting values must sit strictly inside their bounds.

### Everything travels as one flat matrix of ordinary numbers

The batch from `ask()` is a plain matrix of double-precision numbers: one row per chain, one column per parameter value, in declaration order.
No labels, no nested structure, no special types.

**Why.**
This is the one format that maps directly, with no conversion, onto Python's arrays, R's matrices, C's memory, and Rust's arrays.
And it needs no second array type even for discrete parameters: a categorical state travels in the same matrix as a whole-valued double (`3.0` meaning state 3 — exact, since doubles represent every integer up to about 9×10¹⁵), a rule the shipped `Categorical` kind exercises and future integer kinds inherit (§5).

Because numbered columns are fragile — insert a parameter and every index shifts — the parameter space provides `unpack` (wrap the matrix in named, correctly-shaped pieces) and `pack` (build a starting-value matrix from named pieces).
Names live in the declaration, written once; nobody counts columns.

### You get copies, not windows into the library's memory

The batch `ask()` returns in Python or R is an ordinary array that belongs to you.
An earlier draft of this design handed out "live views" instead — arrays that pointed directly at the library's internal memory and silently changed on every `ask`.

**Why we changed course.**
The live version was faster on paper, but the saving was about a microsecond per iteration — invisible next to any real likelihood — and the dangers were real: values that mutate behind your back, saved "copies" that all turn out identical, crashes if a view outlives the sampler.
A design that needs a warning box titled "the one thing to know" for its most basic operation is charging too much for a microsecond.
Copies are boring, safe, and fast enough; if profiling ever shows otherwise, live views can return as an explicit opt-in (§7.7).

---

## 4. Results, randomness, and reproducibility

### The library stores the chains; you look at them whenever you like

Draws accumulate inside the sampler and come out on request as one array, with your parameters in their natural, constrained form — and "on request" means at any moment, not just at the end.
Because you own the loop, there is nothing special about stopping: run a thousand draws, look at R̂, and decide whether to continue; print summaries to a file every ten thousand; keep sampling until the effective sample size crosses a threshold rather than for a count fixed in advance.
This is one of the quiet major wins of the ask/tell design.
Libraries that run the whole chain internally have to invent extra machinery for mid-run monitoring — callbacks, progress hooks, stopping rules — and you get whatever they thought to provide.
Here the "machinery" is an ordinary `if` statement in your own loop, and any check you can write is allowed.

A `thin` option stores every k-th draw for long runs — the memory arithmetic (chains × draws × dimensions × 8 bytes) is documented up front, because a million-draw run can silently mean gigabytes.

### Diagnostics are built in, using the modern definitions

R̂ and effective sample size ship with the library — the rank-normalised, split-chain versions from Vehtari et al. (2021), not the older definitions that can miss real convergence failures.
They are computed by the same core code in every language.
The draws come out in the shapes the wider ecosystems expect (ArviZ in Python; the `posterior`/`bayesplot` packages in R), so richer plotting and summaries are one conversion away.

**Why built in.**
Convergence checking is not optional in serious work, so the zero-effort path should include it — and a C user has no ArviZ to lean on.

### Every random number comes from the library, and runs are exactly repeatable

You give a seed; the library derives all randomness from it using its own generator, never the host language's.
Consequently: same seed, same model, same data ⇒ the same chains, to the last bit, every run — and, by design, across languages too: the Python, R, C, and Rust routes drive identical machinery, and the automated tests include a problem constructed so its likelihood computes identically everywhere, verifying that all four produce byte-for-byte identical chains.

**Why.**
Exact repeatability turns "it did something odd around iteration 40,000" from an anecdote into a reproducible bug report; it makes results in papers and teaching materials checkable; and the cross-language test is a merciless detector of accidental differences between the language layers.
One design detail here is itself future-proofing: random streams are set up so that adding new sampler features later cannot perturb the streams existing chains draw from — reproducibility survives library upgrades of that kind.

---

## 5. Discrete parameters, and samplers made of samplers

The first design deferred discrete parameters entirely.
We changed that decision — deliberately, and late — after working through how a statistician actually handles a mixed continuous/discrete model (the worked case: `signal_detection_demo.ipynb`, "is a pulse of known shape present in this noisy recording?").
What follows is the reasoning.

### A basic discrete kind ships in the first release

Only one — `Categorical(k)`, an unordered one-of-k state — with ordered integers still to come.

**Why.**
Every claim in §7 of the form "this will slot in later without breaking anything" was checked on paper.
For discrete parameters we found we could do better than paper: shipping one discrete kind, one discrete kernel, and the composition machinery forces the whole chain — wire format, transform layer, kind validation, block composition, diagnostics — to be *built* right rather than argued right.
It is the difference between an extensibility story and an extensibility test.
The smallest kind that exercises all of it is `Categorical`; the ordered kinds add proposal styles, not new machinery.

### One code per categorical, never indicator vectors

A one-of-k parameter is stored as a single whole number 0…k−1, not as k zero/one indicators.

**Why.**
With indicators, invalid states (two "hot", none "hot") are representable, and every kernel would have to maintain a constraint the declaration cannot see.
With a single code, invalid states cannot be written down at all, and discrete kernels act on exactly the object they reason about: the set of k states.
Text labels ("absent", "present") can be attached for reading results — they are presentation, and never travel in the arrays.
If a model genuinely has k independent on/off switches, that is k separate binary parameters, declared as such — a different model, not an encoding choice.

### No kernel touches a kind it doesn't understand

A random-walk sampler *refuses* a space containing a categorical (at construction, loudly), and the discrete kernel refuses continuous kinds.
Gradient-based samplers, when they arrive, will refuse categoricals too — a gradient between unordered states is not a thing that exists.

**Why.**
The alternative — silently treating a category label as a number — produces exactly the wrong-answer-shaped bugs this library exists to prevent (a Gaussian step to state 0.37; a covariance adapter averaging labels).
The kind-validation rule was designed in §6 for future safety; discrete kinds are where it starts doing real work.

### Mixed models: blocks, and a sampler made of samplers

So how do you sample a model with both kinds, when no single kernel accepts both?
You split the parameters into **blocks** — each owned by a kernel suited to it — and compose the blocks with `Gibbs`, which updates each block in turn against the full posterior, holding the others at their current values (the classical Metropolis-within-Gibbs scheme).
The crucial property: `Gibbs` is itself just a sampler.
Your loop is unchanged; you still write one log-posterior over the full parameter set; each draw simply takes one trip through the loop per block instead of one in total.

**Why this shape.**
Three candidate designs were on the table.
An "external" mechanism bolted alongside the protocol (an early draft had an untyped `update_discrete(...)` call) — rejected: it doubles the protocol surface for one use case.
Making the *user* always run the alternation — workable (and still supported, below), but it puts sweep bookkeeping in every user's code.
Or a compositor that is an ordinary sampler containing samplers — which needs *nothing new in the protocol at all*, because "several exchanges per draw" was already the contract (that is what "count draws, not iterations" bought, §2).
The composition is also honest about cost: each block's update genuinely requires its own density evaluation (the second block must be evaluated at whatever the first block's update produced — the evaluations are sequential by nature), and the compositor adds zero evaluations beyond that: the density you told for an accepted move is carried across the block boundary rather than recomputed.

Two boundary rules were settled restrictive-now, relaxable-later: blocks partition the parameters at whole-parameter granularity (updating a single 100-element allocation vector site by site is the *kernel's* internal strategy, not a reason to split a parameter across blocks), and no parameter belongs to two blocks (a parameter updated by two kernels is a mixture-of-kernels composition — a planned separate compositor, §7).

**Blocking is not a discrete-parameters feature.**
Discreteness *forces* the split, but the same mechanism serves models where one parameter's density has a kink no gradient sampler can handle, or where two groups of parameters want different tuning.
And it is the door NUTS walks through for mixed models: NUTS on the smooth block, the discrete kernel on the categorical — composed by the same `Gibbs`, driven by the same user loop.

### Marginalise first, sample second

The documentation's first recommendation for a discrete parameter is: if you can cheaply sum it out of your likelihood, do that instead — every parameter is then continuous, any sampler applies, and mixing is typically better.
The sampling machinery is for when marginalisation is genuinely unavailable.

**Why say this in the manual of the library that sells the machinery.**
Because it is true, and because a library for statisticians should recommend the best statistics, not the feature.
The two-hypothesis signal-detection model *can* be marginalised in two lines; a mixture model with 100 allocation variables cannot, and that is what the machinery is for.

### The escape hatch, and its one sharp edge

You can also compose by hand: construct two samplers over two parameter subsets and run the alternation yourself — full control over scan order and scheduling.
But encapsulation cuts one corner: each sampler caches the density of its current point, and when *your* loop moves the other block, that cached value goes stale — every acceptance ratio afterwards would be quietly wrong.
Hence `reanchor`: after one block accepts, you pass the other sampler the corrected density — a number you already hold from the accepting sampler's `tell`, so the fix costs no extra evaluation.
`Gibbs` does this internally; hand-rolled composition must do it explicitly, the documentation says so loudly, and the test suite includes a deliberately-broken loop that omits it, to prove the failure is real and detectable.

### A preview: the same model once NUTS exists

NUTS is not in the first release, but the composition machinery was designed against it, and walking the signal-detection model through both composition styles shows why nothing will need to change.
Take the continuous block (μ, σ) on NUTS and the binary indicator z on the discrete kernel.

**Via `Gibbs`.**
The user's loop is still `ask` → evaluate → `tell` (now supplying gradients too, because a NUTS block asks for them).
Internally, each chain's draw is one sweep: the NUTS block builds its leapfrog trajectory — each step one exchange, z frozen — for however many exchanges the U-turn rule takes; the trajectory ends, a new (μ, σ) is selected, and control passes to the discrete block *because the block finished its draw*, never because something was "accepted"; one more exchange flips z at the new (μ, σ); the sweep is complete and the draw counts.
Chains don't wait for each other: trajectories have different lengths, so one batch may carry chain 0's seventh leapfrog point alongside chain 2's discrete proposal — which the user never notices, because the rule was always "evaluate the rows; don't ask whose they are".
Every evaluation is used exactly once; the density told for the trajectory's selected point anchors the discrete step for free.

**By hand.**
The user constructs a plain NUTS sampler over (μ, σ) only, and keeps z themselves.
Here a real subtlety appears: NUTS chains complete draws at different moments, mid-run the batch rows are deliberately opaque, and the user can neither see nor act on "chain 2 just finished a draw" — the exchange is all-or-nothing across chains.
So the by-hand route does not operate per chain; it operates at **barriers**.
The user runs the sampler with a draw target — "until every chain has exactly N draws" — and the sampler parks chains as they individually get there (the batch shrinking as they do, which the batching rules already permit), until all chains sit at their draw-N positions.
At the barrier the user flips every chain's z at its current (μ, σ) — one density evaluation and their own accept/reject per chain — `reanchor`s the NUTS sampler with the resulting densities (per chain; unchanged values where the flip rejected), raises the target, and continues.
The user-settable draw target is the one small primitive this route needs, and it is reserved in the interface.
The barrier is also what keeps the composition *safe*: chains stopping at a fixed N gives a deterministic scan schedule, where letting each chain run ahead by its own trajectory-dependent amount would make the schedule depend on the chain's state — dangerous territory for correctness.

NUTS itself is entirely ignorant that composition is happening: it sees an ordinary target density, correctly conditioned on whatever z currently is.
That ignorance is the point — any kernel that speaks ask/tell composes this way, with `reanchor` and a draw target as the only machinery.

The honest comparison, then: by hand, z updates once per barrier and fast chains idle at it; under `Gibbs`, each chain's z flips the moment *its own* trajectory ends, while other chains are still mid-flight, because the compositor sits inside the batch assembly where per-chain scheduling lives.
Both are valid; the compositor mixes better and does the bookkeeping for you — which is why it is the primary route and the by-hand version is an escape hatch.
And the reassurance to take from the whole preview: the sampler the library will gain years from now already has its control flow accounted for in the loop the user writes on day one.

---

## 6. What the first release deliberately leaves out

The first release ships **four samplers** (random-walk Metropolis, its self-tuning adaptive variant, the discrete Metropolis kernel, and the `Gibbs` compositor) and **five parameter kinds** (real, bounded, simplex, covariance matrix, categorical).

**Why so small.**
The product being frozen is not the sampler list — it is the *interface*: the verbs, the shapes, the conventions, the guarantees.
Samplers and kinds are additions; interface changes are breakages.
So the first release is the smallest thing that exercises every part of the design for real — the loop, the transforms, the Jacobians, the diagnostics, block composition, all four language routes — while every planned extension has been checked against the interface on paper (this document's §7 is that checklist).
Shipping small and early gets the interface into real users' hands, which is the only reliable way to find out whether it is right, while it is still cheap to adjust.

One safeguard makes small-now compatible with big-later: every sampler checks at construction that it supports every kind in your parameter space (§5).
When ordered integer parameters arrive, handing one to a sampler that predates them produces an immediate clear error — never a silent wrong proposal.

---

## 7. The doors left open

Each subsection here is a planned extension that the interface was explicitly audited against.
The common principle: **additions arrive as new names — new kinds, new optional fields, new functions — never as changed meanings of existing ones.**
That is what makes "your code keeps working" a design property rather than a hope.

### 7.1 Ordered integer parameters — and richer discrete proposals

`Categorical` shipped (§5); still planned are `Integer(lo, hi)` for ordered counts and `Binary()` as sugar for two states.
The distinction is statistical: "current ± 1" is a sensible proposal for a count and meaningless for a label, so ordered kinds get their own proposal machinery rather than reusing the categorical kernel's uniform choice.
Also planned: locally-informed categorical proposals (which weight candidate states by their posterior mass — the batch contract already lets one exchange carry all k candidates), and a random-scan option for `Gibbs`.

*Why nothing breaks:* these follow the shipped `Categorical` through the very doors it already exercised — same whole-numbers-in-the-float-matrix wire format, same pass-through transform layer, same kind validation, same composition.
That is precisely why a discrete kind went into the first release: so these arrive as more of the same, verified, not more of the argued-about.

### 7.2 Gradient-based samplers: MALA, HMC, NUTS

For hard, high-dimensional posteriors, samplers that use the gradient of the log-posterior are the modern answer.
The `tell` step reserves an optional `grad` field for them.

*Why nothing breaks:* the gradient convention is already pinned — you differentiate *your own code with respect to the numbers you were given* (which is exactly what automatic differentiation tools produce, with no adjustments), and the library handles all chain-rule corrections through its internal transforms, just as it handles Jacobians today.
NUTS's variable number of evaluations per draw is covered by the loop contract, and a sampler that requires gradients will say so on the very first iteration rather than running silently wrong.

One consequence is worth knowing in advance: NUTS chains will finish their draws at different rates (each builds a differently-sized trajectory), and rather than making fast chains wait, each batch simply carries every chain's *next needed point* — so chains drift apart in their draw counts, and "number of draws" means the number available from every chain.
Your loop doesn't change; the batch rows just stop being "one draw attempt per chain" and become "one unit of work per chain", which is why the rule was always "evaluate the rows; don't ask whose they are".

For mixed models, NUTS arrives already composable: a `Gibbs` block running NUTS on the smooth parameters next to the discrete kernel on the categorical ones (§5), through the unchanged loop.
One planned refinement supports this: a per-row hint saying which rows of a batch actually want gradients, so callers can skip gradient work on discrete-block rows rather than compute values the sampler will ignore.

### 7.3 Tempering and sequential Monte Carlo

Methods for multimodal posteriors and model evidence.
These need the likelihood and prior reported *separately* (so the likelihood can be "heated"), for which `tell` reserves `log_lik` and `log_prior` fields; their larger populations are just bigger batches, which the contract already allows.

### 7.4 More structured parameter kinds

Correlation matrices, Cholesky factors, and — covering most "custom" needs — *combinators*: ordered vectors, offset/scale reparameterisations, and compositions of existing kinds, all implemented inside the engine, so they stay fast, work identically in every language, and require nothing from you but the declaration.

For the genuinely exotic case there is a planned `Custom` kind where *you* supply the transform and its Jacobian as ordinary functions in your own language.
Doesn't that reintroduce the callbacks §2 forbids?
No — and the distinction is worth understanding.
What we rejected was the library calling your code *from its own loop, on its own schedule*.
A custom transform runs only while you are already inside a call you made — the library briefly hands control back on your own thread, exactly as Python's built-in `sorted` does when it calls your `key` function.
That is routine and safe.
The honest costs are different ones: a parameter space containing your Python functions can only be used from Python, and the library can no longer vouch that the transform's outputs are valid — that promise becomes yours.
There is also an escape valve available *today*, no `Custom` needed: declare the block as plain `Real` and apply your change of variables (with its log-Jacobian) inside your own log-posterior.
`Custom` is a convenience for doing that more safely, not a gate on what is possible.

### 7.5 Warmup phases and checkpointing

Future samplers with distinct adaptation phases will take a `warmup` option; "number of draws" will always mean draws you actually get to keep, so existing loops stay correct.
Saving and restoring a full sampler state mid-run (for long jobs on shared machines) is planned; the internal state has been kept fully serialisable from day one so this is an addition, not a redesign.

### 7.6 The escape hatch for very cheap likelihoods

If your log-posterior is itself compiled code (a compiled ODE model, say), a planned `run_compiled` lets you hand the library a direct reference to it and run millions of iterations entirely inside the engine, with zero boundary crossings — the same samplers, driven from the other side.

### 7.7 Opt-in live views

If profiling ever shows the copy in `ask()` mattering for someone real, the zero-copy "live view" mechanism (§3) can return as an explicit opt-in.
The default will stay safe.

---

## 8. The one permanent "no": reversible jump

Reversible-jump MCMC — chains that move between models *with different numbers of parameters*, growing and shrinking the parameter vector as they run — is permanently out of scope.
A fixed number of columns for the life of a sampler is load-bearing for everything above: the flat matrix, the declared-once layout, the named views, the storage format.

**Why we're comfortable with that.**
The standard fixed-dimension alternative expresses the same models: include a model-index parameter (a `Categorical`) alongside parameter blocks sized for the largest model, and let the likelihood ignore the inactive parts.
This "composite model space" formulation (Godsill, 2001) covers mixture models with unknown component counts, variable selection, and change-point problems — at the cost of some unused columns, which the flat-matrix design makes cheap.
What is genuinely lost is only the ability to avoid carrying the padding, and that trade buys the simplicity of everything else in this document.

---

## Appendix: the decisions at a glance

| Decision | Why, in one line |
|---|---|
| One Rust engine, all languages share it | Subtle MCMC code should be written and tested once |
| You keep the loop (`ask`/`tell`) | The library never calls your language, killing a whole class of cross-language failure |
| Verbs named `ask`/`tell` | The established name for this pattern; collides with nothing |
| Count draws, not iterations | Lets future samplers take variable work per draw without breaking loops |
| Explicit starting points | The library can't know your prior; pretending otherwise fails silently |
| `-inf` legal, `NaN` fatal | One is a statement about your model; the other is a bug that should be loud |
| Unused `tell` fields are errors | You'd want to know you're computing something that's discarded |
| Declare parameters once, then only numbers cross | Rich structure *and* a cheap iteration, from one split |
| Kinds, not just names | Sensible proposals depend on what a parameter *is* |
| Library owns transforms and Jacobians | Hand-derived Jacobians are a silent-error factory |
| Bounds are open intervals, guaranteed strictly | Endpoints are probability-zero; floating point is nudged to keep the promise exact |
| One flat float64 matrix | Maps to every language's arrays unchanged; fits future integer kinds |
| `pack`/`unpack`, never column counting | Numbered columns break when the model grows |
| Copies, not live views | A microsecond is too cheap a price to accept mutating-behind-your-back arrays |
| Diagnostics built in, modern definitions | Convergence checking shouldn't be optional or require another package |
| All randomness from the library's seeded generator | Bit-exact repeatability, within and across languages |
| Small v1: four samplers, five kinds | The interface is the product; everything else is an addition |
| A basic discrete kind ships in v1 | Extensibility verified by shipped code, not by argument |
| Categorical = one code 0…k−1, never indicators | Invalid states become unrepresentable; kernels act on the k-state set |
| Samplers validate kinds at construction | Future kinds fail loudly on old samplers, never silently |
| Mixed models via `Gibbs` blocks; a draw is a full sweep | Composition is just a sampler; the user's loop never changes |
| Marginalise a discrete parameter when you cheaply can | The best statistics beats the feature |
| `reanchor` on every sampler | Hand-rolled composition stays *correct*, not merely possible |
| C interface built second | Keeps the design honest, simple, and attachable from any language |
| Rust core is public | The zero-boundary-cost option costs nothing to offer |
| Reversible jump permanently excluded | Fixed dimensions are load-bearing; composite-space models cover the need |
