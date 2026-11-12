//! The ask/tell protocol shared by every sampler (INTERFACE.md §2, §11).

use ndarray::{Array3, ArrayView2};

use crate::Error;

/// What the caller hands back in `tell`.
///
/// Built with [`Tell::new`] and `with_*` methods, never literally: the struct
/// is `#[non_exhaustive]` so fields (`metric`, …) can be added later.
/// Each sampler declares which fields it requires; supplying a field it does
/// not use is an error.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Tell<'a> {
    /// Log density of each row of the last `ask`, shape `(m,)`.
    pub logp: Option<&'a [f64]>,
    /// Reserved for gradient-based samplers, shape `(m, d_constr)` row-major.
    pub grad: Option<&'a [f64]>,
    /// Reserved for tempering / SMC, shape `(m,)`.
    pub log_lik: Option<&'a [f64]>,
    /// Reserved for tempering / SMC, shape `(m,)`.
    pub log_prior: Option<&'a [f64]>,
}

impl<'a> Tell<'a> {
    /// A `tell` payload carrying the log densities `logp`.
    pub fn new(logp: &'a [f64]) -> Self {
        Tell {
            logp: Some(logp),
            grad: None,
            log_lik: None,
            log_prior: None,
        }
    }

    /// Adds gradients (reserved: no v1 sampler uses them).
    pub fn with_grad(mut self, grad: &'a [f64]) -> Self {
        self.grad = Some(grad);
        self
    }

    /// Adds per-row log-likelihoods (reserved: no v1 sampler uses them).
    pub fn with_log_lik(mut self, log_lik: &'a [f64]) -> Self {
        self.log_lik = Some(log_lik);
        self
    }

    /// Adds per-row log-priors (reserved: no v1 sampler uses them).
    pub fn with_log_prior(mut self, log_prior: &'a [f64]) -> Self {
        self.log_prior = Some(log_prior);
        self
    }
}

/// An MCMC sampler driven by the caller's ask/tell loop.
///
/// The canonical loop counts draws, never exchanges:
///
/// ```ignore
/// while sampler.n_draws() < n {
///     let theta = sampler.ask()?;
///     let logp = evaluate(theta);
///     sampler.tell(Tell::new(&logp))?;
/// }
/// ```
pub trait Sampler {
    /// Returns the `(m, d_constr)` batch of constrained-space points whose log
    /// densities the caller must evaluate. The first call returns `x0`.
    ///
    /// The view borrows the sampler, so the compiler forbids holding it across
    /// the following `tell`; copy it (`to_owned()`) to keep it.
    fn ask(&mut self) -> Result<ArrayView2<'_, f64>, Error>;

    /// Consumes the evaluations of the last `ask`, in row order.
    ///
    /// On error the sampler stays waiting with its chains untouched, so the
    /// caller can fix the problem and `tell` again.
    fn tell(&mut self, t: Tell<'_>) -> Result<(), Error>;

    /// Replaces each chain's cached current-point log density (`(n_chains,)`).
    /// Needed only for hand-rolled composition of several samplers.
    fn reanchor(&mut self, logp: &[f64]) -> Result<(), Error>;

    /// Number of stored draws per chain (draw 0 is `x0`).
    fn n_draws(&self) -> usize;

    /// Number of chains.
    fn n_chains(&self) -> usize;

    /// All stored draws, `(chain, draw, d_constr)`, in constrained space.
    fn draws(&self) -> Array3<f64>;
}
