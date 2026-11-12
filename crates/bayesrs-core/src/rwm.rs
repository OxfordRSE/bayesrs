//! Random-walk Metropolis (INTERFACE.md §5.1).

use ndarray::{Array2, Array3, ArrayView2};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::space::Kind;
use crate::{Error, ParamSpace, Sampler, Tell, rng};

const NAME: &str = "RandomWalkMetropolis";

/// Tuning options for [`RandomWalkMetropolis`].
///
/// Start from `RwmOptions::default()` and set fields; new options may be
/// added in later versions without breaking that pattern.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RwmOptions {
    /// Free-space proposal covariance, `(d_free, d_free)`, symmetric
    /// positive-definite. `None` means `(2.38² / d_free) · I`.
    pub proposal_cov: Option<Array2<f64>>,
    /// Store every `thin`-th draw (draw 0 is always stored). Must be ≥ 1.
    pub thin: usize,
}

impl Default for RwmOptions {
    fn default() -> Self {
        RwmOptions {
            proposal_cov: None,
            thin: 1,
        }
    }
}

/// Monitoring statistics from [`RandomWalkMetropolis::stats`].
///
/// May gain fields in later versions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RwmStats {
    /// Stored draws per chain.
    pub n_draws: usize,
    /// Fraction of Metropolis proposals accepted, per chain
    /// (`NaN` before the first proposal).
    pub acceptance_rate: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Ready,
    WaitingForTell,
}

/// Random-walk Metropolis with a fixed Gaussian proposal in free space.
///
/// ```
/// use bayesrs_core::{ParamSpace, RandomWalkMetropolis, Sampler, Tell};
///
/// let space = ParamSpace::builder().real("x").build()?;
/// let mut s = RandomWalkMetropolis::new(&space, &[0.0, 1.0], 2, 42)?;
/// while s.n_draws() < 100 {
///     let logp: Vec<f64> = s.ask()?.rows().into_iter()
///         .map(|row| -0.5 * row[0] * row[0])
///         .collect();
///     s.tell(Tell::new(&logp))?;
/// }
/// assert_eq!(s.draws().shape(), [2, 100, 1]);
/// # Ok::<(), bayesrs_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct RandomWalkMetropolis {
    space: ParamSpace,
    n_chains: usize,
    thin: usize,
    /// Lower Cholesky factor of the proposal covariance, row-major.
    chol: Vec<f64>,
    rngs: Vec<ChaCha8Rng>,
    state: State,
    initialised: bool,
    /// Current free-space points, `(n_chains, d_free)` row-major.
    current: Vec<f64>,
    /// Current free-space log target: caller's logp + log|det J|.
    current_target: Vec<f64>,
    /// Proposed free-space points of the outstanding ask.
    proposal: Vec<f64>,
    /// Log-uniforms for the outstanding ask's accept/reject, one per chain.
    log_u: Vec<f64>,
    /// The batch handed out by `ask`, `(n_chains, d_constr)`.
    batch: Array2<f64>,
    /// Stored free-space draws, one flat `(n_draws, d_free)` buffer per chain.
    stored: Vec<Vec<f64>>,
    n_stored: usize,
    /// Metropolis steps taken since draw 0.
    iterations: usize,
    accepted: Vec<u64>,
    /// Scratch for standard-normal increments.
    eps: Vec<f64>,
}

impl RandomWalkMetropolis {
    /// A sampler with default options.
    ///
    /// `x0` holds `n_chains` constrained-space starting points, one row of
    /// `space.d_constr()` values per chain.
    pub fn new(space: &ParamSpace, x0: &[f64], n_chains: usize, seed: u64) -> Result<Self, Error> {
        Self::with_options(space, x0, n_chains, seed, RwmOptions::default())
    }

    /// A sampler with the given options.
    pub fn with_options(
        space: &ParamSpace,
        x0: &[f64],
        n_chains: usize,
        seed: u64,
        options: RwmOptions,
    ) -> Result<Self, Error> {
        check_kinds(space)?;
        let (d_free, d_constr) = (space.d_free(), space.d_constr());
        if n_chains == 0 {
            return Err(Error::InvalidOption {
                sampler: NAME,
                option: "n_chains",
                reason: "must be at least 1".into(),
            });
        }
        if options.thin == 0 {
            return Err(Error::InvalidOption {
                sampler: NAME,
                option: "thin",
                reason: "must be at least 1".into(),
            });
        }
        if x0.len() != n_chains * d_constr {
            return Err(Error::WrongLength {
                what: "x0 (n_chains × d_constr)",
                expected: n_chains * d_constr,
                got: x0.len(),
            });
        }

        let mut current = vec![0.0; n_chains * d_free];
        for (c, (row, out)) in x0
            .chunks_exact(d_constr)
            .zip(current.chunks_exact_mut(d_free))
            .enumerate()
        {
            space.to_free(row, out).map_err(|e| match e {
                Error::InvalidStart { block, reason, .. } => Error::InvalidStart {
                    block,
                    chain: c,
                    reason,
                },
                other => other,
            })?;
        }

        let chol = match &options.proposal_cov {
            None => {
                let scale = 2.38 / (d_free as f64).sqrt();
                let mut l = vec![0.0; d_free * d_free];
                for i in 0..d_free {
                    l[i * d_free + i] = scale;
                }
                l
            }
            Some(cov) => proposal_cholesky(cov, d_free)?,
        };

        Ok(RandomWalkMetropolis {
            space: space.clone(),
            n_chains,
            thin: options.thin,
            chol,
            rngs: (0..n_chains as u64).map(|c| rng::stream(seed, c)).collect(),
            state: State::Ready,
            initialised: false,
            current,
            current_target: vec![f64::NAN; n_chains],
            proposal: vec![0.0; n_chains * d_free],
            log_u: vec![0.0; n_chains],
            batch: Array2::from_shape_vec((n_chains, d_constr), x0.to_vec())
                .expect("x0 length checked above"),
            stored: vec![Vec::new(); n_chains],
            n_stored: 0,
            iterations: 0,
            accepted: vec![0; n_chains],
            eps: vec![0.0; d_free],
        })
    }

    /// The parameter space this sampler was built on.
    pub fn space(&self) -> &ParamSpace {
        &self.space
    }

    /// Current state of every chain, `(n_chains, d_constr)`, constrained.
    pub fn position(&self) -> Array2<f64> {
        let (d_free, d_constr) = (self.space.d_free(), self.space.d_constr());
        let mut out = Array2::zeros((self.n_chains, d_constr));
        for (z, mut row) in self.current.chunks_exact(d_free).zip(out.rows_mut()) {
            self.space
                .to_constrained(z, row.as_slice_mut().expect("standard layout"));
        }
        out
    }

    /// Draw count and per-chain acceptance rates.
    pub fn stats(&self) -> RwmStats {
        let proposals = self.iterations as f64;
        RwmStats {
            n_draws: self.n_stored,
            acceptance_rate: self
                .accepted
                .iter()
                .map(|&a| {
                    if self.iterations == 0 {
                        f64::NAN
                    } else {
                        a as f64 / proposals
                    }
                })
                .collect(),
        }
    }

    fn store_current(&mut self) {
        let d_free = self.space.d_free();
        for (c, buf) in self.stored.iter_mut().enumerate() {
            buf.extend_from_slice(&self.current[c * d_free..(c + 1) * d_free]);
        }
        self.n_stored += 1;
    }
}

impl Sampler for RandomWalkMetropolis {
    fn ask(&mut self) -> Result<ArrayView2<'_, f64>, Error> {
        if self.state == State::WaitingForTell {
            return Err(Error::AskBeforeTell);
        }
        if self.initialised {
            let d = self.space.d_free();
            for c in 0..self.n_chains {
                let rng = &mut self.rngs[c];
                for e in &mut self.eps {
                    *e = rng.sample(StandardNormal);
                }
                // All randomness for this exchange is drawn here, so `tell`
                // is a pure function of its input (and safely retryable).
                self.log_u[c] = rng.random::<f64>().ln();

                let z = &self.current[c * d..(c + 1) * d];
                let prop = &mut self.proposal[c * d..(c + 1) * d];
                for i in 0..d {
                    let step: f64 = (0..=i).map(|j| self.chol[i * d + j] * self.eps[j]).sum();
                    prop[i] = z[i] + step;
                }
                let mut row = self.batch.row_mut(c);
                self.space
                    .to_constrained(prop, row.as_slice_mut().expect("standard layout"));
            }
        }
        // Before initialisation the batch already holds x0 verbatim.
        self.state = State::WaitingForTell;
        Ok(self.batch.view())
    }

    fn tell(&mut self, t: Tell<'_>) -> Result<(), Error> {
        if self.state == State::Ready {
            return Err(Error::TellWithoutAsk);
        }
        for (field, present) in [
            ("grad", t.grad.is_some()),
            ("log_lik", t.log_lik.is_some()),
            ("log_prior", t.log_prior.is_some()),
        ] {
            if present {
                return Err(Error::UnusedField {
                    sampler: NAME,
                    field,
                });
            }
        }
        let logp = t.logp.ok_or(Error::MissingField {
            sampler: NAME,
            field: "logp",
        })?;
        if logp.len() != self.n_chains {
            return Err(Error::WrongLength {
                what: "tell(logp)",
                expected: self.n_chains,
                got: logp.len(),
            });
        }
        for (chain, &value) in logp.iter().enumerate() {
            if value.is_nan() || value == f64::INFINITY {
                return Err(Error::InvalidLogDensity {
                    operation: "tell",
                    chain,
                    value,
                });
            }
            if !self.initialised && value == f64::NEG_INFINITY {
                return Err(Error::OutsideSupport {
                    operation: "tell",
                    chain,
                });
            }
        }

        // Validation passed: from here on nothing can fail.
        let d = self.space.d_free();
        if !self.initialised {
            for (c, &lp) in logp.iter().enumerate() {
                let z = &self.current[c * d..(c + 1) * d];
                self.current_target[c] = lp + self.space.log_abs_det_jacobian(z);
            }
            self.initialised = true;
            self.store_current();
        } else {
            for (c, &lp) in logp.iter().enumerate() {
                let prop = &self.proposal[c * d..(c + 1) * d];
                let target = lp + self.space.log_abs_det_jacobian(prop);
                if self.log_u[c] < target - self.current_target[c] {
                    self.current[c * d..(c + 1) * d].copy_from_slice(prop);
                    self.current_target[c] = target;
                    self.accepted[c] += 1;
                }
            }
            self.iterations += 1;
            if self.iterations.is_multiple_of(self.thin) {
                self.store_current();
            }
        }
        self.state = State::Ready;
        Ok(())
    }

    fn reanchor(&mut self, logp: &[f64]) -> Result<(), Error> {
        if self.state == State::WaitingForTell {
            return Err(Error::ReanchorWhileWaiting);
        }
        if logp.len() != self.n_chains {
            return Err(Error::WrongLength {
                what: "reanchor(logp)",
                expected: self.n_chains,
                got: logp.len(),
            });
        }
        for (chain, &value) in logp.iter().enumerate() {
            if value.is_nan() || value == f64::INFINITY {
                return Err(Error::InvalidLogDensity {
                    operation: "reanchor",
                    chain,
                    value,
                });
            }
            if value == f64::NEG_INFINITY {
                return Err(Error::OutsideSupport {
                    operation: "reanchor",
                    chain,
                });
            }
        }
        let d = self.space.d_free();
        for (c, &lp) in logp.iter().enumerate() {
            let z = &self.current[c * d..(c + 1) * d];
            self.current_target[c] = lp + self.space.log_abs_det_jacobian(z);
        }
        Ok(())
    }

    fn n_draws(&self) -> usize {
        self.n_stored
    }

    fn n_chains(&self) -> usize {
        self.n_chains
    }

    fn draws(&self) -> Array3<f64> {
        let (d_free, d_constr) = (self.space.d_free(), self.space.d_constr());
        let mut out = Array3::zeros((self.n_chains, self.n_stored, d_constr));
        for (buf, mut chain) in self.stored.iter().zip(out.outer_iter_mut()) {
            for (z, mut row) in buf.chunks_exact(d_free).zip(chain.rows_mut()) {
                self.space
                    .to_constrained(z, row.as_slice_mut().expect("standard layout"));
            }
        }
        out
    }
}

/// RWM proposes Gaussian steps, which only make sense for continuous kinds.
fn check_kinds(space: &ParamSpace) -> Result<(), Error> {
    for block in space.blocks() {
        match block.kind() {
            Kind::Real => {}
        }
    }
    Ok(())
}

/// Validates a proposal covariance and returns its lower Cholesky factor.
fn proposal_cholesky(cov: &Array2<f64>, d: usize) -> Result<Vec<f64>, Error> {
    let invalid = |reason: String| Error::InvalidOption {
        sampler: NAME,
        option: "proposal_cov",
        reason,
    };
    if cov.dim() != (d, d) {
        return Err(invalid(format!(
            "must have shape ({d}, {d}) (d_free × d_free), got {:?}",
            cov.dim()
        )));
    }
    for i in 0..d {
        for j in 0..d {
            let (a, b) = (cov[[i, j]], cov[[j, i]]);
            if !a.is_finite() {
                return Err(invalid(format!("has a non-finite entry at ({i}, {j})")));
            }
            if (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0) {
                return Err(invalid(format!("is not symmetric at ({i}, {j})")));
            }
        }
    }
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * d + k] * l[j * d + k]).sum();
            if i == j {
                let pivot = cov[[i, i]] - dot;
                if pivot <= 0.0 {
                    return Err(invalid("is not positive-definite".into()));
                }
                l[i * d + i] = pivot.sqrt();
            } else {
                l[i * d + j] = (cov[[i, j]] - dot) / l[j * d + j];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cholesky_reconstructs_the_covariance() {
        let cov = array![[4.0, 2.0, 0.6], [2.0, 2.0, 0.5], [0.6, 0.5, 3.0]];
        let l = proposal_cholesky(&cov, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| l[i * 3 + k] * l[j * 3 + k]).sum();
                assert!((v - cov[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn bad_proposal_covariances_are_rejected() {
        let cases = [
            array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], // wrong shape
            array![[1.0, 0.5], [0.4, 1.0]],             // asymmetric
            array![[1.0, 2.0], [2.0, 1.0]],             // indefinite
            array![[1.0, f64::NAN], [f64::NAN, 1.0]],   // non-finite
        ];
        for cov in cases {
            let err = proposal_cholesky(&cov, 2).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::InvalidOption {
                        option: "proposal_cov",
                        ..
                    }
                ),
                "{cov:?}"
            );
        }
    }
}
