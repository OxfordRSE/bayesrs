//! Core MCMC engine for bayesrs.
//!
//! The caller owns the loop: [`Sampler::ask`] hands out a batch of points,
//! the caller evaluates their log densities, and [`Sampler::tell`] hands them
//! back. The full contract is specified in `docs/INTERFACE.md`.
//!
//! Implemented so far: [`ParamSpace`] with the `Real` kind, and
//! [`RandomWalkMetropolis`].

mod error;
mod rng;
mod rwm;
mod sampler;
mod space;

pub use error::Error;
pub use rwm::{RandomWalkMetropolis, RwmOptions, RwmStats};
pub use sampler::{Sampler, Tell};
pub use space::{Block, Kind, ParamSpace, ParamSpaceBuilder, Transform};

/// Returns a version-bearing greeting.
///
/// Every binding exposes this and must return the identical string: the
/// scaffold's stand-in for the cross-language reproducibility guarantee.
pub fn hello() -> String {
    format!("bayesrs-core {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::hello;

    #[test]
    fn hello_reports_name_and_version() {
        assert_eq!(
            hello(),
            format!("bayesrs-core {}", env!("CARGO_PKG_VERSION"))
        );
    }
}
