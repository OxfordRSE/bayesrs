//! Core MCMC engine for bayesrs.
//!
//! Currently a scaffold: [`hello`] exists so that every binding can expose one
//! function while packaging, testing, and CI are wired up.
//! The real library (samplers, transforms, diagnostics, RNG streams) is
//! specified in `docs/INTERFACE.md`.

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
