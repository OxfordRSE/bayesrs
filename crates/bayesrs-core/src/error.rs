//! The single error type returned by every fallible operation in the core.

use std::fmt;

/// Everything that can go wrong in bayesrs.
///
/// Marked `#[non_exhaustive]` so new variants can be added without breaking
/// callers (INTERFACE.md §11): always include a wildcard arm when matching.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// `ask()` was called while a previous `ask()` still awaits its `tell()`.
    AskBeforeTell,
    /// `tell()` was called with no outstanding `ask()`.
    TellWithoutAsk,
    /// `reanchor()` was called while an `ask()` awaits its `tell()`.
    ReanchorWhileWaiting,
    /// A log density was `NaN` or `+inf`: almost always a bug in the caller's
    /// log-posterior.
    InvalidLogDensity {
        /// The operation that received the value (`"tell"` or `"reanchor"`).
        operation: &'static str,
        /// Index of the offending row / chain.
        chain: usize,
        /// The offending value.
        value: f64,
    },
    /// A chain's current point has log density `-inf` (on the first `tell`,
    /// or passed to `reanchor`): the chain has nowhere to stand.
    OutsideSupport {
        /// The operation that received the value (`"tell"` or `"reanchor"`).
        operation: &'static str,
        /// Index of the offending chain.
        chain: usize,
    },
    /// An array had the wrong length.
    WrongLength {
        /// What the array was (e.g. `"tell(logp)"`, `"x0"`).
        what: &'static str,
        /// The length that was required.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
    /// A `Tell` was missing a field the sampler requires.
    MissingField {
        /// The sampler that requires the field.
        sampler: &'static str,
        /// The missing field.
        field: &'static str,
    },
    /// A `Tell` carried a field the sampler does not use: the caller is
    /// computing something that would be thrown away.
    UnusedField {
        /// The sampler that would ignore the field.
        sampler: &'static str,
        /// The unused field.
        field: &'static str,
    },
    /// A starting value is invalid for its parameter block.
    InvalidStart {
        /// The parameter block.
        block: String,
        /// The chain whose starting value is invalid.
        chain: usize,
        /// Why it is invalid.
        reason: String,
    },
    /// The parameter space declaration is invalid.
    InvalidSpace(String),
    /// A sampler option is invalid.
    InvalidOption {
        /// The sampler the option belongs to.
        sampler: &'static str,
        /// The option name.
        option: &'static str,
        /// Why it is invalid.
        reason: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AskBeforeTell => f.write_str(
                "ask() called again before tell(); tell() the results of the previous ask() first.",
            ),
            Error::TellWithoutAsk => f.write_str("tell() called with no outstanding ask()."),
            Error::ReanchorWhileWaiting => f.write_str(
                "reanchor() called while an ask() awaits its tell(); tell() first, then reanchor().",
            ),
            Error::InvalidLogDensity {
                operation,
                chain,
                value,
            } => write!(
                f,
                "{operation}(): log density for chain {chain} is {value}; \
                 NaN and +inf indicate a bug in the log-posterior (use -inf to reject)."
            ),
            Error::OutsideSupport { operation, chain } => write!(
                f,
                "{operation}(): chain {chain} has log density -inf at its current point; \
                 a chain must start (and stay) inside the support."
            ),
            Error::WrongLength {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected length {expected}, got {got}."),
            Error::MissingField { sampler, field } => {
                write!(f, "{sampler} requires the tell() field `{field}`.")
            }
            Error::UnusedField { sampler, field } => write!(
                f,
                "{sampler} does not use the tell() field `{field}`; \
                 don't compute it (it would be thrown away)."
            ),
            Error::InvalidStart {
                block,
                chain,
                reason,
            } => write!(f, "x0: block `{block}`, chain {chain}: {reason}."),
            Error::InvalidSpace(reason) => write!(f, "invalid parameter space: {reason}."),
            Error::InvalidOption {
                sampler,
                option,
                reason,
            } => write!(f, "{sampler}: option `{option}` {reason}."),
        }
    }
}

impl std::error::Error for Error {}
