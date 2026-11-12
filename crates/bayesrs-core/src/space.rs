//! The parameter space: names, kinds and shapes, declared once
//! (INTERFACE.md §4).
//!
//! Only the `Real` kind exists so far; further kinds slot in as new
//! [`Kind`] variants with their own [`Transform`] behaviour.

use crate::Error;

/// The free ↔ constrained map of one kind of parameter (INTERFACE.md §4.2).
///
/// Slices hold one parameter block's coordinates, flattened row-major.
pub trait Transform {
    /// Number of free (unconstrained) coordinates.
    fn d_free(&self) -> usize;
    /// Number of constrained (user-visible) coordinates.
    fn d_constr(&self) -> usize;
    /// Maps free coordinates to constrained ones.
    fn to_constrained(&self, free: &[f64], out: &mut [f64]);
    /// Maps constrained coordinates to free ones, validating them.
    fn to_free(&self, constr: &[f64], out: &mut [f64]) -> Result<(), String>;
    /// `log |det J|` of the free → constrained map at `free`.
    fn log_abs_det_jacobian(&self, free: &[f64]) -> f64;
}

/// What kind of quantity a parameter block is.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Kind {
    /// Unconstrained real numbers (identity transform).
    Real,
}

impl Kind {
    /// The kind's name as it appears in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Real => "Real",
        }
    }
}

/// One declared parameter: a name, a kind and a shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    name: String,
    kind: Kind,
    shape: Vec<usize>,
    size: usize,
}

impl Block {
    /// The parameter's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The parameter's kind.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// The parameter's shape (`[]` for a scalar).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Transform for Block {
    fn d_free(&self) -> usize {
        match self.kind {
            Kind::Real => self.size,
        }
    }

    fn d_constr(&self) -> usize {
        match self.kind {
            Kind::Real => self.size,
        }
    }

    fn to_constrained(&self, free: &[f64], out: &mut [f64]) {
        match self.kind {
            Kind::Real => out.copy_from_slice(free),
        }
    }

    fn to_free(&self, constr: &[f64], out: &mut [f64]) -> Result<(), String> {
        match self.kind {
            Kind::Real => {
                if let Some(v) = constr.iter().find(|v| !v.is_finite()) {
                    return Err(format!("Real values must be finite, got {v}"));
                }
                out.copy_from_slice(constr);
                Ok(())
            }
        }
    }

    fn log_abs_det_jacobian(&self, _free: &[f64]) -> f64 {
        match self.kind {
            Kind::Real => 0.0,
        }
    }
}

/// The declared parameters of a model, in declaration order.
///
/// Declaration order fixes the column order of every array in the system.
///
/// ```
/// use bayesrs_core::ParamSpace;
///
/// let space = ParamSpace::builder()
///     .real("alpha")
///     .real_shaped("beta", &[2])
///     .build()?;
/// assert_eq!(space.d_constr(), 3);
/// assert_eq!(space.names(), ["alpha", "beta[0]", "beta[1]"]);
/// # Ok::<(), bayesrs_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpace {
    blocks: Vec<Block>,
    d_free: usize,
    d_constr: usize,
}

impl ParamSpace {
    /// Starts declaring a parameter space.
    pub fn builder() -> ParamSpaceBuilder {
        ParamSpaceBuilder::default()
    }

    /// The declared blocks, in order.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Length of a free-space point.
    pub fn d_free(&self) -> usize {
        self.d_free
    }

    /// Length of a constrained-space point (one row of an `ask` batch).
    pub fn d_constr(&self) -> usize {
        self.d_constr
    }

    /// Flattened per-column names in declaration order, 0-based
    /// (`"mu[0]"`, `"Sigma[1,2]"`; scalars are just their name).
    pub fn names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.d_constr);
        for b in &self.blocks {
            if b.shape.is_empty() {
                names.push(b.name.clone());
                continue;
            }
            for flat in 0..b.size {
                let mut idx = vec![0; b.shape.len()];
                let mut rem = flat;
                for (i, &dim) in b.shape.iter().enumerate().rev() {
                    idx[i] = rem % dim;
                    rem /= dim;
                }
                let idx: Vec<String> = idx.iter().map(usize::to_string).collect();
                names.push(format!("{}[{}]", b.name, idx.join(",")));
            }
        }
        names
    }

    /// The constrained-space `(offset, len)` of the named block — `unpack`
    /// for callers indexing flat rows by hand.
    pub fn offset(&self, name: &str) -> Option<(usize, usize)> {
        let mut off = 0;
        for b in &self.blocks {
            if b.name == name {
                return Some((off, b.d_constr()));
            }
            off += b.d_constr();
        }
        None
    }

    /// Maps one constrained point (`d_constr`) to free space (`d_free`).
    pub fn to_free(&self, constr: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.check_len("to_free input", constr.len(), self.d_constr)?;
        self.check_len("to_free output", out.len(), self.d_free)?;
        let (mut c, mut f) = (0, 0);
        for b in &self.blocks {
            let (dc, df) = (b.d_constr(), b.d_free());
            b.to_free(&constr[c..c + dc], &mut out[f..f + df])
                .map_err(|reason| Error::InvalidStart {
                    block: b.name.clone(),
                    chain: 0,
                    reason,
                })?;
            c += dc;
            f += df;
        }
        Ok(())
    }

    /// Maps one free point (`d_free`) to constrained space (`d_constr`).
    ///
    /// # Panics
    /// If either slice has the wrong length.
    pub fn to_constrained(&self, free: &[f64], out: &mut [f64]) {
        assert_eq!(free.len(), self.d_free, "to_constrained: free length");
        assert_eq!(out.len(), self.d_constr, "to_constrained: out length");
        let (mut c, mut f) = (0, 0);
        for b in &self.blocks {
            let (dc, df) = (b.d_constr(), b.d_free());
            b.to_constrained(&free[f..f + df], &mut out[c..c + dc]);
            c += dc;
            f += df;
        }
    }

    /// `log |det J|` of the free → constrained map at one free point.
    ///
    /// # Panics
    /// If `free` has the wrong length.
    pub fn log_abs_det_jacobian(&self, free: &[f64]) -> f64 {
        assert_eq!(free.len(), self.d_free, "log_abs_det_jacobian: length");
        let mut f = 0;
        let mut total = 0.0;
        for b in &self.blocks {
            let df = b.d_free();
            total += b.log_abs_det_jacobian(&free[f..f + df]);
            f += df;
        }
        total
    }

    fn check_len(&self, what: &'static str, got: usize, expected: usize) -> Result<(), Error> {
        if got == expected {
            Ok(())
        } else {
            Err(Error::WrongLength {
                what,
                expected,
                got,
            })
        }
    }
}

/// Declares a [`ParamSpace`] one block at a time; errors surface at
/// [`build`](ParamSpaceBuilder::build).
#[derive(Debug, Default, Clone)]
pub struct ParamSpaceBuilder {
    blocks: Vec<(String, Kind, Vec<usize>)>,
}

impl ParamSpaceBuilder {
    /// Adds a scalar unconstrained real parameter.
    pub fn real(self, name: &str) -> Self {
        self.real_shaped(name, &[])
    }

    /// Adds an unconstrained real parameter of the given shape
    /// (`&[]` for a scalar, `&[3]` for a vector, `&[2, 2]` for a matrix).
    pub fn real_shaped(mut self, name: &str, shape: &[usize]) -> Self {
        self.blocks
            .push((name.to_owned(), Kind::Real, shape.to_vec()));
        self
    }

    /// Validates the declaration and builds the space.
    pub fn build(self) -> Result<ParamSpace, Error> {
        if self.blocks.is_empty() {
            return Err(Error::InvalidSpace("no parameters declared".into()));
        }
        let mut blocks: Vec<Block> = Vec::with_capacity(self.blocks.len());
        for (name, kind, shape) in self.blocks {
            if !is_identifier(&name) {
                return Err(Error::InvalidSpace(format!(
                    "`{name}` is not a valid identifier (letters, digits, underscores; \
                     not starting with a digit)"
                )));
            }
            if blocks.iter().any(|b| b.name == name) {
                return Err(Error::InvalidSpace(format!("`{name}` is declared twice")));
            }
            let size = shape.iter().product();
            if size == 0 {
                return Err(Error::InvalidSpace(format!(
                    "`{name}` has shape {shape:?}, which holds no values"
                )));
            }
            blocks.push(Block {
                name,
                kind,
                shape,
                size,
            });
        }
        let d_free = blocks.iter().map(Transform::d_free).sum();
        let d_constr = blocks.iter().map(Transform::d_constr).sum();
        Ok(ParamSpace {
            blocks,
            d_free,
            d_constr,
        })
    }
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> ParamSpace {
        ParamSpace::builder()
            .real("alpha")
            .real_shaped("beta", &[2, 3])
            .build()
            .unwrap()
    }

    #[test]
    fn dimensions_and_names_follow_declaration_order() {
        let s = space();
        assert_eq!((s.d_free(), s.d_constr()), (7, 7));
        let names = s.names();
        assert_eq!(names[0], "alpha");
        assert_eq!(names[1], "beta[0,0]");
        assert_eq!(names[3], "beta[0,2]");
        assert_eq!(names[4], "beta[1,0]");
        assert_eq!(s.offset("beta"), Some((1, 6)));
        assert_eq!(s.offset("gamma"), None);
    }

    #[test]
    fn real_transform_round_trips_with_zero_jacobian() {
        let s = space();
        let x: Vec<f64> = (0..7).map(|i| i as f64 - 3.5).collect();
        let mut z = vec![0.0; 7];
        let mut back = vec![0.0; 7];
        s.to_free(&x, &mut z).unwrap();
        s.to_constrained(&z, &mut back);
        assert_eq!(x, back);
        assert_eq!(s.log_abs_det_jacobian(&z), 0.0);
    }

    #[test]
    fn non_finite_real_is_rejected() {
        let s = space();
        let mut x = vec![0.0; 7];
        x[2] = f64::NAN;
        let err = s.to_free(&x, &mut [0.0; 7]).unwrap_err();
        assert!(matches!(err, Error::InvalidStart { ref block, .. } if block == "beta"));
    }

    #[test]
    fn invalid_declarations_are_rejected() {
        let cases = [
            ParamSpace::builder().build(),
            ParamSpace::builder().real("a").real("a").build(),
            ParamSpace::builder().real("1a").build(),
            ParamSpace::builder().real("a-b").build(),
            ParamSpace::builder().real_shaped("a", &[3, 0]).build(),
        ];
        for case in cases {
            assert!(matches!(case, Err(Error::InvalidSpace(_))), "{case:?}");
        }
    }
}
