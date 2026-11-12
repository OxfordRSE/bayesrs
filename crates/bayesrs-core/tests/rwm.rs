//! RandomWalkMetropolis against the protocol (INTERFACE.md §2–§3), the
//! reproducibility rules (§5.5, §6) and known posteriors (§12.4).

use bayesrs_core::{Error, ParamSpace, RandomWalkMetropolis, RwmOptions, Sampler, Tell};
use ndarray::{Array2, Array3, Axis, array};

fn space_xy() -> ParamSpace {
    ParamSpace::builder().real("x").real("y").build().unwrap()
}

/// Standard normal in every coordinate.
fn std_normal(theta: ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    theta
        .rows()
        .into_iter()
        .map(|r| -0.5 * r.iter().map(|v| v * v).sum::<f64>())
        .collect()
}

fn run(
    s: &mut RandomWalkMetropolis,
    n: usize,
    logp: impl Fn(ndarray::ArrayView2<'_, f64>) -> Vec<f64>,
) {
    while s.n_draws() < n {
        let lp = logp(s.ask().unwrap());
        s.tell(Tell::new(&lp)).unwrap();
    }
}

// --- protocol ---------------------------------------------------------------

#[test]
fn first_ask_returns_x0_and_first_tell_stores_it_as_draw_0() {
    let x0 = [0.5, -1.0, 2.0, 3.0];
    let mut s = RandomWalkMetropolis::new(&space_xy(), &x0, 2, 1).unwrap();
    assert_eq!(s.n_draws(), 0);
    assert_eq!(s.ask().unwrap(), array![[0.5, -1.0], [2.0, 3.0]]);
    s.tell(Tell::new(&[-1.0, -2.0])).unwrap();
    assert_eq!(s.n_draws(), 1);
    assert_eq!(
        s.draws(),
        Array3::from_shape_vec((2, 1, 2), x0.to_vec()).unwrap()
    );
}

#[test]
fn out_of_order_calls_are_errors() {
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 2], 1, 1).unwrap();
    assert_eq!(s.tell(Tell::new(&[0.0])), Err(Error::TellWithoutAsk));
    s.ask().unwrap();
    assert_eq!(s.ask().unwrap_err(), Error::AskBeforeTell);
    assert_eq!(s.reanchor(&[0.0]), Err(Error::ReanchorWhileWaiting));
    s.tell(Tell::new(&[0.0])).unwrap();
    assert_eq!(s.tell(Tell::new(&[0.0])), Err(Error::TellWithoutAsk));
}

#[test]
fn error_messages_match_the_spec() {
    assert_eq!(
        Error::AskBeforeTell.to_string(),
        "ask() called again before tell(); tell() the results of the previous ask() first."
    );
    assert_eq!(
        Error::TellWithoutAsk.to_string(),
        "tell() called with no outstanding ask()."
    );
}

#[test]
fn neg_inf_on_first_tell_is_fatal_but_legal_afterwards() {
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 1).unwrap();
    s.ask().unwrap();
    assert_eq!(
        s.tell(Tell::new(&[0.0, f64::NEG_INFINITY])),
        Err(Error::OutsideSupport {
            operation: "tell",
            chain: 1
        })
    );
    s.tell(Tell::new(&[0.0, 0.0])).unwrap();
    let start = s.position();
    s.ask().unwrap();
    s.tell(Tell::new(&[f64::NEG_INFINITY; 2])).unwrap();
    assert_eq!(s.position(), start, "-inf proposals are always rejected");
}

#[test]
fn nan_and_pos_inf_are_fatal_and_leave_the_sampler_retryable() {
    let mut a = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 7).unwrap();
    let mut b = a.clone();
    run(&mut a, 3, std_normal);
    run(&mut b, 3, std_normal);

    let lp = std_normal(a.ask().unwrap());
    for bad in [f64::NAN, f64::INFINITY] {
        let err = a.tell(Tell::new(&[lp[0], bad])).unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidLogDensity {
                operation: "tell",
                chain: 1,
                ..
            }
        ));
    }
    // The retry after errors behaves exactly as a clean tell would.
    a.tell(Tell::new(&lp)).unwrap();
    let lp_b = std_normal(b.ask().unwrap());
    b.tell(Tell::new(&lp_b)).unwrap();
    run(&mut a, 50, std_normal);
    run(&mut b, 50, std_normal);
    assert_eq!(a.draws(), b.draws());
}

#[test]
fn wrong_length_missing_and_unused_fields_are_errors() {
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 1).unwrap();
    s.ask().unwrap();
    assert_eq!(
        s.tell(Tell::new(&[0.0])),
        Err(Error::WrongLength {
            what: "tell(logp)",
            expected: 2,
            got: 1
        })
    );
    let g = [0.0; 4];
    for (t, field) in [
        (Tell::new(&[0.0; 2]).with_grad(&g), "grad"),
        (Tell::new(&[0.0; 2]).with_log_lik(&[0.0; 2]), "log_lik"),
        (Tell::new(&[0.0; 2]).with_log_prior(&[0.0; 2]), "log_prior"),
    ] {
        assert_eq!(
            s.tell(t),
            Err(Error::UnusedField {
                sampler: "RandomWalkMetropolis",
                field
            })
        );
    }
    let mut no_logp = Tell::new(&[]);
    no_logp.logp = None;
    assert_eq!(
        s.tell(no_logp),
        Err(Error::MissingField {
            sampler: "RandomWalkMetropolis",
            field: "logp"
        })
    );
    s.tell(Tell::new(&[0.0; 2])).unwrap();
}

#[test]
fn reanchor_validates_its_input() {
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 1).unwrap();
    run(&mut s, 1, std_normal);
    assert!(matches!(s.reanchor(&[0.0]), Err(Error::WrongLength { .. })));
    assert_eq!(
        s.reanchor(&[0.0, f64::NEG_INFINITY]),
        Err(Error::OutsideSupport {
            operation: "reanchor",
            chain: 1
        })
    );
    assert!(matches!(
        s.reanchor(&[f64::NAN, 0.0]),
        Err(Error::InvalidLogDensity {
            operation: "reanchor",
            chain: 0,
            ..
        })
    ));
    s.reanchor(&[-1.0, -1.0]).unwrap();
}

#[test]
fn reanchor_changes_the_acceptance_baseline() {
    // Anchoring every chain at an absurdly high density makes every
    // subsequent (finite) proposal look terrible, so nothing is accepted.
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 3).unwrap();
    run(&mut s, 1, std_normal);
    s.reanchor(&[1e6, 1e6]).unwrap();
    let start = s.position();
    for _ in 0..20 {
        let lp = std_normal(s.ask().unwrap());
        s.tell(Tell::new(&lp)).unwrap();
    }
    assert_eq!(s.position(), start);
}

// --- construction -----------------------------------------------------------

#[test]
fn bad_construction_inputs_are_errors() {
    let sp = space_xy();
    assert!(matches!(
        RandomWalkMetropolis::new(&sp, &[0.0; 3], 2, 1),
        Err(Error::WrongLength {
            expected: 4,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        RandomWalkMetropolis::new(&sp, &[], 0, 1),
        Err(Error::InvalidOption {
            option: "n_chains",
            ..
        })
    ));
    let err = RandomWalkMetropolis::new(&sp, &[0.0, 0.0, 0.0, f64::INFINITY], 2, 1).unwrap_err();
    assert!(
        matches!(err, Error::InvalidStart { ref block, chain: 1, .. } if block == "y"),
        "{err:?}"
    );
    let mut opts = RwmOptions::default();
    opts.thin = 0;
    assert!(matches!(
        RandomWalkMetropolis::with_options(&sp, &[0.0; 4], 2, 1, opts),
        Err(Error::InvalidOption { option: "thin", .. })
    ));
}

// --- reproducibility --------------------------------------------------------

#[test]
fn same_seed_same_chains_different_seed_different_chains() {
    let make = |seed| {
        let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 6], 3, seed).unwrap();
        run(&mut s, 200, std_normal);
        s.draws()
    };
    assert_eq!(make(42), make(42));
    assert_ne!(make(42), make(43));
}

#[test]
fn chains_use_independent_streams() {
    // Identical starting points, but each chain has its own stream.
    let mut s = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 42).unwrap();
    run(&mut s, 50, std_normal);
    let d = s.draws();
    assert_ne!(d.index_axis(Axis(0), 0), d.index_axis(Axis(0), 1));
}

#[test]
fn thinned_chain_equals_unthinned_chain_subsampled() {
    let k = 5;
    let mut full = RandomWalkMetropolis::new(&space_xy(), &[0.0; 4], 2, 9).unwrap();
    let mut opts = RwmOptions::default();
    opts.thin = k;
    let mut thin = RandomWalkMetropolis::with_options(&space_xy(), &[0.0; 4], 2, 9, opts).unwrap();
    run(&mut full, 1 + 40 * k, std_normal);
    run(&mut thin, 41, std_normal);
    let expected = full.draws().slice(ndarray::s![.., ..;k, ..]).to_owned();
    assert_eq!(thin.draws(), expected);
    assert_eq!(thin.n_draws(), 41);
}

// --- statistical correctness ------------------------------------------------

fn pooled_mean_cov(d: &Array3<f64>, burn: usize) -> (Vec<f64>, Array2<f64>) {
    let kept = d.slice(ndarray::s![.., burn.., ..]);
    let dim = d.shape()[2];
    let flat = kept
        .to_shape((kept.shape()[0] * kept.shape()[1], dim))
        .unwrap();
    let mean = flat.mean_axis(Axis(0)).unwrap();
    let centred = &flat - &mean;
    let cov = centred.t().dot(&centred) / (flat.nrows() as f64 - 1.0);
    (mean.to_vec(), cov)
}

#[test]
fn recovers_a_correlated_gaussian() {
    let mu = [1.0, -2.0];
    let sigma = array![[1.0, 0.8], [0.8, 2.0]];
    // Precision matrix of `sigma`.
    let det = 1.0 * 2.0 - 0.8 * 0.8;
    let prec = array![[2.0 / det, -0.8 / det], [-0.8 / det, 1.0 / det]];
    let log_post = |theta: ndarray::ArrayView2<'_, f64>| -> Vec<f64> {
        theta
            .rows()
            .into_iter()
            .map(|r| {
                let dx = [r[0] - mu[0], r[1] - mu[1]];
                let q = dx[0] * (prec[[0, 0]] * dx[0] + prec[[0, 1]] * dx[1])
                    + dx[1] * (prec[[1, 0]] * dx[0] + prec[[1, 1]] * dx[1]);
                -0.5 * q
            })
            .collect()
    };

    let mut opts = RwmOptions::default();
    opts.proposal_cov = Some(&sigma * (2.38_f64.powi(2) / 2.0));
    let x0 = [0.0, 0.0, 3.0, -4.0, -1.0, 0.0, 2.0, -3.0];
    let mut s = RandomWalkMetropolis::with_options(&space_xy(), &x0, 4, 2026, opts).unwrap();
    run(&mut s, 20_000, log_post);

    let (mean, cov) = pooled_mean_cov(&s.draws(), 2_000);
    for i in 0..2 {
        assert!((mean[i] - mu[i]).abs() < 0.1, "mean[{i}] = {}", mean[i]);
        for j in 0..2 {
            let rel = (cov[[i, j]] - sigma[[i, j]]).abs() / sigma[[i, j]];
            assert!(rel < 0.1, "cov[{i},{j}] = {}", cov[[i, j]]);
        }
    }
    for rate in s.stats().acceptance_rate {
        assert!((0.2..0.5).contains(&rate), "acceptance rate {rate}");
    }
}

#[test]
fn default_proposal_samples_a_standard_normal_and_respects_neg_inf_support() {
    // Half-normal via -inf outside x > 0: the mean is sqrt(2/pi).
    let space = ParamSpace::builder().real("x").build().unwrap();
    let mut s = RandomWalkMetropolis::new(&space, &[0.5, 1.0, 1.5, 2.0], 4, 11).unwrap();
    run(&mut s, 20_000, |theta| {
        theta
            .rows()
            .into_iter()
            .map(|r| {
                if r[0] > 0.0 {
                    -0.5 * r[0] * r[0]
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    });
    let d = s.draws();
    assert!(d.iter().all(|&v| v > 0.0));
    let (mean, _) = pooled_mean_cov(&d, 1_000);
    let expected = (2.0 / std::f64::consts::PI).sqrt();
    assert!((mean[0] - expected).abs() < 0.05, "mean = {}", mean[0]);
}
