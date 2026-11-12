//! Deterministic random-number streams (INTERFACE.md §5.5).
//!
//! Every random number in bayesrs comes from a ChaCha8 stream identified by
//! `(seed, stream_id)`:
//!
//! * the 256-bit ChaCha key is four consecutive SplitMix64 outputs, starting
//!   from state `seed`, written little-endian;
//! * the ChaCha stream (nonce) is `stream_id`.
//!
//! Chains use stream ids `0..n_chains`; higher ids are reserved for
//! sampler-level randomness. The exact constants are provisional until the
//! first cross-language reproducibility test passes (INTERFACE.md §13).

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// One step of SplitMix64 (Steele, Lea & Flood 2014): advances `state` and
/// returns the next output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The ChaCha8 stream for `(seed, stream_id)`.
pub fn stream(seed: u64, stream_id: u64) -> ChaCha8Rng {
    let mut state = seed;
    let mut key = [0u8; 32];
    for chunk in key.as_chunks_mut::<8>().0 {
        chunk.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    let mut rng = ChaCha8Rng::from_seed(key);
    rng.set_stream(stream_id);
    rng
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn splitmix64_matches_reference_values() {
        // Reference outputs for seed 1234567 from the published algorithm.
        let mut s = 1234567;
        assert_eq!(splitmix64(&mut s), 6457827717110365317);
        assert_eq!(splitmix64(&mut s), 3203168211198807973);
    }

    #[test]
    fn streams_are_deterministic_and_distinct() {
        let a: Vec<u64> = (0..4).map(|_| stream(42, 0).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        assert_ne!(stream(42, 0).next_u64(), stream(42, 1).next_u64());
        assert_ne!(stream(42, 0).next_u64(), stream(43, 0).next_u64());
    }
}
