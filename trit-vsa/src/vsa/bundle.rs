//! Bundle operation (superposition via majority voting).
//!
//! Bundling combines multiple vectors into one that is similar to all inputs.
//! This is the "addition" operation in hyperdimensional computing.

use crate::packed::PackedTritVec;
use crate::trit::Trit;

/// Compute the majority trit from a slice of trits.
///
/// Returns the trit that appears most frequently. Ties resolve to zero.
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, vsa::majority_trit};
///
/// assert_eq!(majority_trit(&[Trit::P, Trit::P, Trit::N]), Trit::P);
/// assert_eq!(majority_trit(&[Trit::P, Trit::N]), Trit::Z);  // Tie -> Z
/// assert_eq!(majority_trit(&[Trit::N, Trit::N, Trit::N]), Trit::N);
/// ```
#[must_use]
pub fn majority_trit(trits: &[Trit]) -> Trit {
    if trits.is_empty() {
        return Trit::Z;
    }

    let sum: i32 = trits.iter().map(|t| t.value() as i32).sum();

    match sum.cmp(&0) {
        std::cmp::Ordering::Greater => Trit::P,
        std::cmp::Ordering::Less => Trit::N,
        std::cmp::Ordering::Equal => Trit::Z,
    }
}

/// Bundle two vectors using element-wise majority.
///
/// The result is similar to both input vectors.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::bundle};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// a.set(1, Trit::P);
/// b.set(0, Trit::P);
/// b.set(1, Trit::N);
///
/// let result = bundle(&a, &b);
/// assert_eq!(result.get(0), Trit::P);  // Both positive -> P
/// assert_eq!(result.get(1), Trit::Z);  // Tie -> Z
/// ```
#[must_use]
pub fn bundle(a: &PackedTritVec, b: &PackedTritVec) -> PackedTritVec {
    assert_eq!(a.len(), b.len(), "vectors must have same dimensions");
    a.add_clamped(b)
}

/// Bundle multiple vectors using element-wise majority voting.
///
/// For each dimension, counts positive and negative votes and returns
/// the majority. Ties resolve to zero.
///
/// # Panics
///
/// Panics if vectors is empty or if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::bundle_many};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
/// let mut c = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// b.set(0, Trit::P);
/// c.set(0, Trit::N);
///
/// let vectors: Vec<&PackedTritVec> = vec![&a, &b, &c];
/// let result = bundle_many(&vectors);
/// assert_eq!(result.get(0), Trit::P);  // 2 positive, 1 negative -> P
/// ```
#[must_use]
pub fn bundle_many(vectors: &[&PackedTritVec]) -> PackedTritVec {
    assert!(!vectors.is_empty(), "cannot bundle empty vector list");

    let num_dims = vectors[0].len();
    for v in vectors.iter().skip(1) {
        assert_eq!(v.len(), num_dims, "all vectors must have same dimensions");
    }

    let mut result = PackedTritVec::new(num_dims);

    for dim in 0..num_dims {
        let trits: Vec<Trit> = vectors.iter().map(|v| v.get(dim)).collect();
        result.set(dim, majority_trit(&trits));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_trit_basic() {
        assert_eq!(majority_trit(&[Trit::P, Trit::P, Trit::P]), Trit::P);
        assert_eq!(majority_trit(&[Trit::N, Trit::N, Trit::N]), Trit::N);
        assert_eq!(majority_trit(&[Trit::Z, Trit::Z, Trit::Z]), Trit::Z);
    }

    #[test]
    fn test_majority_trit_mixed() {
        assert_eq!(majority_trit(&[Trit::P, Trit::P, Trit::N]), Trit::P);
        assert_eq!(majority_trit(&[Trit::N, Trit::N, Trit::P]), Trit::N);
        assert_eq!(majority_trit(&[Trit::P, Trit::N, Trit::Z]), Trit::Z);
    }

    #[test]
    fn test_majority_trit_tie() {
        assert_eq!(majority_trit(&[Trit::P, Trit::N]), Trit::Z);
        assert_eq!(
            majority_trit(&[Trit::P, Trit::N, Trit::P, Trit::N]),
            Trit::Z
        );
    }

    #[test]
    fn test_majority_trit_empty() {
        assert_eq!(majority_trit(&[]), Trit::Z);
    }

    #[test]
    fn test_bundle_two() {
        let mut a = PackedTritVec::new(4);
        let mut b = PackedTritVec::new(4);

        a.set(0, Trit::P);
        a.set(1, Trit::P);
        a.set(2, Trit::N);

        b.set(0, Trit::P);
        b.set(1, Trit::N);
        b.set(2, Trit::N);

        let result = bundle(&a, &b);

        assert_eq!(result.get(0), Trit::P); // P + P -> P
        assert_eq!(result.get(1), Trit::Z); // P + N -> tie -> Z
        assert_eq!(result.get(2), Trit::N); // N + N -> N
        assert_eq!(result.get(3), Trit::Z); // Z + Z -> Z
    }

    #[test]
    fn test_bundle_many_three() {
        let mut a = PackedTritVec::new(3);
        let mut b = PackedTritVec::new(3);
        let mut c = PackedTritVec::new(3);

        // Dimension 0: P, P, N -> P (2 vs 1)
        a.set(0, Trit::P);
        b.set(0, Trit::P);
        c.set(0, Trit::N);

        // Dimension 1: P, N, Z -> tie -> Z
        a.set(1, Trit::P);
        b.set(1, Trit::N);

        // Dimension 2: N, N, N -> N
        a.set(2, Trit::N);
        b.set(2, Trit::N);
        c.set(2, Trit::N);

        let vectors: Vec<&PackedTritVec> = vec![&a, &b, &c];
        let result = bundle_many(&vectors);

        assert_eq!(result.get(0), Trit::P);
        assert_eq!(result.get(1), Trit::Z);
        assert_eq!(result.get(2), Trit::N);
    }
}
