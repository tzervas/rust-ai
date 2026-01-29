//! Bind operation (composition via subtraction mod 3).
//!
//! Binding creates associations between vectors. Use `unbind` to recover
//! the original: `unbind(bind(a, b), b) == a`.

use crate::packed::PackedTritVec;

/// Bind two vectors element-wise.
///
/// This operation uses ternary subtraction mod 3 for each element.
/// Use `unbind` to recover the original: `unbind(bind(a, b), b) == a`.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::{bind, unbind}};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// a.set(1, Trit::N);
/// b.set(0, Trit::P);
/// b.set(1, Trit::P);
///
/// let bound = bind(&a, &b);
///
/// // Use unbind to recover the original
/// let recovered = unbind(&bound, &b);
/// assert_eq!(recovered.get(0), a.get(0));
/// assert_eq!(recovered.get(1), a.get(1));
/// ```
#[must_use]
pub fn bind(a: &PackedTritVec, b: &PackedTritVec) -> PackedTritVec {
    assert_eq!(a.len(), b.len(), "vectors must have same dimensions");

    let mut result = PackedTritVec::new(a.len());

    for i in 0..a.len() {
        let bound = a.get(i).bind(b.get(i));
        result.set(i, bound);
    }

    result
}

/// Unbind a vector - the inverse of bind.
///
/// This recovers the original vector: `unbind(bind(a, b), b) == a`.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::{bind, unbind}};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// b.set(0, Trit::N);
///
/// let bound = bind(&a, &b);
/// let recovered = unbind(&bound, &b);
///
/// assert_eq!(recovered.get(0), a.get(0));
/// ```
#[must_use]
pub fn unbind(bound: &PackedTritVec, key: &PackedTritVec) -> PackedTritVec {
    assert_eq!(bound.len(), key.len(), "vectors must have same dimensions");

    let mut result = PackedTritVec::new(bound.len());

    for i in 0..bound.len() {
        let unbound = bound.get(i).unbind(key.get(i));
        result.set(i, unbound);
    }

    result
}

/// Bind multiple vectors sequentially.
///
/// Computes `bind(bind(bind(v[0], v[1]), v[2]), ...)`.
///
/// # Panics
///
/// Panics if vectors is empty or if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::bind_many};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
/// let mut c = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// b.set(0, Trit::N);
/// c.set(0, Trit::Z);
///
/// let vectors: Vec<&PackedTritVec> = vec![&a, &b, &c];
/// let result = bind_many(&vectors);
/// ```
#[must_use]
pub fn bind_many(vectors: &[&PackedTritVec]) -> PackedTritVec {
    assert!(!vectors.is_empty(), "cannot bind empty vector list");

    let mut result = vectors[0].clone();

    for v in vectors.iter().skip(1) {
        result = bind(&result, v);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit::Trit;

    #[test]
    fn test_bind_unbind_inverse() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        // Set some values
        for i in 0..64 {
            let trit = match i % 3 {
                0 => Trit::P,
                1 => Trit::N,
                _ => Trit::Z,
            };
            a.set(i, trit);
        }

        for i in 0..64 {
            let trit = match i % 5 {
                0 => Trit::P,
                1 | 2 => Trit::N,
                _ => Trit::Z,
            };
            b.set(i, trit);
        }

        // unbind(bind(a, b), b) == a
        let bound = bind(&a, &b);
        let recovered = unbind(&bound, &b);

        for i in 0..64 {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at dimension {i}");
        }
    }

    #[test]
    fn test_bind_with_zero() {
        let mut a = PackedTritVec::new(4);
        let zero = PackedTritVec::new(4);

        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(2, Trit::Z);

        // Binding with zero should return the original
        let result = bind(&a, &zero);

        assert_eq!(result.get(0), Trit::P);
        assert_eq!(result.get(1), Trit::N);
        assert_eq!(result.get(2), Trit::Z);
    }

    #[test]
    fn test_bind_truth_table() {
        // Verify the truth table for bind
        // bind(a, b) = (a - b) mod 3

        let mut a = PackedTritVec::new(9);
        let mut b = PackedTritVec::new(9);

        // All combinations of (a, b)
        let combinations = [
            (Trit::N, Trit::N),
            (Trit::N, Trit::Z),
            (Trit::N, Trit::P),
            (Trit::Z, Trit::N),
            (Trit::Z, Trit::Z),
            (Trit::Z, Trit::P),
            (Trit::P, Trit::N),
            (Trit::P, Trit::Z),
            (Trit::P, Trit::P),
        ];

        for (i, &(ta, tb)) in combinations.iter().enumerate() {
            a.set(i, ta);
            b.set(i, tb);
        }

        let result = bind(&a, &b);

        // Expected results based on (a - b) mod 3
        let expected = [
            Trit::Z, // N - N = 0
            Trit::N, // N - Z = -1
            Trit::P, // N - P = -2 mod 3 = 1
            Trit::P, // Z - N = 1
            Trit::Z, // Z - Z = 0
            Trit::N, // Z - P = -1
            Trit::N, // P - N = 2 mod 3 = -1
            Trit::P, // P - Z = 1
            Trit::Z, // P - P = 0
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                result.get(i),
                exp,
                "bind({}, {}) should be {}",
                combinations[i].0,
                combinations[i].1,
                exp
            );
        }
    }

    #[test]
    fn test_unbind() {
        let mut a = PackedTritVec::new(4);
        let mut b = PackedTritVec::new(4);

        a.set(0, Trit::P);
        a.set(1, Trit::N);
        b.set(0, Trit::N);
        b.set(1, Trit::P);

        let bound = bind(&a, &b);
        let recovered = unbind(&bound, &b);

        assert_eq!(recovered.get(0), a.get(0));
        assert_eq!(recovered.get(1), a.get(1));
    }

    #[test]
    fn test_bind_many() {
        let mut a = PackedTritVec::new(4);
        let mut b = PackedTritVec::new(4);
        let mut c = PackedTritVec::new(4);

        a.set(0, Trit::P);
        b.set(0, Trit::N);
        c.set(0, Trit::Z);

        let vectors: Vec<&PackedTritVec> = vec![&a, &b, &c];
        let result = bind_many(&vectors);

        // Should be equivalent to bind(bind(a, b), c)
        let expected = bind(&bind(&a, &b), &c);

        for i in 0..4 {
            assert_eq!(result.get(i), expected.get(i));
        }
    }
}
