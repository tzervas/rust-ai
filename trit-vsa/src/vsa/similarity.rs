//! Similarity measures for ternary vectors.
//!
//! These functions measure how similar two vectors are, used for retrieval
//! and associative memory operations in VSA.

use crate::packed::PackedTritVec;
use crate::sparse::SparseVec;

/// Compute cosine similarity between two packed vectors.
///
/// Returns a value in [-1, 1] where:
/// - 1.0 = identical
/// - 0.0 = orthogonal
/// - -1.0 = opposite
///
/// # Formula
///
/// ```text
/// cos(a, b) = (a · b) / (||a|| * ||b||)
/// ```
///
/// For ternary vectors, the norms are `sqrt(count_nonzero)`.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::cosine_similarity};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// a.set(1, Trit::P);
/// b.set(0, Trit::P);
/// b.set(1, Trit::P);
///
/// // Identical non-zero parts -> similarity = 1.0
/// let sim = cosine_similarity(&a, &b);
/// assert!((sim - 1.0).abs() < 0.001);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cosine_similarity(a: &PackedTritVec, b: &PackedTritVec) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same dimensions");

    let dot = a.dot(b) as f32;
    let norm_a = (a.count_nonzero() as f32).sqrt();
    let norm_b = (b.count_nonzero() as f32).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine similarity between two sparse vectors.
///
/// More efficient than converting to packed when vectors are very sparse.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn cosine_similarity_sparse(a: &SparseVec, b: &SparseVec) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same dimensions");

    let dot = a.dot(b) as f32;
    let norm_a = (a.count_nonzero() as f32).sqrt();
    let norm_b = (b.count_nonzero() as f32).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute Hamming distance between two packed vectors.
///
/// Counts the number of positions where the vectors differ.
///
/// # Panics
///
/// Panics if vectors have different dimensions.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit, vsa::hamming_distance};
///
/// let mut a = PackedTritVec::new(4);
/// let mut b = PackedTritVec::new(4);
///
/// a.set(0, Trit::P);
/// a.set(1, Trit::N);
/// b.set(0, Trit::P);
/// b.set(1, Trit::P);  // Different from a
///
/// assert_eq!(hamming_distance(&a, &b), 1);
/// ```
#[must_use]
pub fn hamming_distance(a: &PackedTritVec, b: &PackedTritVec) -> usize {
    assert_eq!(a.len(), b.len(), "vectors must have same dimensions");

    let mut distance = 0;

    for i in 0..a.num_words() {
        // Count positions where planes differ
        let plus_diff = a.plus_plane()[i] ^ b.plus_plane()[i];
        let minus_diff = a.minus_plane()[i] ^ b.minus_plane()[i];

        // A position differs if either plane differs
        // But we need to be careful: (P,Z), (Z,N), (P,N) all count as different
        // XOR on plus_plane catches P<->Z and P<->N (partially)
        // XOR on minus_plane catches N<->Z and P<->N (partially)
        // We need: positions where (a_plus, a_minus) != (b_plus, b_minus)
        distance += (plus_diff | minus_diff).count_ones() as usize;
    }

    // Adjust for padding bits
    let total_bits = a.num_words() * 32;
    let padding = total_bits - a.len();
    if padding > 0 {
        // Subtract any differences in padding region
        let last_word = a.num_words() - 1;
        let padding_mask = !0u32 << (32 - padding);
        let plus_diff = a.plus_plane()[last_word] ^ b.plus_plane()[last_word];
        let minus_diff = a.minus_plane()[last_word] ^ b.minus_plane()[last_word];
        distance -= ((plus_diff | minus_diff) & padding_mask).count_ones() as usize;
    }

    distance
}

/// Compute normalized Hamming similarity.
///
/// Returns a value in [0, 1] where:
/// - 1.0 = identical
/// - 0.0 = completely different
///
/// # Formula
///
/// ```text
/// similarity = 1 - (hamming_distance / num_dims)
/// ```
///
/// # Panics
///
/// Panics if vectors have different dimensions.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn hamming_similarity(a: &PackedTritVec, b: &PackedTritVec) -> f32 {
    if a.is_empty() {
        return 1.0;
    }
    1.0 - (hamming_distance(a, b) as f32 / a.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit::Trit;

    #[test]
    fn test_cosine_identical() {
        let mut a = PackedTritVec::new(64);

        for i in 0..32 {
            a.set(i, Trit::P);
        }
        for i in 32..48 {
            a.set(i, Trit::N);
        }

        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_opposite() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        for i in 0..32 {
            a.set(i, Trit::P);
            b.set(i, Trit::N);
        }

        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 0.001,
            "opposite vectors should have similarity -1.0"
        );
    }

    #[test]
    fn test_cosine_orthogonal() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        // a has +1 in positions 0-15
        for i in 0..16 {
            a.set(i, Trit::P);
        }

        // b has +1 in positions 16-31 (no overlap)
        for i in 16..32 {
            b.set(i, Trit::P);
        }

        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 0.001,
            "orthogonal vectors should have similarity ~0"
        );
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);
        b.set(0, Trit::P);

        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should have similarity 0");
    }

    #[test]
    fn test_hamming_identical() {
        let mut a = PackedTritVec::new(64);
        a.set(0, Trit::P);
        a.set(10, Trit::N);

        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_one_diff() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        a.set(0, Trit::P);
        b.set(0, Trit::N);

        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_multiple_diff() {
        let mut a = PackedTritVec::new(10);
        let mut b = PackedTritVec::new(10);

        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(2, Trit::Z);

        b.set(0, Trit::N); // Different
        b.set(1, Trit::N); // Same
        b.set(2, Trit::P); // Different

        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_hamming_similarity() {
        let a = PackedTritVec::new(100);
        let b = PackedTritVec::new(100);

        // Identical (all zeros)
        let sim = hamming_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_sparse() {
        let mut a = SparseVec::new(1000);
        let mut b = SparseVec::new(1000);

        a.set(0, Trit::P);
        a.set(1, Trit::P);
        b.set(0, Trit::P);
        b.set(1, Trit::P);

        let sim = cosine_similarity_sparse(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }
}
