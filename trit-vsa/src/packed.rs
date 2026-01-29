//! Bitsliced packed ternary vector storage.
//!
//! This module provides `PackedTritVec`, an efficient representation of ternary
//! vectors using bitsliced storage. Each trit is stored as two bits across
//! separate "plus" and "minus" planes.
//!
//! ## Representation
//!
//! ```text
//! Value | +plane | -plane
//! ------+--------+-------
//!   +1  |   1    |   0
//!    0  |   0    |   0
//!   -1  |   0    |   1
//! ```
//!
//! This representation enables efficient popcount-based operations like dot
//! products and similarity computations.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{Result, TernaryError};
use crate::trit::Trit;

/// A packed ternary vector using bitsliced storage.
///
/// The vector is stored in two separate planes:
/// - `plus`: bits set where value is +1
/// - `minus`: bits set where value is -1
///
/// This representation enables O(n/32) dot products via popcount.
///
/// # Examples
///
/// ```
/// use trit_vsa::{PackedTritVec, Trit};
///
/// let mut vec = PackedTritVec::new(100);
/// vec.set(0, Trit::P);
/// vec.set(1, Trit::N);
/// vec.set(50, Trit::P);
///
/// assert_eq!(vec.get(0), Trit::P);
/// assert_eq!(vec.get(1), Trit::N);
/// assert_eq!(vec.get(2), Trit::Z);  // Default is zero
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct PackedTritVec {
    /// Positive plane: bit set if value is +1.
    plus: Vec<u32>,
    /// Negative plane: bit set if value is -1.
    minus: Vec<u32>,
    /// Number of logical dimensions.
    num_dims: usize,
}

impl PackedTritVec {
    /// Create a new packed vector with given dimension count.
    ///
    /// All values are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `num_dims` - Number of logical dimensions (trit count)
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::PackedTritVec;
    ///
    /// let vec = PackedTritVec::new(1000);
    /// assert_eq!(vec.len(), 1000);
    /// ```
    #[must_use]
    pub fn new(num_dims: usize) -> Self {
        let num_words = num_dims.div_ceil(32);
        Self {
            plus: vec![0u32; num_words],
            minus: vec![0u32; num_words],
            num_dims,
        }
    }

    /// Create from existing planes.
    ///
    /// # Arguments
    ///
    /// * `plus` - Positive plane words
    /// * `minus` - Negative plane words
    /// * `num_dims` - Logical dimension count
    ///
    /// # Errors
    ///
    /// Returns error if planes have wrong size for `num_dims`.
    pub fn from_planes(plus: Vec<u32>, minus: Vec<u32>, num_dims: usize) -> Result<Self> {
        let expected_words = num_dims.div_ceil(32);
        if plus.len() != expected_words || minus.len() != expected_words {
            return Err(TernaryError::DimensionMismatch {
                expected: expected_words,
                actual: plus.len().max(minus.len()),
            });
        }
        Ok(Self {
            plus,
            minus,
            num_dims,
        })
    }

    /// Create from a slice of trits.
    ///
    /// # Arguments
    ///
    /// * `trits` - Slice of trit values
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::{PackedTritVec, Trit};
    ///
    /// let trits = [Trit::P, Trit::N, Trit::Z, Trit::P];
    /// let vec = PackedTritVec::from_trits(&trits);
    /// assert_eq!(vec.len(), 4);
    /// assert_eq!(vec.get(0), Trit::P);
    /// assert_eq!(vec.get(1), Trit::N);
    /// ```
    #[must_use]
    pub fn from_trits(trits: &[Trit]) -> Self {
        let mut vec = Self::new(trits.len());
        for (i, &trit) in trits.iter().enumerate() {
            vec.set(i, trit);
        }
        vec
    }

    /// Create from a slice of integer values (-1, 0, +1).
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of i8 values (must be -1, 0, or +1)
    ///
    /// # Errors
    ///
    /// Returns error if any value is not -1, 0, or +1.
    pub fn from_i8_slice(values: &[i8]) -> Result<Self> {
        let mut vec = Self::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let trit = Trit::from_value(v as i32)?;
            vec.set(i, trit);
        }
        Ok(vec)
    }

    /// Get the number of logical dimensions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.num_dims
    }

    /// Check if the vector is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.num_dims == 0
    }

    /// Get the number of u32 words used for each plane.
    #[must_use]
    pub fn num_words(&self) -> usize {
        self.plus.len()
    }

    /// Set a dimension to a trit value.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension index (0-based)
    /// * `value` - Trit value to set
    ///
    /// # Panics
    ///
    /// Panics if `dim >= len()`.
    pub fn set(&mut self, dim: usize, value: Trit) {
        assert!(dim < self.num_dims, "dimension out of bounds");

        let word_idx = dim / 32;
        let bit_idx = dim % 32;
        let mask = 1u32 << bit_idx;

        // Clear both planes first
        self.plus[word_idx] &= !mask;
        self.minus[word_idx] &= !mask;

        // Set appropriate plane
        match value {
            Trit::P => self.plus[word_idx] |= mask,
            Trit::N => self.minus[word_idx] |= mask,
            Trit::Z => {} // Already cleared
        }
    }

    /// Get the trit value at a dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `dim >= len()`.
    #[must_use]
    pub fn get(&self, dim: usize) -> Trit {
        assert!(dim < self.num_dims, "dimension out of bounds");

        let word_idx = dim / 32;
        let bit_idx = dim % 32;
        let mask = 1u32 << bit_idx;

        let is_plus = (self.plus[word_idx] & mask) != 0;
        let is_minus = (self.minus[word_idx] & mask) != 0;

        debug_assert!(
            !(is_plus && is_minus),
            "invalid state: both planes set at dim {dim}"
        );

        Trit::from_bits(is_plus, is_minus)
    }

    /// Count non-zero elements.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        let plus_count: u32 = self.plus.iter().map(|w| w.count_ones()).sum();
        let minus_count: u32 = self.minus.iter().map(|w| w.count_ones()).sum();
        (plus_count + minus_count) as usize
    }

    /// Count positive (+1) elements.
    #[must_use]
    pub fn count_positive(&self) -> usize {
        self.plus.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Count negative (-1) elements.
    #[must_use]
    pub fn count_negative(&self) -> usize {
        self.minus.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Calculate sparsity (fraction of zeros).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f32 {
        if self.num_dims == 0 {
            return 1.0;
        }
        1.0 - (self.count_nonzero() as f32 / self.num_dims as f32)
    }

    /// Compute dot product with another vector using popcount.
    ///
    /// # Formula
    ///
    /// ```text
    /// dot = popcount(a+ & b+) + popcount(a- & b-)
    ///     - popcount(a+ & b-) - popcount(a- & b+)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    #[must_use]
    pub fn dot(&self, other: &PackedTritVec) -> i32 {
        assert_eq!(
            self.num_words(),
            other.num_words(),
            "vectors must have same size"
        );

        let mut result: i32 = 0;

        for i in 0..self.num_words() {
            let pp = (self.plus[i] & other.plus[i]).count_ones() as i32;
            let mm = (self.minus[i] & other.minus[i]).count_ones() as i32;
            let pm = (self.plus[i] & other.minus[i]).count_ones() as i32;
            let mp = (self.minus[i] & other.plus[i]).count_ones() as i32;

            result += pp + mm - pm - mp;
        }

        result
    }

    /// Compute the sum of all elements.
    #[must_use]
    pub fn sum(&self) -> i32 {
        self.count_positive() as i32 - self.count_negative() as i32
    }

    /// Element-wise negation (in place).
    pub fn negate(&mut self) {
        std::mem::swap(&mut self.plus, &mut self.minus);
    }

    /// Return a negated copy.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            plus: self.minus.clone(),
            minus: self.plus.clone(),
            num_dims: self.num_dims,
        }
    }

    /// Element-wise addition with another vector.
    ///
    /// Returns a new vector where each element is the sum clamped to {-1, 0, +1}.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    #[must_use]
    pub fn add_clamped(&self, other: &PackedTritVec) -> Self {
        assert_eq!(self.num_dims, other.num_dims, "dimension mismatch");

        let mut result = Self::new(self.num_dims);

        for i in 0..self.num_words() {
            // For each bit position:
            // sum = a_plus - a_minus + b_plus - b_minus
            // We need to clamp to {-1, 0, +1}

            // Cases for resulting +1 (sum >= 1):
            // - Both positive (pp=1, no negatives cancel)
            // - One positive, nothing negative against it

            // Simplified: result is positive if more positives than negatives
            let both_plus = self.plus[i] & other.plus[i];
            let both_minus = self.minus[i] & other.minus[i];
            let a_plus_only = self.plus[i] & !other.plus[i] & !other.minus[i];
            let b_plus_only = !self.plus[i] & !self.minus[i] & other.plus[i];
            let a_minus_only = self.minus[i] & !other.plus[i] & !other.minus[i];
            let b_minus_only = !self.plus[i] & !self.minus[i] & other.minus[i];

            // Result is positive if:
            // - both positive, or
            // - one positive and other is zero
            result.plus[i] = both_plus | a_plus_only | b_plus_only;

            // Result is negative if:
            // - both negative, or
            // - one negative and other is zero
            result.minus[i] = both_minus | a_minus_only | b_minus_only;

            // Clear conflicts (when a+ & b- or a- & b+, result is 0)
            let conflict = (self.plus[i] & other.minus[i]) | (self.minus[i] & other.plus[i]);
            result.plus[i] &= !conflict;
            result.minus[i] &= !conflict;
        }

        result
    }

    /// Get reference to the plus plane.
    #[must_use]
    pub fn plus_plane(&self) -> &[u32] {
        &self.plus
    }

    /// Get reference to the minus plane.
    #[must_use]
    pub fn minus_plane(&self) -> &[u32] {
        &self.minus
    }

    /// Convert to a vector of trits.
    #[must_use]
    pub fn to_trits(&self) -> Vec<Trit> {
        (0..self.num_dims).map(|i| self.get(i)).collect()
    }

    /// Convert to a vector of i8 values.
    #[must_use]
    pub fn to_i8_vec(&self) -> Vec<i8> {
        (0..self.num_dims).map(|i| self.get(i).value()).collect()
    }
}

impl fmt::Debug for PackedTritVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PackedTritVec(dims={}, nonzero={}, sparsity={:.2}%)",
            self.num_dims,
            self.count_nonzero(),
            self.sparsity() * 100.0
        )
    }
}

impl PartialEq for PackedTritVec {
    fn eq(&self, other: &Self) -> bool {
        self.num_dims == other.num_dims && self.plus == other.plus && self.minus == other.minus
    }
}

impl Eq for PackedTritVec {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_new() {
        let vec = PackedTritVec::new(100);
        assert_eq!(vec.len(), 100);
        assert_eq!(vec.num_words(), 4); // ceil(100/32) = 4

        // All zeros initially
        for i in 0..100 {
            assert_eq!(vec.get(i), Trit::Z);
        }
    }

    #[test]
    fn test_packed_set_get() {
        let mut vec = PackedTritVec::new(100);

        vec.set(0, Trit::P);
        vec.set(1, Trit::N);
        vec.set(50, Trit::P);
        vec.set(99, Trit::N);

        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.get(1), Trit::N);
        assert_eq!(vec.get(2), Trit::Z);
        assert_eq!(vec.get(50), Trit::P);
        assert_eq!(vec.get(99), Trit::N);
    }

    #[test]
    fn test_packed_overwrite() {
        let mut vec = PackedTritVec::new(10);

        vec.set(0, Trit::P);
        assert_eq!(vec.get(0), Trit::P);

        vec.set(0, Trit::N);
        assert_eq!(vec.get(0), Trit::N);

        vec.set(0, Trit::Z);
        assert_eq!(vec.get(0), Trit::Z);
    }

    #[test]
    fn test_packed_count() {
        let mut vec = PackedTritVec::new(100);

        vec.set(0, Trit::P);
        vec.set(1, Trit::P);
        vec.set(2, Trit::N);
        vec.set(3, Trit::N);
        vec.set(4, Trit::N);

        assert_eq!(vec.count_positive(), 2);
        assert_eq!(vec.count_negative(), 3);
        assert_eq!(vec.count_nonzero(), 5);
    }

    #[test]
    fn test_packed_sparsity() {
        let mut vec = PackedTritVec::new(100);

        // 100% sparse initially
        assert!((vec.sparsity() - 1.0).abs() < 0.001);

        // Set 5 non-zero values -> 95% sparse
        for i in 0..5 {
            vec.set(i, Trit::P);
        }
        assert!((vec.sparsity() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_packed_dot_product() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        // Set up: a = [1, -1, 0, 1], b = [1, 1, -1, 0]
        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(3, Trit::P);

        b.set(0, Trit::P);
        b.set(1, Trit::P);
        b.set(2, Trit::N);

        // dot = 1*1 + (-1)*1 + 0*(-1) + 1*0 = 1 - 1 + 0 + 0 = 0
        assert_eq!(a.dot(&b), 0);

        // Change b[1] to -1
        b.set(1, Trit::N);
        // dot = 1*1 + (-1)*(-1) + 0*(-1) + 1*0 = 1 + 1 = 2
        assert_eq!(a.dot(&b), 2);
    }

    #[test]
    fn test_packed_sum() {
        let mut vec = PackedTritVec::new(10);

        vec.set(0, Trit::P);
        vec.set(1, Trit::P);
        vec.set(2, Trit::N);

        assert_eq!(vec.sum(), 1); // 1 + 1 - 1 = 1
    }

    #[test]
    fn test_packed_negate() {
        let mut vec = PackedTritVec::new(10);

        vec.set(0, Trit::P);
        vec.set(1, Trit::N);

        vec.negate();

        assert_eq!(vec.get(0), Trit::N);
        assert_eq!(vec.get(1), Trit::P);
    }

    #[test]
    fn test_packed_negated() {
        let mut vec = PackedTritVec::new(10);
        vec.set(0, Trit::P);
        vec.set(1, Trit::N);

        let neg = vec.negated();

        // Original unchanged
        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.get(1), Trit::N);

        // Negated copy
        assert_eq!(neg.get(0), Trit::N);
        assert_eq!(neg.get(1), Trit::P);
    }

    #[test]
    fn test_packed_from_trits() {
        let trits = [Trit::P, Trit::N, Trit::Z, Trit::P, Trit::N];
        let vec = PackedTritVec::from_trits(&trits);

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.get(1), Trit::N);
        assert_eq!(vec.get(2), Trit::Z);
        assert_eq!(vec.get(3), Trit::P);
        assert_eq!(vec.get(4), Trit::N);
    }

    #[test]
    fn test_packed_to_trits() {
        let mut vec = PackedTritVec::new(5);
        vec.set(0, Trit::P);
        vec.set(1, Trit::N);
        vec.set(3, Trit::P);

        let trits = vec.to_trits();
        assert_eq!(trits, vec![Trit::P, Trit::N, Trit::Z, Trit::P, Trit::Z]);
    }

    #[test]
    fn test_packed_add_clamped() {
        let mut a = PackedTritVec::new(5);
        let mut b = PackedTritVec::new(5);

        // a = [+1, -1, 0, +1, -1]
        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(3, Trit::P);
        a.set(4, Trit::N);

        // b = [+1, +1, -1, 0, -1]
        b.set(0, Trit::P);
        b.set(1, Trit::P);
        b.set(2, Trit::N);
        b.set(4, Trit::N);

        let result = a.add_clamped(&b);

        // Expected: [+1, 0, -1, +1, -1] (clamped sums)
        assert_eq!(result.get(0), Trit::P); // 1+1 -> 1 (clamped)
        assert_eq!(result.get(1), Trit::Z); // -1+1 -> 0
        assert_eq!(result.get(2), Trit::N); // 0-1 -> -1
        assert_eq!(result.get(3), Trit::P); // 1+0 -> 1
        assert_eq!(result.get(4), Trit::N); // -1-1 -> -1 (clamped)
    }

    #[test]
    fn test_packed_from_i8_slice() {
        let values: Vec<i8> = vec![1, -1, 0, 1, -1];
        let vec = PackedTritVec::from_i8_slice(&values).unwrap();

        assert_eq!(vec.to_i8_vec(), values);
    }

    #[test]
    fn test_packed_equality() {
        let mut a = PackedTritVec::new(10);
        let mut b = PackedTritVec::new(10);

        a.set(0, Trit::P);
        b.set(0, Trit::P);

        assert_eq!(a, b);

        b.set(1, Trit::N);
        assert_ne!(a, b);
    }
}
