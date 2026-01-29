//! Sparse ternary vector storage using COO format.
//!
//! This module provides `SparseVec`, an efficient representation for highly
//! sparse ternary vectors. It stores only non-zero indices and their signs.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{Result, TernaryError};
use crate::packed::PackedTritVec;
use crate::trit::Trit;

/// A sparse ternary vector using COO (Coordinate) format.
///
/// Only non-zero values are stored, making this efficient for vectors where
/// most elements are zero (high sparsity).
///
/// # Storage
///
/// Non-zero indices are stored separately for positive and negative values:
/// - `positive_indices`: indices where value is +1
/// - `negative_indices`: indices where value is -1
///
/// # When to Use
///
/// Use `SparseVec` when sparsity > 90% for memory efficiency.
/// Use `PackedTritVec` for denser vectors or when operations like dot product
/// need consistent O(n) time regardless of sparsity.
///
/// # Examples
///
/// ```
/// use trit_vsa::{SparseVec, Trit};
///
/// let mut vec = SparseVec::new(1000);
/// vec.set(10, Trit::P);
/// vec.set(500, Trit::N);
///
/// assert_eq!(vec.get(10), Trit::P);
/// assert_eq!(vec.get(500), Trit::N);
/// assert_eq!(vec.get(0), Trit::Z);
/// assert_eq!(vec.count_nonzero(), 2);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct SparseVec {
    /// Indices where value is +1 (sorted).
    positive_indices: Vec<usize>,
    /// Indices where value is -1 (sorted).
    negative_indices: Vec<usize>,
    /// Logical dimension count.
    num_dims: usize,
}

impl SparseVec {
    /// Create a new sparse vector with given dimension count.
    ///
    /// All values are initialized to zero (no storage needed).
    #[must_use]
    pub fn new(num_dims: usize) -> Self {
        Self {
            positive_indices: Vec::new(),
            negative_indices: Vec::new(),
            num_dims,
        }
    }

    /// Create from separate index lists.
    ///
    /// # Arguments
    ///
    /// * `positive_indices` - Indices where value is +1
    /// * `negative_indices` - Indices where value is -1
    /// * `num_dims` - Logical dimension count
    ///
    /// # Errors
    ///
    /// Returns error if any index is out of bounds or if there are duplicates
    /// across positive and negative lists.
    pub fn from_indices(
        mut positive_indices: Vec<usize>,
        mut negative_indices: Vec<usize>,
        num_dims: usize,
    ) -> Result<Self> {
        // Validate and sort
        positive_indices.sort_unstable();
        negative_indices.sort_unstable();

        // Check bounds
        if let Some(&max) = positive_indices.last() {
            if max >= num_dims {
                return Err(TernaryError::IndexOutOfBounds {
                    index: max,
                    size: num_dims,
                });
            }
        }
        if let Some(&max) = negative_indices.last() {
            if max >= num_dims {
                return Err(TernaryError::IndexOutOfBounds {
                    index: max,
                    size: num_dims,
                });
            }
        }

        // Check for overlap (same index can't be both positive and negative)
        let mut pi = 0;
        let mut ni = 0;
        while pi < positive_indices.len() && ni < negative_indices.len() {
            match positive_indices[pi].cmp(&negative_indices[ni]) {
                std::cmp::Ordering::Equal => {
                    return Err(TernaryError::InvalidValue(positive_indices[pi] as i32));
                }
                std::cmp::Ordering::Less => pi += 1,
                std::cmp::Ordering::Greater => ni += 1,
            }
        }

        Ok(Self {
            positive_indices,
            negative_indices,
            num_dims,
        })
    }

    /// Create from a slice of trits.
    #[must_use]
    pub fn from_trits(trits: &[Trit]) -> Self {
        let mut positive_indices = Vec::new();
        let mut negative_indices = Vec::new();

        for (i, &trit) in trits.iter().enumerate() {
            match trit {
                Trit::P => positive_indices.push(i),
                Trit::N => negative_indices.push(i),
                Trit::Z => {}
            }
        }

        Self {
            positive_indices,
            negative_indices,
            num_dims: trits.len(),
        }
    }

    /// Create from a [`PackedTritVec`].
    #[must_use]
    pub fn from_packed(packed: &PackedTritVec) -> Self {
        let mut positive_indices = Vec::new();
        let mut negative_indices = Vec::new();

        for i in 0..packed.len() {
            match packed.get(i) {
                Trit::P => positive_indices.push(i),
                Trit::N => negative_indices.push(i),
                Trit::Z => {}
            }
        }

        Self {
            positive_indices,
            negative_indices,
            num_dims: packed.len(),
        }
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

    /// Set a dimension to a trit value.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= len()`.
    pub fn set(&mut self, dim: usize, value: Trit) {
        assert!(dim < self.num_dims, "dimension out of bounds");

        // Remove from current lists
        self.positive_indices.retain(|&i| i != dim);
        self.negative_indices.retain(|&i| i != dim);

        // Add to appropriate list
        match value {
            Trit::P => {
                let pos = self.positive_indices.partition_point(|&x| x < dim);
                self.positive_indices.insert(pos, dim);
            }
            Trit::N => {
                let pos = self.negative_indices.partition_point(|&x| x < dim);
                self.negative_indices.insert(pos, dim);
            }
            Trit::Z => {} // Already removed
        }
    }

    /// Get the trit value at a dimension.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= len()`.
    #[must_use]
    pub fn get(&self, dim: usize) -> Trit {
        assert!(dim < self.num_dims, "dimension out of bounds");

        if self.positive_indices.binary_search(&dim).is_ok() {
            Trit::P
        } else if self.negative_indices.binary_search(&dim).is_ok() {
            Trit::N
        } else {
            Trit::Z
        }
    }

    /// Get the number of dimensions.
    #[must_use]
    pub fn num_dims(&self) -> usize {
        self.num_dims
    }

    /// Count non-zero elements.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        self.positive_indices.len() + self.negative_indices.len()
    }

    /// Count positive (+1) elements.
    #[must_use]
    pub fn count_positive(&self) -> usize {
        self.positive_indices.len()
    }

    /// Count negative (-1) elements.
    #[must_use]
    pub fn count_negative(&self) -> usize {
        self.negative_indices.len()
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

    /// Compute dot product with another sparse vector.
    ///
    /// This is O(k1 + k2) where k1 and k2 are the number of non-zero elements.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    #[must_use]
    pub fn dot(&self, other: &SparseVec) -> i32 {
        assert_eq!(
            self.num_dims, other.num_dims,
            "vectors must have same dimensions"
        );

        let mut result: i32 = 0;

        // Count intersections between same-sign indices
        result += Self::count_intersection(&self.positive_indices, &other.positive_indices) as i32;
        result += Self::count_intersection(&self.negative_indices, &other.negative_indices) as i32;

        // Subtract intersections between opposite-sign indices
        result -= Self::count_intersection(&self.positive_indices, &other.negative_indices) as i32;
        result -= Self::count_intersection(&self.negative_indices, &other.positive_indices) as i32;

        result
    }

    /// Compute dot product with a packed vector.
    ///
    /// Efficient when this sparse vector has few non-zeros.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    #[must_use]
    pub fn dot_packed(&self, other: &PackedTritVec) -> i32 {
        assert_eq!(
            self.num_dims,
            other.len(),
            "vectors must have same dimensions"
        );

        let mut result: i32 = 0;

        // Sum contributions from positive indices
        for &idx in &self.positive_indices {
            result += other.get(idx).value() as i32;
        }

        // Sum contributions from negative indices (note: we add negative of other's value)
        for &idx in &self.negative_indices {
            result -= other.get(idx).value() as i32;
        }

        result
    }

    /// Compute the sum of all elements.
    #[must_use]
    pub fn sum(&self) -> i32 {
        self.positive_indices.len() as i32 - self.negative_indices.len() as i32
    }

    /// Return a negated copy.
    #[must_use]
    pub fn negated(&self) -> Self {
        Self {
            positive_indices: self.negative_indices.clone(),
            negative_indices: self.positive_indices.clone(),
            num_dims: self.num_dims,
        }
    }

    /// Get reference to positive indices.
    #[must_use]
    pub fn positive_indices(&self) -> &[usize] {
        &self.positive_indices
    }

    /// Get reference to negative indices.
    #[must_use]
    pub fn negative_indices(&self) -> &[usize] {
        &self.negative_indices
    }

    /// Convert to a [`PackedTritVec`].
    #[must_use]
    pub fn to_packed(&self) -> PackedTritVec {
        let mut packed = PackedTritVec::new(self.num_dims);
        for &idx in &self.positive_indices {
            packed.set(idx, Trit::P);
        }
        for &idx in &self.negative_indices {
            packed.set(idx, Trit::N);
        }
        packed
    }

    /// Convert to a vector of trits.
    #[must_use]
    pub fn to_trits(&self) -> Vec<Trit> {
        let mut result = vec![Trit::Z; self.num_dims];
        for &idx in &self.positive_indices {
            result[idx] = Trit::P;
        }
        for &idx in &self.negative_indices {
            result[idx] = Trit::N;
        }
        result
    }

    /// Memory size in bytes (approximate).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        // Vec overhead + index storage
        std::mem::size_of::<Self>()
            + self.positive_indices.capacity() * std::mem::size_of::<usize>()
            + self.negative_indices.capacity() * std::mem::size_of::<usize>()
    }

    // Internal: count intersection of two sorted lists
    fn count_intersection(a: &[usize], b: &[usize]) -> usize {
        let mut count = 0;
        let mut ai = 0;
        let mut bi = 0;

        while ai < a.len() && bi < b.len() {
            match a[ai].cmp(&b[bi]) {
                std::cmp::Ordering::Equal => {
                    count += 1;
                    ai += 1;
                    bi += 1;
                }
                std::cmp::Ordering::Less => ai += 1,
                std::cmp::Ordering::Greater => bi += 1,
            }
        }

        count
    }
}

impl fmt::Debug for SparseVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseVec(dims={}, pos={}, neg={}, sparsity={:.2}%)",
            self.num_dims,
            self.positive_indices.len(),
            self.negative_indices.len(),
            self.sparsity() * 100.0
        )
    }
}

impl PartialEq for SparseVec {
    fn eq(&self, other: &Self) -> bool {
        self.num_dims == other.num_dims
            && self.positive_indices == other.positive_indices
            && self.negative_indices == other.negative_indices
    }
}

impl Eq for SparseVec {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_new() {
        let vec = SparseVec::new(1000);
        assert_eq!(vec.len(), 1000);
        assert_eq!(vec.count_nonzero(), 0);
        assert!((vec.sparsity() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sparse_set_get() {
        let mut vec = SparseVec::new(100);

        vec.set(10, Trit::P);
        vec.set(20, Trit::N);
        vec.set(50, Trit::P);

        assert_eq!(vec.get(10), Trit::P);
        assert_eq!(vec.get(20), Trit::N);
        assert_eq!(vec.get(50), Trit::P);
        assert_eq!(vec.get(0), Trit::Z);
        assert_eq!(vec.get(99), Trit::Z);
    }

    #[test]
    fn test_sparse_overwrite() {
        let mut vec = SparseVec::new(10);

        vec.set(0, Trit::P);
        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.count_nonzero(), 1);

        vec.set(0, Trit::N);
        assert_eq!(vec.get(0), Trit::N);
        assert_eq!(vec.count_nonzero(), 1);

        vec.set(0, Trit::Z);
        assert_eq!(vec.get(0), Trit::Z);
        assert_eq!(vec.count_nonzero(), 0);
    }

    #[test]
    fn test_sparse_dot() {
        let mut a = SparseVec::new(100);
        let mut b = SparseVec::new(100);

        // a = [+1 at 0, -1 at 1, +1 at 10]
        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(10, Trit::P);

        // b = [+1 at 0, +1 at 1, -1 at 20]
        b.set(0, Trit::P);
        b.set(1, Trit::P);
        b.set(20, Trit::N);

        // dot = 1*1 + (-1)*1 + 1*0 + 0*(-1) = 1 - 1 = 0
        assert_eq!(a.dot(&b), 0);

        // Modify b[1] to -1
        b.set(1, Trit::N);
        // dot = 1*1 + (-1)*(-1) + 1*0 + 0*(-1) = 1 + 1 = 2
        assert_eq!(a.dot(&b), 2);
    }

    #[test]
    fn test_sparse_dot_packed() {
        let mut sparse = SparseVec::new(64);
        let mut packed = PackedTritVec::new(64);

        sparse.set(0, Trit::P);
        sparse.set(1, Trit::N);

        packed.set(0, Trit::P);
        packed.set(1, Trit::P);
        packed.set(2, Trit::N);

        // dot = 1*1 + (-1)*1 = 0
        assert_eq!(sparse.dot_packed(&packed), 0);

        packed.set(1, Trit::N);
        // dot = 1*1 + (-1)*(-1) = 2
        assert_eq!(sparse.dot_packed(&packed), 2);
    }

    #[test]
    fn test_sparse_from_trits() {
        let trits = [Trit::P, Trit::N, Trit::Z, Trit::P, Trit::Z];
        let vec = SparseVec::from_trits(&trits);

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.count_positive(), 2);
        assert_eq!(vec.count_negative(), 1);

        assert_eq!(vec.to_trits(), trits);
    }

    #[test]
    fn test_sparse_to_packed_roundtrip() {
        let mut sparse = SparseVec::new(100);
        sparse.set(0, Trit::P);
        sparse.set(50, Trit::N);
        sparse.set(99, Trit::P);

        let packed = sparse.to_packed();
        let back = SparseVec::from_packed(&packed);

        assert_eq!(sparse, back);
    }

    #[test]
    fn test_sparse_negated() {
        let mut vec = SparseVec::new(10);
        vec.set(0, Trit::P);
        vec.set(1, Trit::N);

        let neg = vec.negated();

        assert_eq!(neg.get(0), Trit::N);
        assert_eq!(neg.get(1), Trit::P);
    }

    #[test]
    fn test_sparse_from_indices() {
        let pos = vec![0, 10, 50];
        let neg = vec![5, 20];
        let vec = SparseVec::from_indices(pos, neg, 100).unwrap();

        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.get(10), Trit::P);
        assert_eq!(vec.get(50), Trit::P);
        assert_eq!(vec.get(5), Trit::N);
        assert_eq!(vec.get(20), Trit::N);
        assert_eq!(vec.get(1), Trit::Z);
    }

    #[test]
    fn test_sparse_from_indices_overlap_error() {
        let pos = vec![0, 10];
        let neg = vec![10, 20]; // 10 is in both - invalid
        let result = SparseVec::from_indices(pos, neg, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_from_indices_bounds_error() {
        let pos = vec![100]; // Out of bounds for dim=100
        let neg = vec![];
        let result = SparseVec::from_indices(pos, neg, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_sum() {
        let mut vec = SparseVec::new(100);
        vec.set(0, Trit::P);
        vec.set(1, Trit::P);
        vec.set(2, Trit::N);

        assert_eq!(vec.sum(), 1); // 1 + 1 - 1 = 1
    }
}
