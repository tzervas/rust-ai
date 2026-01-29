//! Balanced ternary arithmetic utilities.
//!
//! This module provides helper functions for balanced ternary arithmetic
//! beyond what's available on individual types.

use crate::error::Result;
use crate::trit::Trit;
use crate::tryte::Tryte3;
use crate::word::Word6;

/// Convert an integer to balanced ternary representation.
///
/// Returns a vector of trits from least significant to most significant.
///
/// # Arguments
///
/// * `value` - Integer to convert
/// * `min_trits` - Minimum number of trits in output (padded with zeros)
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, arithmetic::to_balanced_ternary};
///
/// let trits = to_balanced_ternary(5, 4);
/// // 5 = -1 + (-1)*3 + 1*9 = -1 - 3 + 9 = 5
/// assert_eq!(trits[0], Trit::N);  // -1
/// assert_eq!(trits[1], Trit::N);  // -1
/// assert_eq!(trits[2], Trit::P);  // +1
/// assert_eq!(trits[3], Trit::Z);  // padding
/// ```
#[must_use]
pub fn to_balanced_ternary(mut value: i64, min_trits: usize) -> Vec<Trit> {
    let mut trits = Vec::new();

    if value == 0 {
        while trits.len() < min_trits {
            trits.push(Trit::Z);
        }
        if trits.is_empty() {
            trits.push(Trit::Z);
        }
        return trits;
    }

    while value != 0 {
        // Handle negative values correctly
        let mut rem = value % 3;
        value /= 3;

        // Adjust for balanced ternary
        if rem == 2 {
            rem = -1;
            value += 1;
        } else if rem == -2 {
            rem = 1;
            value -= 1;
        } else if rem == -1 && value > 0 {
            // rem is already -1, no adjustment needed
        }

        let trit = match rem {
            -1 => Trit::N,
            0 => Trit::Z,
            1 => Trit::P,
            _ => unreachable!("unexpected remainder: {}", rem),
        };
        trits.push(trit);
    }

    // Ensure minimum length
    while trits.len() < min_trits {
        trits.push(Trit::Z);
    }

    trits
}

/// Convert balanced ternary representation to integer.
///
/// # Arguments
///
/// * `trits` - Slice of trits from least significant to most significant
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, arithmetic::from_balanced_ternary};
///
/// let trits = [Trit::N, Trit::N, Trit::P];  // 5
/// assert_eq!(from_balanced_ternary(&trits), 5);
/// ```
#[must_use]
pub fn from_balanced_ternary(trits: &[Trit]) -> i64 {
    let mut value: i64 = 0;
    let mut power: i64 = 1;

    for &trit in trits {
        value += trit.value() as i64 * power;
        power *= 3;
    }

    value
}

/// Add two balanced ternary numbers with carry propagation.
///
/// Returns the sum as a vector of trits.
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, arithmetic::{add_ternary, from_balanced_ternary}};
///
/// let a = [Trit::P, Trit::Z, Trit::P];  // 1 + 9 = 10
/// let b = [Trit::N, Trit::P, Trit::Z];  // -1 + 3 = 2
/// let sum = add_ternary(&a, &b);
/// assert_eq!(from_balanced_ternary(&sum), 12);
/// ```
#[must_use]
pub fn add_ternary(a: &[Trit], b: &[Trit]) -> Vec<Trit> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len + 1);
    let mut carry = Trit::Z;

    for i in 0..max_len {
        let ta = a.get(i).copied().unwrap_or(Trit::Z);
        let tb = b.get(i).copied().unwrap_or(Trit::Z);

        let (sum1, carry1) = ta.add_with_carry(tb);
        let (sum2, carry2) = sum1.add_with_carry(carry);

        result.push(sum2);

        // Combine carries
        let (carry_sum, _) = carry1.add_with_carry(carry2);
        carry = carry_sum;
    }

    if carry != Trit::Z {
        result.push(carry);
    }

    result
}

/// Multiply two balanced ternary numbers.
///
/// Returns the product as a vector of trits.
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, arithmetic::{multiply_ternary, from_balanced_ternary}};
///
/// let a = [Trit::P, Trit::P];  // 1 + 3 = 4
/// let b = [Trit::N, Trit::P];  // -1 + 3 = 2
/// let product = multiply_ternary(&a, &b);
/// assert_eq!(from_balanced_ternary(&product), 8);
/// ```
#[must_use]
pub fn multiply_ternary(a: &[Trit], b: &[Trit]) -> Vec<Trit> {
    if a.is_empty() || b.is_empty() {
        return vec![Trit::Z];
    }

    let mut result = vec![Trit::Z; a.len() + b.len()];

    for (i, &ta) in a.iter().enumerate() {
        if ta == Trit::Z {
            continue;
        }

        let mut partial = vec![Trit::Z; i]; // Shift by position

        for &tb in b {
            partial.push(ta * tb);
        }

        result = add_ternary(&result, &partial);
    }

    // Trim leading zeros
    while result.len() > 1 && result.last() == Some(&Trit::Z) {
        result.pop();
    }

    result
}

/// Negate a balanced ternary number.
///
/// # Examples
///
/// ```
/// use trit_vsa::{Trit, arithmetic::{negate_ternary, from_balanced_ternary}};
///
/// let a = [Trit::P, Trit::N, Trit::P];  // 1 - 3 + 9 = 7
/// let neg = negate_ternary(&a);
/// assert_eq!(from_balanced_ternary(&neg), -7);
/// ```
#[must_use]
pub fn negate_ternary(trits: &[Trit]) -> Vec<Trit> {
    trits.iter().map(|&t| -t).collect()
}

/// Compare two balanced ternary numbers.
///
/// Returns:
/// - `Ordering::Less` if a < b
/// - `Ordering::Equal` if a == b
/// - `Ordering::Greater` if a > b
#[must_use]
pub fn compare_ternary(a: &[Trit], b: &[Trit]) -> std::cmp::Ordering {
    let va = from_balanced_ternary(a);
    let vb = from_balanced_ternary(b);
    va.cmp(&vb)
}

/// Convert a Tryte3 to an integer.
#[must_use]
pub fn tryte_to_int(tryte: Tryte3) -> i32 {
    tryte.value()
}

/// Convert an integer to a Tryte3.
///
/// # Errors
///
/// Returns error if value is outside [-13, +13].
pub fn int_to_tryte(value: i32) -> Result<Tryte3> {
    Tryte3::from_value(value)
}

/// Convert a Word6 to an integer.
#[must_use]
pub fn word_to_int(word: Word6) -> i32 {
    word.value()
}

/// Convert an integer to a Word6.
///
/// # Errors
///
/// Returns error if value is outside [-364, +364].
pub fn int_to_word(value: i32) -> Result<Word6> {
    Word6::from_value(value)
}

/// Check if a value can be represented as a single trit.
#[must_use]
pub const fn is_valid_trit(value: i32) -> bool {
    matches!(value, -1..=1)
}

/// Check if a value can be represented as a Tryte3.
#[must_use]
pub const fn is_valid_tryte(value: i32) -> bool {
    value >= -13 && value <= 13
}

/// Check if a value can be represented as a Word6.
#[must_use]
pub const fn is_valid_word(value: i32) -> bool {
    value >= -364 && value <= 364
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_from_balanced_ternary_roundtrip() {
        for value in -1000..=1000 {
            let trits = to_balanced_ternary(value, 1);
            let back = from_balanced_ternary(&trits);
            assert_eq!(back, value, "roundtrip failed for {value}");
        }
    }

    #[test]
    fn test_add_ternary() {
        // Test various additions
        let test_cases = [(5, 3), (10, 20), (-5, 5), (100, -50), (0, 0), (-100, -100)];

        for (a, b) in test_cases {
            let ta = to_balanced_ternary(a, 1);
            let tb = to_balanced_ternary(b, 1);
            let sum = add_ternary(&ta, &tb);
            let result = from_balanced_ternary(&sum);
            assert_eq!(result, a + b, "{a} + {b} = {} (expected {})", result, a + b);
        }
    }

    #[test]
    fn test_multiply_ternary() {
        let test_cases = [(4, 2), (3, 3), (5, -2), (-3, -4), (0, 100), (1, 50)];

        for (a, b) in test_cases {
            let ta = to_balanced_ternary(a, 1);
            let tb = to_balanced_ternary(b, 1);
            let product = multiply_ternary(&ta, &tb);
            let result = from_balanced_ternary(&product);
            assert_eq!(result, a * b, "{a} * {b} = {} (expected {})", result, a * b);
        }
    }

    #[test]
    fn test_negate_ternary() {
        for value in -100..=100 {
            let trits = to_balanced_ternary(value, 1);
            let neg = negate_ternary(&trits);
            let result = from_balanced_ternary(&neg);
            assert_eq!(result, -value);
        }
    }

    #[test]
    fn test_compare_ternary() {
        use std::cmp::Ordering;

        let a = to_balanced_ternary(10, 4);
        let b = to_balanced_ternary(5, 4);
        let c = to_balanced_ternary(10, 4);

        assert_eq!(compare_ternary(&a, &b), Ordering::Greater);
        assert_eq!(compare_ternary(&b, &a), Ordering::Less);
        assert_eq!(compare_ternary(&a, &c), Ordering::Equal);
    }

    #[test]
    fn test_validity_checks() {
        assert!(is_valid_trit(0));
        assert!(is_valid_trit(1));
        assert!(is_valid_trit(-1));
        assert!(!is_valid_trit(2));

        assert!(is_valid_tryte(0));
        assert!(is_valid_tryte(13));
        assert!(is_valid_tryte(-13));
        assert!(!is_valid_tryte(14));

        assert!(is_valid_word(0));
        assert!(is_valid_word(364));
        assert!(is_valid_word(-364));
        assert!(!is_valid_word(365));
    }
}
