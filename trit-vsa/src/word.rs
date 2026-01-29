//! Word type representing 6 trits (values -364 to +364).
//!
//! A Word6 consists of 6 trits, capable of representing 729 distinct values (3^6).
//! This is useful for balanced ternary arithmetic at the word level.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg};

use crate::error::{Result, TernaryError};
use crate::trit::Trit;
use crate::tryte::Tryte3;

/// Minimum value representable by a Word6 (-364).
pub const WORD6_MIN: i32 = -364;
/// Maximum value representable by a Word6 (+364).
pub const WORD6_MAX: i32 = 364;

/// A balanced ternary word consisting of 6 trits.
///
/// # Value Range
///
/// A Word6 can represent values from -364 to +364:
/// ```text
/// Value = t0*1 + t1*3 + t2*9 + t3*27 + t4*81 + t5*243
/// Min: -1 - 3 - 9 - 27 - 81 - 243 = -364
/// Max: +1 + 3 + 9 + 27 + 81 + 243 = +364
/// ```
///
/// # Internal Representation
///
/// Stored as a u16 with 2 bits per trit:
/// - Bits 0-1: trit 0 (least significant)
/// - Bits 2-3: trit 1
/// - ...
/// - Bits 10-11: trit 5 (most significant)
///
/// # Examples
///
/// ```
/// use trit_vsa::Word6;
///
/// let w = Word6::from_value(100).unwrap();
/// assert_eq!(w.value(), 100);
///
/// let w_neg = -w;
/// assert_eq!(w_neg.value(), -100);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Word6(u16);

impl Word6 {
    /// Create a word from an integer value.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer value (-364 to +364)
    ///
    /// # Errors
    ///
    /// Returns `TernaryError::InvalidWordValue` if value is outside range.
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Word6;
    ///
    /// let w = Word6::from_value(123).unwrap();
    /// assert_eq!(w.value(), 123);
    ///
    /// assert!(Word6::from_value(365).is_err());
    /// assert!(Word6::from_value(-365).is_err());
    /// ```
    pub fn from_value(value: i32) -> Result<Self> {
        if !(WORD6_MIN..=WORD6_MAX).contains(&value) {
            return Err(TernaryError::InvalidWordValue(value));
        }

        let trits = Self::value_to_trits(value);
        Ok(Self::from_trits(trits))
    }

    /// Create a word from six trits.
    ///
    /// # Arguments
    ///
    /// * `trits` - Array of 6 trits [t0, t1, t2, t3, t4, t5] where t0 is least significant
    #[must_use]
    pub fn from_trits(trits: [Trit; 6]) -> Self {
        let mut encoded: u16 = 0;
        for (i, &trit) in trits.iter().enumerate() {
            encoded |= (Self::encode_trit(trit) as u16) << (i * 2);
        }
        Self(encoded)
    }

    /// Create a word from two trytes.
    ///
    /// # Arguments
    ///
    /// * `low` - Low tryte (trits 0-2)
    /// * `high` - High tryte (trits 3-5)
    #[must_use]
    pub fn from_trytes(low: Tryte3, high: Tryte3) -> Self {
        let low_trits = low.to_trits();
        let high_trits = high.to_trits();
        Self::from_trits([
            low_trits[0],
            low_trits[1],
            low_trits[2],
            high_trits[0],
            high_trits[1],
            high_trits[2],
        ])
    }

    /// Get the integer value of the word.
    #[must_use]
    pub fn value(self) -> i32 {
        let trits = self.to_trits();
        let mut result: i32 = 0;
        let mut power: i32 = 1;
        for trit in trits {
            result += trit.value() as i32 * power;
            power *= 3;
        }
        result
    }

    /// Extract the six trits.
    ///
    /// # Returns
    ///
    /// Array [t0, ..., t5] where t0 is least significant.
    #[must_use]
    pub fn to_trits(self) -> [Trit; 6] {
        let mut trits = [Trit::Z; 6];
        for (i, trit) in trits.iter_mut().enumerate() {
            *trit = Self::decode_trit(((self.0 >> (i * 2)) & 0b11) as u8);
        }
        trits
    }

    /// Split into two trytes.
    ///
    /// # Returns
    ///
    /// Tuple `(low, high)` where low contains trits 0-2 and high contains trits 3-5.
    #[must_use]
    pub fn to_trytes(self) -> (Tryte3, Tryte3) {
        let trits = self.to_trits();
        (
            Tryte3::from_trits([trits[0], trits[1], trits[2]]),
            Tryte3::from_trits([trits[3], trits[4], trits[5]]),
        )
    }

    /// Get a specific trit by index.
    ///
    /// # Arguments
    ///
    /// * `index` - Trit index (0-5)
    ///
    /// # Panics
    ///
    /// Panics if index >= 6.
    #[must_use]
    pub fn get_trit(self, index: usize) -> Trit {
        assert!(index < 6, "trit index out of bounds");
        Self::decode_trit(((self.0 >> (index * 2)) & 0b11) as u8)
    }

    /// Create a zero word.
    #[must_use]
    pub const fn zero() -> Self {
        // All trits are Z (encoded as 1): 01|01|01|01|01|01
        Self(0b01_01_01_01_01_01)
    }

    /// Check if the word is zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.value() == 0
    }

    /// Get the raw packed representation.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }

    // Internal: convert value to trit array using balanced ternary conversion
    fn value_to_trits(mut value: i32) -> [Trit; 6] {
        let mut trits = [Trit::Z; 6];

        for trit in &mut trits {
            if value == 0 {
                *trit = Trit::Z;
                continue;
            }

            let mut rem = value % 3;
            value /= 3;

            // Adjust for balanced ternary
            if rem == 2 {
                rem = -1;
                value += 1;
            } else if rem == -2 {
                rem = 1;
                value -= 1;
            }

            *trit = match rem {
                -1 => Trit::N,
                0 => Trit::Z,
                1 => Trit::P,
                _ => unreachable!(),
            };
        }

        trits
    }

    fn encode_trit(trit: Trit) -> u8 {
        match trit {
            Trit::N => 0,
            Trit::Z => 1,
            Trit::P => 2,
        }
    }

    fn decode_trit(bits: u8) -> Trit {
        match bits & 0b11 {
            0 => Trit::N,
            1 | 3 => Trit::Z, // 3 is invalid, treat as zero
            2 => Trit::P,
            _ => unreachable!(),
        }
    }
}

impl Default for Word6 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Neg for Word6 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let trits = self.to_trits();
        Self::from_trits([
            -trits[0], -trits[1], -trits[2], -trits[3], -trits[4], -trits[5],
        ])
    }
}

impl Add for Word6 {
    type Output = (Self, Trit);

    /// Add two words, returning (result, carry).
    fn add(self, other: Self) -> Self::Output {
        let a = self.to_trits();
        let b = other.to_trits();
        let mut result = [Trit::Z; 6];
        let mut carry = Trit::Z;

        for i in 0..6 {
            let (sum1, carry1) = a[i].add_with_carry(b[i]);
            let (sum2, carry2) = sum1.add_with_carry(carry);

            result[i] = sum2;
            let (carry_sum, _) = carry1.add_with_carry(carry2);
            carry = carry_sum;
        }

        (Self::from_trits(result), carry)
    }
}

impl Mul for Word6 {
    type Output = (Self, Self);

    /// Multiply two words, returning (low, high) result.
    ///
    /// The full result is `low + high * 729`.
    fn mul(self, other: Self) -> Self::Output {
        let product = self.value() as i64 * other.value() as i64;

        // Split into low and high words
        let low_val = ((product % 729) + 729 + 364) % 729 - 364;
        let high_val = (product - low_val) / 729;

        (
            Self::from_value(low_val as i32).unwrap_or_else(|_| Self::zero()),
            Self::from_value(high_val.clamp(WORD6_MIN as i64, WORD6_MAX as i64) as i32)
                .unwrap_or_else(|_| Self::zero()),
        )
    }
}

impl fmt::Debug for Word6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let trits = self.to_trits();
        write!(
            f,
            "Word6({}{}{}{}{}{} = {})",
            trits[5],
            trits[4],
            trits[3],
            trits[2],
            trits[1],
            trits[0],
            self.value()
        )
    }
}

impl fmt::Display for Word6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl TryFrom<i32> for Word6 {
    type Error = TernaryError;

    fn try_from(value: i32) -> Result<Self> {
        Self::from_value(value)
    }
}

impl From<Word6> for i32 {
    fn from(word: Word6) -> Self {
        word.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_range() {
        // Test boundary values
        assert!(Word6::from_value(WORD6_MIN).is_ok());
        assert!(Word6::from_value(WORD6_MAX).is_ok());
        assert!(Word6::from_value(WORD6_MIN - 1).is_err());
        assert!(Word6::from_value(WORD6_MAX + 1).is_err());

        // Test roundtrip for all values (this is a larger test)
        for v in WORD6_MIN..=WORD6_MAX {
            let w = Word6::from_value(v).expect("valid value");
            assert_eq!(w.value(), v, "failed for value {v}");
        }
    }

    #[test]
    fn test_word_zero() {
        let z = Word6::zero();
        assert_eq!(z.value(), 0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_word_negation() {
        let test_values = [0, 1, -1, 100, -100, 364, -364];
        for v in test_values {
            let w = Word6::from_value(v).unwrap();
            let neg = -w;
            assert_eq!(neg.value(), -v);
        }
    }

    #[test]
    fn test_word_addition() {
        // Simple addition
        let a = Word6::from_value(100).unwrap();
        let b = Word6::from_value(50).unwrap();
        let (result, carry) = a + b;
        assert_eq!(result.value() + carry.value() as i32 * 729, 150);

        // Addition with overflow
        let a = Word6::from_value(300).unwrap();
        let b = Word6::from_value(200).unwrap();
        let (result, carry) = a + b;
        let total = result.value() + carry.value() as i32 * 729;
        assert_eq!(total, 500);
    }

    #[test]
    fn test_word_multiplication() {
        let a = Word6::from_value(10).unwrap();
        let b = Word6::from_value(20).unwrap();
        let (low, high) = a * b;
        let total = low.value() + high.value() * 729;
        assert_eq!(total, 200);

        // Larger multiplication
        let a = Word6::from_value(50).unwrap();
        let b = Word6::from_value(50).unwrap();
        let (low, high) = a * b;
        let total = low.value() + high.value() * 729;
        assert_eq!(total, 2500);
    }

    #[test]
    fn test_word_tryte_conversion() {
        let w = Word6::from_value(100).unwrap();
        let (low, high) = w.to_trytes();

        // Reconstruct
        let reconstructed = Word6::from_trytes(low, high);
        assert_eq!(reconstructed.value(), 100);
    }

    #[test]
    fn test_word_get_trit() {
        let w = Word6::from_trits([Trit::N, Trit::Z, Trit::P, Trit::N, Trit::Z, Trit::P]);
        assert_eq!(w.get_trit(0), Trit::N);
        assert_eq!(w.get_trit(1), Trit::Z);
        assert_eq!(w.get_trit(2), Trit::P);
        assert_eq!(w.get_trit(3), Trit::N);
        assert_eq!(w.get_trit(4), Trit::Z);
        assert_eq!(w.get_trit(5), Trit::P);
    }
}
