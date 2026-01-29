//! Tryte type representing 3 trits (values -13 to +13).
//!
//! A tryte is a balanced ternary byte, consisting of 3 trits.
//! It can represent 27 distinct values (3^3 = 27), ranging from -13 to +13.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg};

use crate::error::{Result, TernaryError};
use crate::trit::Trit;

/// Minimum value representable by a Tryte3 (-13).
pub const TRYTE3_MIN: i32 = -13;
/// Maximum value representable by a Tryte3 (+13).
pub const TRYTE3_MAX: i32 = 13;

/// A balanced ternary byte consisting of 3 trits.
///
/// # Value Range
///
/// A Tryte3 can represent values from -13 to +13:
/// ```text
/// Value = t0 * 1 + t1 * 3 + t2 * 9
/// Min: -1 - 3 - 9 = -13
/// Max: +1 + 3 + 9 = +13
/// ```
///
/// # Internal Representation
///
/// Stored as a single u8 with the following encoding:
/// - Bits 0-1: trit 0 (least significant)
/// - Bits 2-3: trit 1
/// - Bits 4-5: trit 2 (most significant)
///
/// Each trit is encoded as: 0=N(-1), 1=Z(0), 2=P(+1)
///
/// # Examples
///
/// ```
/// use trit_vsa::Tryte3;
///
/// let t = Tryte3::from_value(5).unwrap();
/// assert_eq!(t.value(), 5);
///
/// // Decompose into trits: 5 = -1 + (-1)*3 + 1*9 = -1 - 3 + 9
/// let trits = t.to_trits();
/// assert_eq!(trits[0].value(), -1);  // t0 = -1
/// assert_eq!(trits[1].value(), -1);  // t1 = -1
/// assert_eq!(trits[2].value(), 1);   // t2 = +1
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tryte3(u8);

impl Tryte3 {
    /// Create a tryte from an integer value.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer value (-13 to +13)
    ///
    /// # Errors
    ///
    /// Returns `TernaryError::InvalidTryteValue` if value is outside range.
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Tryte3;
    ///
    /// let t = Tryte3::from_value(7).unwrap();
    /// assert_eq!(t.value(), 7);
    ///
    /// assert!(Tryte3::from_value(14).is_err());
    /// assert!(Tryte3::from_value(-14).is_err());
    /// ```
    pub fn from_value(value: i32) -> Result<Self> {
        if !(TRYTE3_MIN..=TRYTE3_MAX).contains(&value) {
            return Err(TernaryError::InvalidTryteValue(value));
        }

        let trits = Self::value_to_trits(value);
        Ok(Self::from_trits(trits))
    }

    /// Create a tryte from three trits.
    ///
    /// # Arguments
    ///
    /// * `trits` - Array of 3 trits [t0, t1, t2] where t0 is least significant
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::{Tryte3, Trit};
    ///
    /// let t = Tryte3::from_trits([Trit::P, Trit::Z, Trit::N]);
    /// // Value = 1 + 0*3 + (-1)*9 = 1 - 9 = -8
    /// assert_eq!(t.value(), -8);
    /// ```
    #[must_use]
    pub fn from_trits(trits: [Trit; 3]) -> Self {
        let encoded = Self::encode_trit(trits[0])
            | (Self::encode_trit(trits[1]) << 2)
            | (Self::encode_trit(trits[2]) << 4);
        Self(encoded)
    }

    /// Get the integer value of the tryte.
    #[must_use]
    pub fn value(self) -> i32 {
        let trits = self.to_trits();
        trits[0].value() as i32 + trits[1].value() as i32 * 3 + trits[2].value() as i32 * 9
    }

    /// Extract the three trits.
    ///
    /// # Returns
    ///
    /// Array [t0, t1, t2] where t0 is least significant.
    #[must_use]
    pub fn to_trits(self) -> [Trit; 3] {
        [
            Self::decode_trit(self.0 & 0b11),
            Self::decode_trit((self.0 >> 2) & 0b11),
            Self::decode_trit((self.0 >> 4) & 0b11),
        ]
    }

    /// Get a specific trit by index.
    ///
    /// # Arguments
    ///
    /// * `index` - Trit index (0, 1, or 2)
    ///
    /// # Panics
    ///
    /// Panics if index >= 3.
    #[must_use]
    pub fn get_trit(self, index: usize) -> Trit {
        assert!(index < 3, "trit index out of bounds");
        Self::decode_trit((self.0 >> (index * 2)) & 0b11)
    }

    /// Create a zero tryte.
    #[must_use]
    pub const fn zero() -> Self {
        // All trits are Z (encoded as 1), so: 01|01|01 = 0b010101 = 21
        Self(0b01_01_01)
    }

    /// Check if the tryte is zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.value() == 0
    }

    /// Get the raw packed representation.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    // Internal: convert value to trit array using balanced ternary conversion
    fn value_to_trits(mut value: i32) -> [Trit; 3] {
        let mut trits = [Trit::Z; 3];

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

    // Internal: encode a trit to 2 bits (0=N, 1=Z, 2=P)
    fn encode_trit(trit: Trit) -> u8 {
        match trit {
            Trit::N => 0,
            Trit::Z => 1,
            Trit::P => 2,
        }
    }

    // Internal: decode 2 bits to a trit
    fn decode_trit(bits: u8) -> Trit {
        match bits & 0b11 {
            0 => Trit::N,
            1 | 3 => Trit::Z, // 3 is invalid, treat as zero
            2 => Trit::P,
            _ => unreachable!(),
        }
    }
}

impl Default for Tryte3 {
    fn default() -> Self {
        Self::zero()
    }
}

impl Neg for Tryte3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let trits = self.to_trits();
        Self::from_trits([-trits[0], -trits[1], -trits[2]])
    }
}

impl Add for Tryte3 {
    type Output = (Self, Trit);

    /// Add two trytes, returning (result, carry).
    fn add(self, other: Self) -> Self::Output {
        let a = self.to_trits();
        let b = other.to_trits();
        let mut result = [Trit::Z; 3];
        let mut carry = Trit::Z;

        for i in 0..3 {
            // Add trits and carry
            let (sum1, carry1) = a[i].add_with_carry(b[i]);
            let (sum2, carry2) = sum1.add_with_carry(carry);

            result[i] = sum2;
            // Combine carries
            let (carry_sum, _) = carry1.add_with_carry(carry2);
            carry = carry_sum;
        }

        (Self::from_trits(result), carry)
    }
}

impl Mul for Tryte3 {
    type Output = (Self, Self);

    /// Multiply two trytes, returning (low, high) result.
    ///
    /// The full result is `low + high * 27`.
    fn mul(self, other: Self) -> Self::Output {
        let product = self.value() * other.value();

        // Balanced ternary division for 6-trit result
        let low_val = ((product % 27) + 27 + 13) % 27 - 13;
        let high_val = (product - low_val) / 27;

        (
            Self::from_value(low_val).unwrap_or_else(|_| Self::zero()),
            Self::from_value(high_val.clamp(TRYTE3_MIN, TRYTE3_MAX))
                .unwrap_or_else(|_| Self::zero()),
        )
    }
}

impl fmt::Debug for Tryte3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let trits = self.to_trits();
        write!(
            f,
            "Tryte3({}{}{} = {})",
            trits[2],
            trits[1],
            trits[0],
            self.value()
        )
    }
}

impl fmt::Display for Tryte3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl TryFrom<i32> for Tryte3 {
    type Error = TernaryError;

    fn try_from(value: i32) -> Result<Self> {
        Self::from_value(value)
    }
}

impl From<Tryte3> for i32 {
    fn from(tryte: Tryte3) -> Self {
        tryte.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tryte_range() {
        // Test all valid values
        for v in TRYTE3_MIN..=TRYTE3_MAX {
            let t = Tryte3::from_value(v).expect("valid value");
            assert_eq!(t.value(), v);
        }

        // Test invalid values
        assert!(Tryte3::from_value(TRYTE3_MIN - 1).is_err());
        assert!(Tryte3::from_value(TRYTE3_MAX + 1).is_err());
    }

    #[test]
    fn test_tryte_zero() {
        let z = Tryte3::zero();
        assert_eq!(z.value(), 0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_tryte_trits_roundtrip() {
        for v in TRYTE3_MIN..=TRYTE3_MAX {
            let t = Tryte3::from_value(v).unwrap();
            let trits = t.to_trits();
            let reconstructed = Tryte3::from_trits(trits);
            assert_eq!(reconstructed.value(), v);
        }
    }

    #[test]
    fn test_tryte_negation() {
        for v in TRYTE3_MIN..=TRYTE3_MAX {
            let t = Tryte3::from_value(v).unwrap();
            let neg = -t;
            assert_eq!(neg.value(), -v);
        }
    }

    #[test]
    fn test_tryte_addition() {
        // Test some specific additions
        let a = Tryte3::from_value(5).unwrap();
        let b = Tryte3::from_value(3).unwrap();
        let (result, carry) = a + b;
        assert_eq!(result.value() + carry.value() as i32 * 27, 8);

        // Test with overflow
        let a = Tryte3::from_value(13).unwrap();
        let b = Tryte3::from_value(1).unwrap();
        let (result, carry) = a + b;
        // 13 + 1 = 14, which needs carry
        let total = result.value() + carry.value() as i32 * 27;
        assert_eq!(total, 14);
    }

    #[test]
    fn test_tryte_multiplication() {
        // Small values
        let a = Tryte3::from_value(3).unwrap();
        let b = Tryte3::from_value(4).unwrap();
        let (low, high) = a * b;
        let total = low.value() + high.value() * 27;
        assert_eq!(total, 12);

        // Test with larger values
        let a = Tryte3::from_value(10).unwrap();
        let b = Tryte3::from_value(10).unwrap();
        let (low, high) = a * b;
        let total = low.value() + high.value() * 27;
        assert_eq!(total, 100);
    }

    #[test]
    fn test_tryte_get_trit() {
        let t = Tryte3::from_trits([Trit::N, Trit::Z, Trit::P]);
        assert_eq!(t.get_trit(0), Trit::N);
        assert_eq!(t.get_trit(1), Trit::Z);
        assert_eq!(t.get_trit(2), Trit::P);
    }

    #[test]
    fn test_tryte_specific_values() {
        // Test value 5: should be -1 + 0*3 + 1*9 = -1 + 9 = 8? No...
        // Actually: 5 = 5*1 in balanced ternary:
        // 5 / 3 = 1 remainder 2 -> trit = -1, new value = 2
        // 2 / 3 = 0 remainder 2 -> trit = -1, new value = 1
        // 1 / 3 = 0 remainder 1 -> trit = +1, done
        // So 5 = [-1, -1, 1] = -1 + (-1)*3 + 1*9 = -1 - 3 + 9 = 5 ✓

        let t = Tryte3::from_value(5).unwrap();
        let trits = t.to_trits();
        let reconstructed =
            trits[0].value() as i32 + trits[1].value() as i32 * 3 + trits[2].value() as i32 * 9;
        assert_eq!(reconstructed, 5);
    }
}
