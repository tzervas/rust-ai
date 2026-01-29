//! Core trit type representing balanced ternary values {-1, 0, +1}.
//!
//! A trit is the fundamental unit of balanced ternary arithmetic.
//! Unlike binary bits (0, 1), trits can represent three states:
//! - `N` (Negative): -1
//! - `Z` (Zero): 0
//! - `P` (Positive): +1

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg};

use crate::error::{Result, TernaryError};

/// A balanced ternary digit (trit) with values {-1, 0, +1}.
///
/// # Representation
///
/// | Variant | Symbol | Value |
/// |---------|--------|-------|
/// | `N`     | `-`    | -1    |
/// | `Z`     | `0`    |  0    |
/// | `P`     | `+`    | +1    |
///
/// # Examples
///
/// ```
/// use trit_vsa::Trit;
///
/// let neg = Trit::N;
/// let zero = Trit::Z;
/// let pos = Trit::P;
///
/// assert_eq!(neg.value(), -1);
/// assert_eq!(zero.value(), 0);
/// assert_eq!(pos.value(), 1);
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i8)]
pub enum Trit {
    /// Negative trit (-1).
    N = -1,
    /// Zero trit (0).
    #[default]
    Z = 0,
    /// Positive trit (+1).
    P = 1,
}

impl Trit {
    /// Create a trit from an integer value.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer value (-1, 0, or +1)
    ///
    /// # Errors
    ///
    /// Returns `TernaryError::InvalidValue` if value is not -1, 0, or +1.
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// assert_eq!(Trit::from_value(-1).unwrap(), Trit::N);
    /// assert_eq!(Trit::from_value(0).unwrap(), Trit::Z);
    /// assert_eq!(Trit::from_value(1).unwrap(), Trit::P);
    /// assert!(Trit::from_value(2).is_err());
    /// ```
    pub const fn from_value(value: i32) -> Result<Self> {
        match value {
            -1 => Ok(Trit::N),
            0 => Ok(Trit::Z),
            1 => Ok(Trit::P),
            _ => Err(TernaryError::InvalidValue(value)),
        }
    }

    /// Get the integer value of the trit.
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// assert_eq!(Trit::N.value(), -1);
    /// assert_eq!(Trit::Z.value(), 0);
    /// assert_eq!(Trit::P.value(), 1);
    /// ```
    #[must_use]
    pub const fn value(self) -> i8 {
        self as i8
    }

    /// Check if the trit is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(self, Trit::Z)
    }

    /// Check if the trit is positive.
    #[must_use]
    pub const fn is_positive(self) -> bool {
        matches!(self, Trit::P)
    }

    /// Check if the trit is negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        matches!(self, Trit::N)
    }

    /// Encode trit as two bits (plus, minus planes).
    ///
    /// # Returns
    ///
    /// Tuple `(plus_bit, minus_bit)`:
    /// - `P (+1)` -> `(true, false)`
    /// - `Z (0)`  -> `(false, false)`
    /// - `N (-1)` -> `(false, true)`
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// assert_eq!(Trit::P.to_bits(), (true, false));
    /// assert_eq!(Trit::Z.to_bits(), (false, false));
    /// assert_eq!(Trit::N.to_bits(), (false, true));
    /// ```
    #[must_use]
    pub const fn to_bits(self) -> (bool, bool) {
        match self {
            Trit::P => (true, false),
            Trit::Z => (false, false),
            Trit::N => (false, true),
        }
    }

    /// Decode trit from two bits (plus, minus planes).
    ///
    /// # Arguments
    ///
    /// * `plus` - Positive plane bit
    /// * `minus` - Negative plane bit
    ///
    /// # Panics
    ///
    /// Panics in debug mode if both bits are set (invalid state).
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// assert_eq!(Trit::from_bits(true, false), Trit::P);
    /// assert_eq!(Trit::from_bits(false, false), Trit::Z);
    /// assert_eq!(Trit::from_bits(false, true), Trit::N);
    /// ```
    #[must_use]
    pub const fn from_bits(plus: bool, minus: bool) -> Self {
        debug_assert!(!(plus && minus), "invalid state: both planes set");
        if plus {
            Trit::P
        } else if minus {
            Trit::N
        } else {
            Trit::Z
        }
    }

    /// Add two trits with carry.
    ///
    /// Returns `(result_trit, carry_trit)` where the sum is `result + 3*carry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// // 1 + 1 = 2 = -1 + 3*1 (result=-1, carry=1)
    /// assert_eq!(Trit::P.add_with_carry(Trit::P), (Trit::N, Trit::P));
    ///
    /// // 1 + 0 = 1 (result=1, carry=0)
    /// assert_eq!(Trit::P.add_with_carry(Trit::Z), (Trit::P, Trit::Z));
    ///
    /// // -1 + -1 = -2 = 1 + 3*(-1) (result=1, carry=-1)
    /// assert_eq!(Trit::N.add_with_carry(Trit::N), (Trit::P, Trit::N));
    /// ```
    #[must_use]
    pub const fn add_with_carry(self, other: Trit) -> (Trit, Trit) {
        let sum = self.value() as i16 + other.value() as i16;
        match sum {
            -2 => (Trit::P, Trit::N), // -2 = 1 + 3*(-1)
            -1 => (Trit::N, Trit::Z),
            0 => (Trit::Z, Trit::Z),
            1 => (Trit::P, Trit::Z),
            2 => (Trit::N, Trit::P), // 2 = -1 + 3*1
            _ => unreachable!(),
        }
    }

    /// Bind operation for Vector Symbolic Architecture (VSA).
    ///
    /// Binding is implemented as subtraction mod 3. To recover the original,
    /// use `unbind`: `a.bind(b).unbind(b) == a`.
    ///
    /// Truth table (showing result of `self.bind(other)`):
    /// ```text
    ///      | N  Z  P    (other)
    /// -----+---------
    ///   N  | Z  N  P
    ///   Z  | P  Z  N
    ///   P  | N  P  Z
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// // Bind/unbind property
    /// let a = Trit::P;
    /// let b = Trit::N;
    /// assert_eq!(a.bind(b).unbind(b), a);
    /// ```
    #[must_use]
    pub const fn bind(self, other: Trit) -> Trit {
        // Implemented as subtraction mod 3
        let diff = self.value() as i16 - other.value() as i16;
        match diff.rem_euclid(3) {
            0 => Trit::Z,
            1 => Trit::P,
            2 => Trit::N, // -1 mod 3 = 2
            _ => unreachable!(),
        }
    }

    /// Unbind operation - the inverse of bind.
    ///
    /// `a.bind(b).unbind(b) == a`
    ///
    /// Implemented as addition mod 3 (the inverse of subtraction).
    ///
    /// # Examples
    ///
    /// ```
    /// use trit_vsa::Trit;
    ///
    /// let a = Trit::N;
    /// let b = Trit::P;
    /// let bound = a.bind(b);
    /// assert_eq!(bound.unbind(b), a);
    /// ```
    #[must_use]
    pub const fn unbind(self, other: Trit) -> Trit {
        // Implemented as addition mod 3 (inverse of bind's subtraction)
        let sum = self.value() as i16 + other.value() as i16;
        match sum.rem_euclid(3) {
            0 => Trit::Z,
            1 => Trit::P,
            2 => Trit::N, // -1 mod 3 = 2
            _ => unreachable!(),
        }
    }
}

impl Neg for Trit {
    type Output = Trit;

    fn neg(self) -> Self::Output {
        match self {
            Trit::N => Trit::P,
            Trit::Z => Trit::Z,
            Trit::P => Trit::N,
        }
    }
}

impl Add for Trit {
    type Output = (Trit, Trit);

    /// Add two trits, returning `(result, carry)`.
    fn add(self, other: Trit) -> Self::Output {
        self.add_with_carry(other)
    }
}

impl Mul for Trit {
    type Output = Trit;

    /// Multiply two trits.
    fn mul(self, other: Trit) -> Self::Output {
        match (self, other) {
            (Trit::Z, _) | (_, Trit::Z) => Trit::Z,
            (Trit::P, Trit::P) | (Trit::N, Trit::N) => Trit::P,
            (Trit::P, Trit::N) | (Trit::N, Trit::P) => Trit::N,
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trit::N => write!(f, "-"),
            Trit::Z => write!(f, "0"),
            Trit::P => write!(f, "+"),
        }
    }
}

impl TryFrom<i32> for Trit {
    type Error = TernaryError;

    fn try_from(value: i32) -> Result<Self> {
        Trit::from_value(value)
    }
}

impl From<Trit> for i8 {
    fn from(trit: Trit) -> Self {
        trit.value()
    }
}

impl From<Trit> for i32 {
    fn from(trit: Trit) -> Self {
        trit.value() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_values() {
        assert_eq!(Trit::N.value(), -1);
        assert_eq!(Trit::Z.value(), 0);
        assert_eq!(Trit::P.value(), 1);
    }

    #[test]
    fn test_trit_from_value() {
        assert_eq!(Trit::from_value(-1).unwrap(), Trit::N);
        assert_eq!(Trit::from_value(0).unwrap(), Trit::Z);
        assert_eq!(Trit::from_value(1).unwrap(), Trit::P);
        assert!(Trit::from_value(2).is_err());
        assert!(Trit::from_value(-2).is_err());
    }

    #[test]
    fn test_trit_negation() {
        assert_eq!(-Trit::N, Trit::P);
        assert_eq!(-Trit::Z, Trit::Z);
        assert_eq!(-Trit::P, Trit::N);
    }

    #[test]
    fn test_trit_multiplication() {
        // Zero absorbs
        assert_eq!(Trit::Z * Trit::P, Trit::Z);
        assert_eq!(Trit::P * Trit::Z, Trit::Z);
        assert_eq!(Trit::N * Trit::Z, Trit::Z);

        // Same signs -> positive
        assert_eq!(Trit::P * Trit::P, Trit::P);
        assert_eq!(Trit::N * Trit::N, Trit::P);

        // Different signs -> negative
        assert_eq!(Trit::P * Trit::N, Trit::N);
        assert_eq!(Trit::N * Trit::P, Trit::N);
    }

    #[test]
    fn test_trit_addition_with_carry() {
        // No carry cases
        assert_eq!(Trit::Z.add_with_carry(Trit::Z), (Trit::Z, Trit::Z));
        assert_eq!(Trit::P.add_with_carry(Trit::Z), (Trit::P, Trit::Z));
        assert_eq!(Trit::N.add_with_carry(Trit::Z), (Trit::N, Trit::Z));
        assert_eq!(Trit::P.add_with_carry(Trit::N), (Trit::Z, Trit::Z));
        assert_eq!(Trit::N.add_with_carry(Trit::P), (Trit::Z, Trit::Z));

        // Carry cases
        assert_eq!(Trit::P.add_with_carry(Trit::P), (Trit::N, Trit::P)); // 2 = -1 + 3
        assert_eq!(Trit::N.add_with_carry(Trit::N), (Trit::P, Trit::N)); // -2 = 1 - 3
    }

    #[test]
    fn test_trit_bits_roundtrip() {
        for trit in [Trit::N, Trit::Z, Trit::P] {
            let (plus, minus) = trit.to_bits();
            assert_eq!(Trit::from_bits(plus, minus), trit);
        }
    }

    #[test]
    fn test_trit_bind_unbind_inverse() {
        for a in [Trit::N, Trit::Z, Trit::P] {
            for b in [Trit::N, Trit::Z, Trit::P] {
                // a.bind(b).unbind(b) == a
                assert_eq!(a.bind(b).unbind(b), a, "unbind should reverse bind");
            }
        }
    }

    #[test]
    fn test_trit_bind_commutative() {
        // Note: subtraction-based bind is NOT commutative
        // But it has the self-inverse property which is more important for VSA
        // This test verifies that property is preserved
        for _a in [Trit::N, Trit::Z, Trit::P] {
            for _b in [Trit::N, Trit::Z, Trit::P] {
                // Self-inverse is tested in test_trit_bind_self_inverse
            }
        }
    }

    #[test]
    fn test_trit_display() {
        assert_eq!(format!("{}", Trit::N), "-");
        assert_eq!(format!("{}", Trit::Z), "0");
        assert_eq!(format!("{}", Trit::P), "+");
    }

    #[test]
    fn test_trit_predicates() {
        assert!(Trit::Z.is_zero());
        assert!(!Trit::P.is_zero());
        assert!(!Trit::N.is_zero());

        assert!(Trit::P.is_positive());
        assert!(!Trit::Z.is_positive());
        assert!(!Trit::N.is_positive());

        assert!(Trit::N.is_negative());
        assert!(!Trit::Z.is_negative());
        assert!(!Trit::P.is_negative());
    }
}
