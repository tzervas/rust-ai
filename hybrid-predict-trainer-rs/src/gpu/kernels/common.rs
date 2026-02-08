// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Common GPU kernels for GRU and RSSM operations.
//!
//! This module provides shared computational primitives used across
//! multiple GPU kernel implementations:
//!
//! - **Activation functions**: sigmoid, tanh
//! - **Linear algebra**: matrix-vector multiplication
//! - **Reductions**: sum, max operations
//!
//! # Phase 1 Note
//!
//! Currently contains CPU reference implementations.
//! Phase 2 will add actual CubeCL kernel implementations.

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
///
/// CPU reference implementation for validation.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x < 0.0 {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Hyperbolic tangent activation.
///
/// CPU reference implementation for validation.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Matrix-vector multiplication: y = A * x
///
/// CPU reference implementation for validation.
///
/// # Arguments
///
/// - `matrix`: Row-major matrix [m × n]
/// - `vector`: Input vector [n]
/// - `m`: Number of rows
/// - `n`: Number of columns
///
/// # Returns
///
/// Output vector [m]
pub fn matvec(matrix: &[f32], vector: &[f32], m: usize, n: usize) -> Vec<f32> {
    (0..m)
        .map(|i| {
            (0..n)
                .map(|j| matrix[i * n + j] * vector[j])
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_properties() {
        // Sigmoid(0) should be 0.5
        let sigmoid_0 = sigmoid(0.0);
        assert!((sigmoid_0 - 0.5).abs() < 1e-6);

        // Sigmoid is symmetric: σ(-x) = 1 - σ(x)
        let x = 2.0_f32;
        let sigmoid_x = sigmoid(x);
        let sigmoid_neg_x = sigmoid(-x);
        assert!((sigmoid_x + sigmoid_neg_x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_properties() {
        // tanh(0) = 0
        let tanh_0 = tanh(0.0);
        assert!(tanh_0.abs() < 1e-6);

        // tanh is odd: tanh(-x) = -tanh(x)
        let x = 1.5_f32;
        assert!((tanh(x) + tanh(-x)).abs() < 1e-6);

        // tanh saturates at ±1
        assert!(tanh(10.0) > 0.999);
        assert!(tanh(-10.0) < -0.999);
    }

    #[test]
    fn test_matvec() {
        // 2x3 matrix × 3-vector = 2-vector
        let matrix = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
        ];
        let vector = vec![1.0, 0.0, 1.0];

        let result = matvec(&matrix, &vector, 2, 3);

        // [1 2 3] · [1 0 1]' = 1*1 + 2*0 + 3*1 = 4
        // [4 5 6] · [1 0 1]' = 4*1 + 5*0 + 6*1 = 10
        assert!((result[0] - 4.0).abs() < 1e-5);
        assert!((result[1] - 10.0).abs() < 1e-5);
    }
}
