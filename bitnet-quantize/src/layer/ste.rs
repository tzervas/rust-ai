//! Straight-Through Estimator (STE) for gradient estimation.
//!
//! STE allows gradients to flow through non-differentiable quantization
//! by using the identity function in the backward pass.

use candle_core::Tensor;

use crate::error::Result;

/// Apply STE forward pass (quantize then dequantize).
///
/// This creates a "fake quantized" tensor that:
/// - Has values as if they were quantized and dequantized
/// - Can receive gradients normally during backprop
///
/// # Arguments
///
/// * `input` - Input tensor to quantize
/// * `scale` - Scale factor for quantization
/// * `min_val` - Minimum quantized value (e.g., -1 for ternary)
/// * `max_val` - Maximum quantized value (e.g., +1 for ternary)
///
/// # Errors
///
/// Returns error if tensor operations fail.
pub fn ste_forward(input: &Tensor, scale: f32, min_val: f32, max_val: f32) -> Result<Tensor> {
    // Quantize: round(x / scale) clamped to [min_val, max_val]
    let scaled = input.affine(1.0 / f64::from(scale), 0.0)?;

    // Round and clamp (fake quantization)
    let rounded = scaled.round()?;
    let clamped = rounded.clamp(min_val, max_val)?;

    // Dequantize: multiply by scale
    let dequantized = clamped.affine(f64::from(scale), 0.0)?;

    Ok(dequantized)
}

/// Compute STE backward pass (identity gradient).
///
/// In the backward pass, gradients flow through unchanged.
/// This is handled automatically by the tensor operations,
/// but this function is provided for clarity.
///
/// # Arguments
///
/// * `grad_output` - Gradient from the next layer
///
/// # Returns
///
/// The same gradient (identity function)
#[must_use]
pub fn ste_backward(grad_output: &Tensor) -> Tensor {
    grad_output.clone()
}

/// Apply ternary STE (quantize to {-1, 0, +1}).
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `scale` - Scale factor (typically AbsMean of the input)
///
/// # Errors
///
/// Returns error if tensor operations fail.
pub fn ternary_ste(input: &Tensor, scale: f32) -> Result<Tensor> {
    ste_forward(input, scale, -1.0, 1.0)
}

/// Apply INT8 STE (quantize to [-127, 127]).
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `scale` - Scale factor (typically AbsMax / 127)
///
/// # Errors
///
/// Returns error if tensor operations fail.
pub fn int8_ste(input: &Tensor, scale: f32) -> Result<Tensor> {
    ste_forward(input, scale, -127.0, 127.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ternary_ste() {
        let device = Device::Cpu;

        // Create input with various magnitudes
        let values: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let input = Tensor::from_vec(values, (5,), &device).unwrap();

        // Scale = 1.0 for simplicity
        let output = ternary_ste(&input, 1.0).unwrap();
        let result: Vec<f32> = output.to_vec1().unwrap();

        // Should be clamped to {-1, 0, 1}
        assert_eq!(result[0], -1.0); // -2.0 -> -1
        assert_eq!(result[1], -1.0); // -0.5 rounds to -1
        assert_eq!(result[2], 0.0); // 0 stays 0
        assert_eq!(result[3], 1.0); // 0.5 rounds to 1
        assert_eq!(result[4], 1.0); // 2.0 -> 1
    }

    #[test]
    fn test_int8_ste() {
        let device = Device::Cpu;

        let values: Vec<f32> = vec![-200.0, -50.0, 0.0, 50.0, 200.0];
        let input = Tensor::from_vec(values, (5,), &device).unwrap();

        let output = int8_ste(&input, 1.0).unwrap();
        let result: Vec<f32> = output.to_vec1().unwrap();

        // Should be clamped to [-127, 127]
        assert_eq!(result[0], -127.0);
        assert_eq!(result[1], -50.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 50.0);
        assert_eq!(result[4], 127.0);
    }

    #[test]
    fn test_ste_with_scale() {
        let device = Device::Cpu;

        let values: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
        let input = Tensor::from_vec(values, (4,), &device).unwrap();

        // Scale = 2.0 means values are divided by 2 before rounding
        let output = ternary_ste(&input, 2.0).unwrap();
        let result: Vec<f32> = output.to_vec1().unwrap();

        // 0.5/2 = 0.25 -> 0 -> 0
        // 1.0/2 = 0.5 -> 1 -> 2
        // 1.5/2 = 0.75 -> 1 -> 2
        // 2.0/2 = 1.0 -> 1 -> 2
        assert!((result[0] - 0.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 2.0).abs() < 0.01);
        assert!((result[3] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_ste_backward_identity() {
        let device = Device::Cpu;

        let grad = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let result = ste_backward(&grad);

        let grad_vec: Vec<f32> = grad.to_vec1().unwrap();
        let result_vec: Vec<f32> = result.to_vec1().unwrap();

        assert_eq!(grad_vec, result_vec);
    }
}
