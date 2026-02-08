//! GRU forward pass using Burn tensors with CubeCL backend.
//!
//! This module provides an alternative GPU implementation using Burn's high-level
//! tensor API, which handles CubeCL buffer management internally.
//!
//! # Approach
//!
//! Instead of using low-level CubeCL buffer APIs directly, this implementation:
//! 1. Converts input data to Burn tensors
//! 2. Uses Burn's CudaBackend which internally uses CubeCL
//! 3. Implements GRU operations as Burn tensor operations
//! 4. Returns results as Vec<f32>
//!
//! This approach avoids the complexity of manual buffer management while still
//! leveraging GPU acceleration.

use crate::error::{HybridResult, HybridTrainingError};
use crate::gpu::kernels::gru::{GpuGruWeights, GruKernelConfig};

#[cfg(all(feature = "cuda", feature = "candle"))]
use burn::tensor::{backend::Backend, Tensor, TensorData};

/// GRU forward pass using Burn tensors.
///
/// This is an alternative to the low-level CubeCL implementation that uses
/// Burn's high-level tensor API with GPU backend.
#[cfg(all(feature = "cuda", feature = "candle"))]
pub fn gru_forward_burn<B: Backend>(
    config: &GruKernelConfig,
    weights: &GpuGruWeights,
    hidden: &[f32],
    input: &[f32],
    device: &B::Device,
) -> HybridResult<Vec<f32>> {
    // Validate dimensions
    if hidden.len() != config.hidden_dim {
        return Err((
            HybridTrainingError::ConfigError {
                detail: format!(
                    "hidden size mismatch: expected {}, got {}",
                    config.hidden_dim,
                    hidden.len()
                ),
            },
            None,
        ));
    }

    if input.len() != config.input_dim {
        return Err((
            HybridTrainingError::ConfigError {
                detail: format!(
                    "input size mismatch: expected {}, got {}",
                    config.input_dim,
                    input.len()
                ),
            },
            None,
        ));
    }

    let _span = tracing::debug_span!(
        "gru_forward_burn",
        hidden_dim = config.hidden_dim,
        input_dim = config.input_dim
    )
    .entered();

    // Convert to Burn tensors
    let h_data = TensorData::from(hidden);
    let x_data = TensorData::from(input);
    let h = Tensor::<B, 1>::from_data(h_data, device);
    let x = Tensor::<B, 1>::from_data(x_data, device);

    // Create weight matrices as 1D tensors then reshape
    let w_z_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.w_z[..]), device);
    let w_z = w_z_1d.reshape([config.hidden_dim, config.input_dim]);

    let u_z_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.u_z[..]), device);
    let u_z = u_z_1d.reshape([config.hidden_dim, config.hidden_dim]);

    let b_z = Tensor::<B, 1>::from_data(TensorData::from(&weights.b_z[..]), device);

    let w_r_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.w_r[..]), device);
    let w_r = w_r_1d.reshape([config.hidden_dim, config.input_dim]);

    let u_r_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.u_r[..]), device);
    let u_r = u_r_1d.reshape([config.hidden_dim, config.hidden_dim]);

    let b_r = Tensor::<B, 1>::from_data(TensorData::from(&weights.b_r[..]), device);

    let w_h_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.w_h[..]), device);
    let w_h = w_h_1d.reshape([config.hidden_dim, config.input_dim]);

    let u_h_1d = Tensor::<B, 1>::from_data(TensorData::from(&weights.u_h[..]), device);
    let u_h = u_h_1d.reshape([config.hidden_dim, config.hidden_dim]);

    let b_h = Tensor::<B, 1>::from_data(TensorData::from(&weights.b_h[..]), device);

    // Compute update gate: z = σ(W_z·x + U_z·h + b_z)
    use burn::tensor::activation;
    let z_input = w_z.matmul(x.clone().unsqueeze());
    let z_hidden = u_z.matmul(h.clone().unsqueeze());
    let z_pre = z_input.squeeze::<1>() + z_hidden.squeeze::<1>() + b_z;
    let z = activation::sigmoid(z_pre);

    // Compute reset gate: r = σ(W_r·x + U_r·h + b_r)
    let r_input = w_r.matmul(x.clone().unsqueeze());
    let r_hidden = u_r.matmul(h.clone().unsqueeze());
    let r_pre = r_input.squeeze::<1>() + r_hidden.squeeze::<1>() + b_r;
    let r = activation::sigmoid(r_pre);

    // Compute candidate: h̃ = tanh(W_h·x + U_h·(r⊙h) + b_h)
    let r_h = r * h.clone();
    let h_cand_input = w_h.matmul(x.unsqueeze());
    let h_cand_hidden = u_h.matmul(r_h.unsqueeze());
    let h_cand_pre = h_cand_input.squeeze::<1>() + h_cand_hidden.squeeze::<1>() + b_h;
    let h_cand = activation::tanh(h_cand_pre);

    // Compute output: h' = (1-z)⊙h + z⊙h̃
    let one = Tensor::<B, 1>::ones(burn::tensor::Shape::from([config.hidden_dim]), device);
    let h_new = (one - z.clone()) * h + z * h_cand;

    // Convert back to Vec<f32>
    let output_data = h_new.into_data();
    Ok(output_data.to_vec().unwrap())
}

/// Non-CUDA/Candle fallback.
#[cfg(not(all(feature = "cuda", feature = "candle")))]
pub fn gru_forward_burn<B>(
    _config: &GruKernelConfig,
    _weights: &GpuGruWeights,
    _hidden: &[f32],
    _input: &[f32],
    _device: &B,
) -> HybridResult<Vec<f32>> {
    Err((
        HybridTrainingError::GpuError {
            detail: "Burn-based GRU requires both 'cuda' and 'candle' features".to_string(),
        },
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA + Candle features
    fn test_gru_burn_output_dims() {
        // Placeholder - would test with actual Burn backend
    }
}
