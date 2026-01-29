//! GGUF export for BitNet models.
//!
//! This module provides export functionality using qlora-rs GGUF infrastructure.

#![cfg(feature = "gguf-export")]

use std::path::Path;

use crate::error::{BitNetError, Result};
use crate::layer::BitLinear;
use crate::quantization::dequantize_weights;

/// Export a BitLinear layer to GGUF format.
///
/// This dequantizes the ternary weights and re-quantizes using qlora-rs NF4
/// for GGUF compatibility.
///
/// # Arguments
///
/// * `layer` - BitLinear layer to export
/// * `output_path` - Path to write the GGUF file
/// * `layer_name` - Name for the layer in the GGUF file
///
/// # Errors
///
/// Returns error if export fails.
pub fn export_gguf<P: AsRef<Path>>(
    layer: &BitLinear,
    output_path: P,
    layer_name: &str,
) -> Result<()> {
    use candle_core::Device;

    let device = Device::Cpu;

    // Dequantize BitNet weights
    let weights = dequantize_weights(layer.quantized_weight(), &device)?;

    // Quantize to NF4 using qlora-rs
    let nf4_quantized = qlora_rs::quantize_nf4(&weights, 64)
        .map_err(|e| BitNetError::Serialization(e.to_string()))?;

    // Create GGUF metadata
    let metadata = qlora_rs::GgufMetadata {
        model_name: format!("bitnet-{}", layer_name),
        model_type: "bitnet".to_string(),
        model_size: nf4_quantized.numel(),
    };

    // Export using qlora-rs
    qlora_rs::export_gguf(&[(layer_name, &nf4_quantized)], Some(metadata), output_path)
        .map_err(|e| BitNetError::Serialization(e.to_string()))?;

    Ok(())
}
