//! Weight and activation quantization example.
//!
//! Run with: `cargo run --example quantization`

use bitnet_quantize::{
    dequantize_activations, dequantize_weights, quantize_activations, quantize_weights,
    BitNetConfig,
};
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== bitnet-quantize Quantization Example ===\n");

    let device = Device::Cpu;
    let config = BitNetConfig::default().with_group_size(64);

    // Weight quantization
    println!("1. Weight Quantization (AbsMean -> Ternary)");
    println!("   Method: W_q = round(W / mean(|W|)) clamped to {{-1, 0, +1}}\n");

    let weights = Tensor::randn(0.0f32, 0.5, (128, 256), &device)?;
    println!("   Original weight shape: {:?}", weights.shape());

    let weight_vec: Vec<f32> = weights.flatten_all()?.to_vec1()?;
    let orig_mean = weight_vec.iter().map(|x| x.abs()).sum::<f32>() / weight_vec.len() as f32;
    println!("   Original mean(|W|): {:.4}", orig_mean);

    let quantized_weights = quantize_weights(&weights, &config)?;
    println!(
        "   Quantized weight sparsity: {:.1}%",
        quantized_weights.sparsity() * 100.0
    );
    println!(
        "   Compression ratio: {:.2}x",
        quantized_weights.compression_ratio()
    );
    println!(
        "   Number of scale groups: {}",
        quantized_weights.scales.len()
    );

    let restored_weights = dequantize_weights(&quantized_weights, &device)?;
    println!("   Restored weight shape: {:?}", restored_weights.shape());

    // Compute reconstruction error
    let diff = weights.sub(&restored_weights)?;
    let diff_vec: Vec<f32> = diff.flatten_all()?.to_vec1()?;
    let mse: f32 = diff_vec.iter().map(|x| x * x).sum::<f32>() / diff_vec.len() as f32;
    println!("   Reconstruction MSE: {:.6}", mse);

    // Activation quantization
    println!("\n2. Activation Quantization (AbsMax -> INT8)");
    println!("   Method: X_q = round(X * 127 / max(|X|)) clamped to [-127, 127]\n");

    let activations = Tensor::randn(0.0f32, 2.0, (4, 64), &device)?;
    println!("   Original activation shape: {:?}", activations.shape());

    let act_vec: Vec<f32> = activations.flatten_all()?.to_vec1()?;
    let orig_max = act_vec.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("   Original max(|X|): {:.4}", orig_max);

    let quantized_acts = quantize_activations(&activations, &config)?;
    println!("   Quantized shape: {:?}", quantized_acts.shape);
    println!("   Per-token scales: {:?}", quantized_acts.scales);

    // Check that values are in INT8 range
    let in_range = quantized_acts
        .data
        .iter()
        .all(|&x| (-127..=127).contains(&(x as i16)));
    println!("   All values in [-127, 127]: {}", in_range);

    let restored_acts = dequantize_activations(&quantized_acts, &device)?;
    println!("   Restored activation shape: {:?}", restored_acts.shape());

    // Compute reconstruction error
    let act_diff = activations.sub(&restored_acts)?;
    let act_diff_vec: Vec<f32> = act_diff.flatten_all()?.to_vec1()?;
    let act_mse: f32 = act_diff_vec.iter().map(|x| x * x).sum::<f32>() / act_diff_vec.len() as f32;
    println!("   Reconstruction MSE: {:.6}", act_mse);

    // Compare compression
    println!("\n3. Compression Summary");
    let orig_weight_bytes = 128 * 256 * 4; // FP32
    let quant_weight_bits = 128 * 256 * 2; // 2 bits per weight
    let scale_bytes = quantized_weights.scales.len() * 4; // FP32 scales
    let total_quant_bytes = quant_weight_bits / 8 + scale_bytes;
    println!("   Original weight size: {} bytes", orig_weight_bytes);
    println!(
        "   Quantized weight size: {} bytes (incl. scales)",
        total_quant_bytes
    );
    println!(
        "   Actual compression: {:.2}x",
        orig_weight_bytes as f32 / total_quant_bytes as f32
    );

    println!("\nDone!");
    Ok(())
}
