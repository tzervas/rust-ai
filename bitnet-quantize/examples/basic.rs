//! Basic usage of bitnet-rs.
//!
//! Run with: `cargo run --example basic`

use bitnet_quantize::{BitLinear, BitNetConfig};
use candle_core::{Device, Tensor};
use candle_nn::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== bitnet-quantize Basic Example ===\n");

    let device = Device::Cpu;

    // Create configuration
    println!("1. Configuration");
    let config = BitNetConfig::default().with_group_size(64);
    println!("   Group size: {}", config.group_size);
    println!("   Activation bits: {}", config.activation_bits);
    println!("   Per-token activation: {}", config.per_token_activation);

    // Create a weight matrix (simulating a linear layer)
    println!("\n2. Creating BitLinear layer");
    let weight = Tensor::randn(0.0f32, 0.5, (256, 512), &device)?;
    let bias = Tensor::zeros((256,), candle_core::DType::F32, &device)?;

    let layer = BitLinear::from_weight(&weight, Some(&bias), &config)?;

    println!("   Input features: {}", layer.in_features());
    println!("   Output features: {}", layer.out_features());
    println!("   Weight sparsity: {:.1}%", layer.sparsity() * 100.0);
    println!("   Compression ratio: {:.2}x", layer.compression_ratio());

    // Forward pass
    println!("\n3. Forward pass");
    let batch_size = 4;
    let input = Tensor::randn(0.0f32, 1.0, (batch_size, 512), &device)?;
    println!("   Input shape: {:?}", input.shape());

    let output = layer.forward(&input)?;
    println!("   Output shape: {:?}", output.shape());

    // Get some statistics
    let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;
    let mean: f32 = output_vec.iter().sum::<f32>() / output_vec.len() as f32;
    let max = output_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = output_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "   Output stats: mean={:.4}, min={:.4}, max={:.4}",
        mean, min, max
    );

    // 3D input (batch, seq_len, hidden)
    println!("\n4. 3D input (sequence model)");
    let seq_input = Tensor::randn(0.0f32, 1.0, (2, 16, 512), &device)?;
    println!("   Input shape: {:?}", seq_input.shape());

    let seq_output = layer.forward(&seq_input)?;
    println!("   Output shape: {:?}", seq_output.shape());

    println!("\nDone!");
    Ok(())
}
