//! Test data loading with token counting verification.
//!
//! This example demonstrates:
//! - Loading JSONL datasets
//! - Batch formation with padding
//! - Token counting per step
//! - Integration with DataLoader
//!
//! Usage:
//!   cargo run --example test_data_loading --release

use std::io::Write;
use tempfile::NamedTempFile;

use candle_core::Device;
use tritter_model_rs::data::{create_data_loader, DataConfig};

fn create_test_dataset() -> anyhow::Result<NamedTempFile> {
    let mut file = NamedTempFile::with_suffix(".jsonl")?;

    // Write test data
    writeln!(file, r#"{{"text": "Hello world from Rust"}}"#)?;
    writeln!(file, r#"{{"text": "This is a longer sequence with more tokens for testing"}}"#)?;
    writeln!(file, r#"{{"text": "Short text"}}"#)?;
    writeln!(file, r#"{{"text": "Another example with code: fn main() {{ println!(\"test\"); }}"}}"#)?;
    writeln!(file, r#"{{"text": "Testing tokenization and batch formation"}}"#)?;
    writeln!(file, r#"{{"text": "Multiple datasets can be loaded from directories"}}"#)?;
    writeln!(file, r#"{{"text": "This tests the data loading infrastructure"}}"#)?;
    writeln!(file, r#"{{"text": "Final example for completeness"}}"#)?;

    file.flush()?;
    Ok(file)
}

fn main() -> anyhow::Result<()> {
    println!("=== Data Loading Test ===\n");

    // Create test dataset
    let test_file = create_test_dataset()?;
    println!("Created test dataset: {:?}", test_file.path());

    // Configure data loading
    let config = DataConfig::default()
        .with_batch_size(2)
        .with_max_seq_length(128);

    println!("Data Config:");
    println!("  Batch size: {}", config.batch_size);
    println!("  Max sequence length: {}", config.max_seq_length);
    println!("  Workers: {}", config.num_workers);
    println!();

    // Create data loader
    let device = Device::Cpu;
    let loader = create_data_loader(test_file.path(), config.clone(), device)?;

    println!("Loading batches...");
    println!("{:-<80}", "");
    println!("{:>5} | {:>10} | {:>10} | {:>15} | {:>15}",
             "Batch", "Batch Size", "Seq Length", "Tokens/Step", "Total Tokens");
    println!("{:-<80}", "");

    let mut batch_count = 0;
    let mut total_tokens = 0u64;

    for batch_result in loader {
        let batch = batch_result?;

        let batch_size = batch.input_ids.dims()[0];
        let seq_len = batch.input_ids.dims()[1];

        // Calculate tokens for this step
        let tokens_this_step = (batch_size * seq_len) as u64;
        total_tokens += tokens_this_step;

        batch_count += 1;

        println!("{:>5} | {:>10} | {:>10} | {:>15} | {:>15}",
                 batch_count,
                 batch_size,
                 seq_len,
                 tokens_this_step,
                 total_tokens);

        // Verify attention mask exists
        if batch.attention_mask.is_none() {
            println!("WARNING: Batch {} missing attention mask!", batch_count);
        }

        // Verify tensor shapes
        if let Some(mask) = &batch.attention_mask {
            let mask_dims = mask.dims();
            if mask_dims[0] != batch_size || mask_dims[1] != seq_len {
                println!("ERROR: Attention mask shape mismatch! Expected [{}, {}], got {:?}",
                         batch_size, seq_len, mask_dims);
            }
        }
    }

    println!("{:-<80}", "");
    println!("\n=== Summary ===");
    println!("Total batches: {}", batch_count);
    println!("Total tokens processed: {}", total_tokens);
    println!("Average tokens per batch: {}", if batch_count > 0 { total_tokens / batch_count as u64 } else { 0 });

    // Expected calculation
    let expected_batches = 8 / config.batch_size; // 8 examples / batch_size
    println!("\nExpected batches: {} (8 examples / batch_size {})", expected_batches, config.batch_size);

    if batch_count == expected_batches as u64 {
        println!("✓ Batch count matches expected");
    } else {
        println!("⚠ Batch count mismatch! Expected {}, got {}", expected_batches, batch_count);
    }

    println!("\n✓ Data loading test complete!");

    Ok(())
}
