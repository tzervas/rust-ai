//! VRAM budget detection example.
//!
//! This example demonstrates how to detect available VRAM and get
//! recommended training configurations based on model size.
//!
//! # Running
//!
//! ```bash
//! cargo run --example check_vram_budget
//! ```

use hybrid_predict_trainer_rs::vram_budget::VramBudget;

fn main() {
    println!("🔍 Detecting VRAM Budget...\n");

    match VramBudget::detect() {
        Ok(budget) => {
            budget.print_summary();
            println!();

            // Test configurations for different model sizes
            println!("📊 Recommended Configurations:\n");

            // SimpleMLP (100 MB)
            println!("═══ SimpleMLP (~100 MB) ═══");
            let config = budget.recommend_config_for_model(100, 100_000);
            config.print_summary();
            println!();

            // SmolLM2-135M (~270 MB)
            println!("═══ SmolLM2-135M (~270 MB) ═══");
            let config = budget.recommend_config_for_model(270, 135_000_000);
            config.print_summary();
            println!();

            // TinyLlama-1.1B (~2200 MB)
            println!("═══ TinyLlama-1.1B (~2200 MB) ═══");
            let config = budget.recommend_config_for_model(2200, 1_100_000_000);
            config.print_summary();
            println!();

            // Safety recommendations
            println!("⚠️  Safety Recommendations:");
            println!("   • Always monitor VRAM during first 100 training steps");
            println!(
                "   • Keep peak usage below {} MB",
                budget.available_for_training_mb
            );
            println!("   • Use: nvidia-smi --loop=1 to watch in real-time");
            println!("   • If OOM occurs: reduce batch_size by 50%");
        }
        Err(e) => {
            eprintln!("❌ Failed to detect VRAM: {:?}", e);
            eprintln!("\nMake sure:");
            eprintln!("  • nvidia-smi is installed");
            eprintln!("  • NVIDIA drivers are loaded");
            eprintln!("  • GPU is accessible");
        }
    }
}
