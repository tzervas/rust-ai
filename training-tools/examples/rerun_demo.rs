//! Demo of the Rerun visualization integration.
//!
//! Run with:
//! ```bash
//! # Start the Rerun viewer first
//! rerun
//!
//! # Then run this example
//! cargo run --example rerun_demo --features rerun
//! ```

#[cfg(feature = "rerun")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use training_tools::rerun_viz::{LayerInfo, LayerType, RerunLogger, TrainingPhaseLog};

    // Create a logger connected to the Rerun viewer
    let logger = RerunLogger::new("training_demo")?;

    println!("Connected to Rerun viewer. Streaming training metrics...");

    // Simulate training loop
    for step in 0..1000 {
        // Simulate loss curve (exponential decay with noise)
        let base_loss = 5.0 * (-0.003 * step as f32).exp();
        let noise = (step as f32 * 0.1).sin() * 0.1;
        let loss = base_loss + noise + 0.1;

        // Simulate learning rate schedule (warmup + cosine decay)
        let warmup_steps = 100.0;
        let lr = if step < 100 {
            1e-4 * (step as f32 / warmup_steps)
        } else {
            let progress = (step - 100) as f32 / 900.0;
            1e-4 * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
        };

        // Simulate gradient norm
        let grad_norm = 1.0 + (step as f32 * 0.05).cos() * 0.3;

        // Log the training step
        logger.log_step(step, loss, lr, grad_norm);

        // Log extended metrics every 10 steps
        if step % 10 == 0 {
            use training_tools::rerun_viz::MetricsLogger;
            let rec = logger.recording_stream();
            let metrics = MetricsLogger::new(rec);

            let phase = if step < 100 {
                TrainingPhaseLog::Warmup
            } else if step < 800 {
                TrainingPhaseLog::Training
            } else {
                TrainingPhaseLog::Cooldown
            };

            metrics.log_step_extended(step, loss, lr, grad_norm, phase, 50.0);
        }

        // Log embeddings periodically
        if step % 100 == 0 && step > 0 {
            // Generate sample embeddings (simulating token embeddings evolving)
            let embeddings: Vec<Vec<f32>> = (0..50)
                .map(|i| {
                    vec![
                        (i as f32 * 0.1).sin() + (step as f32 * 0.01),
                        (i as f32 * 0.1).cos() + (step as f32 * 0.01),
                        i as f32 / 50.0 - 0.5,
                    ]
                })
                .collect();

            let labels: Vec<String> = (0..50).map(|i| format!("token_{}", i)).collect();

            logger.log_embeddings(&embeddings, &labels);
        }

        // Log attention patterns at specific steps
        if step == 100 || step == 500 || step == 900 {
            // Simulate attention weights (diagonal-ish pattern with some noise)
            let seq_len = 16;
            let weights: Vec<Vec<f32>> = (0..seq_len)
                .map(|i| {
                    (0..seq_len)
                        .map(|j| {
                            let distance = (i as i32 - j as i32).abs() as f32;
                            let base = (-distance * 0.3).exp();
                            (base + (step as f32 * 0.001)).min(1.0)
                        })
                        .collect()
                })
                .collect();

            logger.log_attention(0, 0, &weights);
        }

        // Small delay to simulate real training
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Log network architecture at the end
    let layers = vec![
        LayerInfo::new("embed", LayerType::Embedding, 50257, 768),
        LayerInfo::new("attn_0", LayerType::Attention, 768, 768).with_params(2359296),
        LayerInfo::new("mlp_0", LayerType::MLP, 768, 3072).with_params(4718592),
        LayerInfo::new("ln_0", LayerType::LayerNorm, 768, 768).with_params(1536),
        LayerInfo::new("attn_1", LayerType::Attention, 768, 768).with_params(2359296),
        LayerInfo::new("mlp_1", LayerType::MLP, 768, 3072).with_params(4718592),
        LayerInfo::new("ln_f", LayerType::LayerNorm, 768, 768).with_params(1536),
        LayerInfo::new("lm_head", LayerType::Head, 768, 50257).with_params(38597376),
    ];

    logger.log_network(&layers);

    // Log a loss landscape
    let mut surface = Vec::new();
    for i in 0..30 {
        let row: Vec<f32> = (0..30)
            .map(|j| {
                let x = (i as f32 - 15.0) / 5.0;
                let y = (j as f32 - 15.0) / 5.0;
                // Bowl shape with some ridges
                x * x + y * y + 0.3 * (3.0 * x).sin() * (3.0 * y).sin()
            })
            .collect();
        surface.push(row);
    }

    // Optimization trajectory (gradient descent path)
    let trajectory: Vec<[f32; 3]> = (0..50)
        .map(|i| {
            let t = i as f32 / 50.0;
            let x = 2.0 * (1.0 - t) - 0.1 * t;
            let y = 2.0 * (1.0 - t) + 0.1 * t;
            let z = x * x + y * y;
            [x, y, z]
        })
        .collect();

    logger.log_landscape(&surface, &trajectory);

    println!("Demo complete! Check the Rerun viewer.");

    Ok(())
}

#[cfg(not(feature = "rerun"))]
fn main() {
    eprintln!("This example requires the 'rerun' feature.");
    eprintln!("Run with: cargo run --example rerun_demo --features rerun");
    std::process::exit(1);
}
