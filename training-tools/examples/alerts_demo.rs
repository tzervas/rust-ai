//! Demonstrates the training alerts and notification system
//!
//! Run with: cargo run --example alerts_demo

use training_tools::alerts::{Alert, AlertCondition, AlertManager, AlertSeverity, MetricsSnapshot};

fn main() {
    println!("=== Training Alerts Demo ===\n");

    // Create alert manager with default conditions
    let mut manager = AlertManager::new();
    println!("Created AlertManager with default conditions:");
    println!("  - Loss spike (2x threshold)");
    println!("  - Plateau (100 steps)");
    println!("  - Diverging");
    println!("  - Memory high (90%)");
    println!("  - Gradient exploding (>100.0)");
    println!("  - Gradient vanishing (<1e-6)\n");

    // Simulate normal training for baseline
    println!("Simulating normal training (steps 0-50)...");
    for step in 0..50 {
        let metrics = MetricsSnapshot {
            step,
            loss: 2.0 + (step as f32 * -0.01), // Decreasing loss
            gradient_norm: Some(1.0),
            memory_usage: Some(0.5),
        };
        manager.check_metrics(metrics);
    }
    println!("  No alerts triggered during normal training\n");

    // Simulate loss spike
    println!("Simulating loss spike at step 50...");
    let metrics = MetricsSnapshot {
        step: 50,
        loss: 8.0, // 4x the recent baseline
        gradient_norm: Some(1.0),
        memory_usage: Some(0.5),
    };
    let alerts = manager.check_metrics(metrics);
    for alert in &alerts {
        println!("  {}", alert.format());
    }
    println!();

    // Simulate gradient explosion
    println!("Simulating gradient explosion at step 51...");
    let metrics = MetricsSnapshot {
        step: 51,
        loss: 2.0,
        gradient_norm: Some(250.0), // Above threshold
        memory_usage: Some(0.5),
    };
    let alerts = manager.check_metrics(metrics);
    for alert in &alerts {
        println!("  {}", alert.format());
    }
    println!();

    // Simulate high memory usage
    println!("Simulating high memory usage at step 52...");
    let metrics = MetricsSnapshot {
        step: 52,
        loss: 2.0,
        gradient_norm: Some(1.0),
        memory_usage: Some(0.95), // 95% usage
    };
    let alerts = manager.check_metrics(metrics);
    for alert in &alerts {
        println!("  {}", alert.format());
    }
    println!();

    // Simulate plateau by not improving for many steps
    println!("Simulating plateau (no improvement for 110 steps)...");
    for step in 53..163 {
        let metrics = MetricsSnapshot {
            step,
            loss: 2.5, // Worse than best
            gradient_norm: Some(1.0),
            memory_usage: Some(0.5),
        };
        let alerts = manager.check_metrics(metrics);
        if !alerts.is_empty() {
            for alert in &alerts {
                println!("  {}", alert.format());
            }
        }
    }
    println!();

    // Simulate gradient vanishing
    println!("Simulating gradient vanishing at step 163...");
    let metrics = MetricsSnapshot {
        step: 163,
        loss: 2.0,
        gradient_norm: Some(1e-7), // Very small
        memory_usage: Some(0.5),
    };
    let alerts = manager.check_metrics(metrics);
    for alert in &alerts {
        println!("  {}", alert.format());
    }
    println!();

    // Simulate diverging training
    println!("Simulating diverging training (20 consecutive increases)...");
    for step in 164..184 {
        let metrics = MetricsSnapshot {
            step,
            loss: 2.0 + ((step - 164) as f32 * 0.1), // Increasing loss
            gradient_norm: Some(1.0),
            memory_usage: Some(0.5),
        };
        let alerts = manager.check_metrics(metrics);
        if !alerts.is_empty() {
            for alert in &alerts {
                println!("  {}", alert.format());
            }
        }
    }
    println!();

    // Display summary
    println!("=== Alert Summary ===");
    println!("Total alerts: {}", manager.count());
    println!("Info: {}", manager.count_by_severity(AlertSeverity::Info));
    println!(
        "Warning: {}",
        manager.count_by_severity(AlertSeverity::Warning)
    );
    println!(
        "Critical: {}",
        manager.count_by_severity(AlertSeverity::Critical)
    );
    println!();

    // Show all critical alerts
    println!("=== Critical Alerts ===");
    for alert in manager.get_by_severity(AlertSeverity::Critical) {
        println!("{}", alert.format());
    }
    println!();

    // Demonstrate acknowledgment
    println!(
        "Unacknowledged alerts: {}",
        manager.get_unacknowledged().count()
    );
    manager.acknowledge_all();
    println!(
        "After acknowledging all: {}",
        manager.get_unacknowledged().count()
    );
    println!();

    // Demonstrate JSON export
    println!("=== JSON Export ===");
    match manager.export_json() {
        Ok(json) => {
            let preview = if json.len() > 200 {
                format!("{}...", &json[..200])
            } else {
                json
            };
            println!("{}", preview);
        }
        Err(e) => println!("Failed to export: {}", e),
    }
    println!();

    // Demonstrate custom conditions
    println!("=== Custom Alert Manager ===");
    let custom_conditions = vec![
        AlertCondition::LossSpike { threshold: 1.5 }, // More sensitive
        AlertCondition::GradientExploding { threshold: 50.0 }, // Lower threshold
    ];
    let mut custom_manager = AlertManager::with_conditions(custom_conditions);
    custom_manager.set_max_history(100); // Smaller history

    println!("Created custom manager with:");
    println!("  - Sensitive loss spike (1.5x)");
    println!("  - Lower gradient threshold (50.0)");
    println!("  - Max history: 100");

    println!("\n=== Demo Complete ===");
}
