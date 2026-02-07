//! Loss Landscape 3D Visualization Demo
//!
//! Demonstrates how to:
//! - Generate a 3D loss landscape from weight perturbations
//! - Track optimization trajectory through the landscape
//! - Compute gradient fields for visualization
//! - Export mesh data for 3D rendering
//!
//! Run with: cargo run --example landscape_demo -p training-tools --no-default-features

use std::f32::consts::PI;
use training_tools::landscape::{
    DirectionMethod, GradientField, LandscapeConfig, LossLandscape, MeshExporter, TrajectoryTracker,
};

fn main() {
    println!("=== Loss Landscape 3D Visualization Demo ===\n");

    // Define weight dimensions (small for demo)
    let dim = 100;
    let center_weights: Vec<f32> = vec![0.0; dim];

    // Define a loss function: Rosenbrock function (banana-shaped valley)
    // L(x, y) = (1-x)^2 + 100*(y-x^2)^2
    // Extended to higher dimensions
    let rosenbrock = |weights: &[f32]| -> f32 {
        let mut loss = 0.0;
        for i in 0..(weights.len() - 1) {
            let x = weights[i];
            let y = weights[i + 1];
            loss += (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
        }
        loss / weights.len() as f32
    };

    // ========== 1. Generate Loss Landscape ==========
    println!("1. Generating loss landscape...");

    let config = LandscapeConfig {
        resolution: 50, // 50x50 grid
        range: 2.0,     // Perturbation range [-2, 2]
        direction_method: DirectionMethod::Random,
    };

    let mut landscape = LossLandscape::compute(&center_weights, rosenbrock, &config);

    println!("   Resolution: {}x{}", config.resolution, config.resolution);
    println!("   Range: [{}, {}]", -config.range, config.range);
    println!("   Min loss: {:.4}", landscape.min_loss());
    println!("   Max loss: {:.4}", landscape.max_loss());
    println!();

    // ========== 2. Simulate Optimization Trajectory ==========
    println!("2. Simulating optimization trajectory...");

    // Simulate gradient descent on a 2D projection
    let mut x = 1.5;
    let mut y = 1.5;
    let lr = 0.001;
    let steps = 100;

    for step in 0..steps {
        // Approximate loss at this point
        let loss = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);

        // Create weight vector at this trajectory point
        let mut weights = center_weights.clone();
        // Project onto the landscape's direction vectors
        let dir1 = landscape.direction1();
        let dir2 = landscape.direction2();
        for i in 0..dim {
            weights[i] = center_weights[i] + x * dir1[i] + y * dir2[i];
        }

        landscape.add_trajectory_point(&weights, loss);

        // Gradient descent step
        let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let dy = 200.0 * (y - x * x);
        x -= lr * dx;
        y -= lr * dy;

        if step % 20 == 0 {
            println!(
                "   Step {}: pos=({:.3}, {:.3}), loss={:.4}",
                step, x, y, loss
            );
        }
    }

    println!("   Total trajectory points: {}", landscape.trajectory.len());
    println!();

    // ========== 3. Generate Mesh for 3D Rendering ==========
    println!("3. Generating 3D mesh...");

    let (vertices, indices) = landscape.to_mesh();
    println!("   Vertices: {}", vertices.len());
    println!("   Triangles: {}", indices.len());

    // Sample some vertices
    println!("   Sample vertices:");
    for v in vertices.iter().take(3) {
        println!("     [{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]);
    }
    println!();

    // ========== 4. Gradient Field Analysis ==========
    println!("4. Analyzing gradient field...");

    let samples = landscape.sample_gradient_field(10);
    println!("   Gradient samples: {}", samples.len());

    // Find interesting points
    let mut max_gradient = 0.0_f32;
    let mut min_gradient = f32::INFINITY;
    let mut max_point = (0.0, 0.0);
    let mut min_point = (0.0, 0.0);

    for sample in &samples {
        if sample.magnitude > max_gradient {
            max_gradient = sample.magnitude;
            max_point = (sample.x, sample.y);
        }
        if sample.magnitude < min_gradient {
            min_gradient = sample.magnitude;
            min_point = (sample.x, sample.y);
        }
    }

    println!(
        "   Steepest gradient: {:.4} at ({:.2}, {:.2})",
        max_gradient, max_point.0, max_point.1
    );
    println!(
        "   Flattest point: {:.4} at ({:.2}, {:.2})",
        min_gradient, min_point.0, min_point.1
    );
    println!();

    // ========== 5. Trajectory Statistics ==========
    println!("5. Trajectory analysis...");

    let dir1 = landscape.direction1().to_vec();
    let dir2 = landscape.direction2().to_vec();
    let tracker_weights = center_weights.clone();

    let mut tracker = TrajectoryTracker::new(tracker_weights, dir1, dir2);

    // Re-simulate to populate tracker
    let mut x = 1.5;
    let mut y = 1.5;
    for step in 0..steps {
        let loss = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);

        let mut weights = center_weights.clone();
        let d1 = landscape.direction1();
        let d2 = landscape.direction2();
        for i in 0..dim {
            weights[i] = center_weights[i] + x * d1[i] + y * d2[i];
        }

        tracker.record(&weights, loss, step);

        let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let dy = 200.0 * (y - x * x);
        x -= lr * dx;
        y -= lr * dy;
    }

    let stats = tracker.stats();
    println!("   Path length: {:.4}", stats.path_length);
    println!("   Net displacement: {:.4}", stats.net_displacement);
    println!("   Efficiency: {:.4}", stats.efficiency);
    println!("   Loss improvement: {:.4}", stats.loss_improvement);
    println!("   Oscillation count: {}", stats.oscillation_count);
    println!();

    // ========== 6. Mesh Export Formats ==========
    println!("6. Exporting mesh data...");

    let exporter = MeshExporter::new(&landscape.surface, landscape.x_range, landscape.y_range);

    // OBJ format
    let obj = exporter.to_obj();
    println!("   OBJ format: {} bytes", obj.len());
    println!("   First 3 lines:");
    for line in obj.lines().take(3) {
        println!("     {}", line);
    }

    // PLY format
    let ply = exporter.to_ply();
    println!("   PLY format: {} bytes", ply.len());

    // Contour lines
    let contours = exporter.generate_contours(&[0.5, 1.0, 2.0, 5.0, 10.0]);
    let total_segments: usize = contours.iter().map(|(_, s)| s.len()).sum();
    println!(
        "   Contour lines: {} levels, {} total segments",
        contours.len(),
        total_segments
    );
    println!();

    // ========== 7. Critical Points ==========
    println!("7. Finding critical points...");

    let gradient_field =
        GradientField::from_surface(&landscape.surface, landscape.x_range, landscape.y_range);

    let minima = gradient_field.find_minima(0.5);
    let saddles = gradient_field.find_saddle_points(0.5);

    println!("   Local minima: {}", minima.len());
    for (x, y, loss) in minima.iter().take(3) {
        println!("     ({:.3}, {:.3}) -> loss = {:.4}", x, y, loss);
    }

    println!("   Saddle points: {}", saddles.len());
    for (x, y, loss) in saddles.iter().take(3) {
        println!("     ({:.3}, {:.3}) -> loss = {:.4}", x, y, loss);
    }
    println!();

    // ========== 8. Curvature Analysis ==========
    println!("8. Curvature analysis at center...");

    let (lambda1, lambda2) = gradient_field.curvature_at(0.0, 0.0);
    println!("   Eigenvalues: ({:.4}, {:.4})", lambda1, lambda2);

    if lambda1 > 0.0 && lambda2 > 0.0 {
        println!("   Classification: Local minimum (bowl shape)");
    } else if lambda1 < 0.0 && lambda2 < 0.0 {
        println!("   Classification: Local maximum (hill shape)");
    } else {
        println!("   Classification: Saddle point");
    }
    println!();

    println!("=== Demo Complete ===");
}
