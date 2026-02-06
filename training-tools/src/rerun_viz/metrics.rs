//! Training metrics logging for Rerun visualization.
//!
//! This module handles streaming of core training metrics:
//! - Loss curves
//! - Learning rate schedules
//! - Gradient norms
//! - Training phase indicators

use rerun::RecordingStream;

/// Training phase for visualization purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhaseLog {
    /// Initial warmup phase (LR ramping up)
    Warmup,
    /// Main training phase
    Training,
    /// Validation/evaluation phase
    Validation,
    /// Cooldown phase (LR decaying)
    Cooldown,
    /// Prediction phase (hybrid training)
    Prediction,
    /// Correction phase (hybrid training)
    Correction,
}

impl TrainingPhaseLog {
    /// Get a numeric value for phase visualization.
    pub fn as_ordinal(&self) -> f32 {
        match self {
            TrainingPhaseLog::Warmup => 0.0,
            TrainingPhaseLog::Training => 1.0,
            TrainingPhaseLog::Validation => 2.0,
            TrainingPhaseLog::Cooldown => 3.0,
            TrainingPhaseLog::Prediction => 4.0,
            TrainingPhaseLog::Correction => 5.0,
        }
    }

    /// Get a color for this phase (RGB).
    pub fn color(&self) -> [u8; 3] {
        match self {
            TrainingPhaseLog::Warmup => [255, 200, 100], // Light orange
            TrainingPhaseLog::Training => [100, 200, 100], // Green
            TrainingPhaseLog::Validation => [100, 150, 255], // Light blue
            TrainingPhaseLog::Cooldown => [200, 100, 200], // Purple
            TrainingPhaseLog::Prediction => [255, 255, 100], // Yellow
            TrainingPhaseLog::Correction => [255, 100, 100], // Red
        }
    }

    /// Get the phase name as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingPhaseLog::Warmup => "warmup",
            TrainingPhaseLog::Training => "training",
            TrainingPhaseLog::Validation => "validation",
            TrainingPhaseLog::Cooldown => "cooldown",
            TrainingPhaseLog::Prediction => "prediction",
            TrainingPhaseLog::Correction => "correction",
        }
    }
}

/// Logger for training metrics time series.
pub struct MetricsLogger<'a> {
    rec: &'a RecordingStream,
}

impl<'a> MetricsLogger<'a> {
    /// Create a new metrics logger.
    pub fn new(rec: &'a RecordingStream) -> Self {
        Self { rec }
    }

    /// Log a training step with core metrics.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Loss value at this step
    /// * `lr` - Current learning rate
    /// * `grad_norm` - Gradient L2 norm
    pub fn log_step(&self, step: u64, loss: f32, lr: f32, grad_norm: f32) {
        // Set timeline to current step
        self.rec.set_time_sequence("step", step as i64);

        // Log loss as a scalar
        let _ = self
            .rec
            .log("training/loss", &rerun::Scalar::new(loss as f64));

        // Log learning rate (often very small, so log-scale might be useful)
        let _ = self
            .rec
            .log("training/learning_rate", &rerun::Scalar::new(lr as f64));

        // Log gradient norm
        let _ = self.rec.log(
            "training/gradient_norm",
            &rerun::Scalar::new(grad_norm as f64),
        );

        // Log log-scale learning rate for better visualization
        let log_lr = if lr > 0.0 { lr.log10() } else { -10.0 };
        let _ = self.rec.log(
            "training/log_learning_rate",
            &rerun::Scalar::new(log_lr as f64),
        );
    }

    /// Log extended metrics including phase and additional stats.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Loss value
    /// * `lr` - Learning rate
    /// * `grad_norm` - Gradient norm
    /// * `phase` - Current training phase
    /// * `batch_time_ms` - Time for this batch in milliseconds
    pub fn log_step_extended(
        &self,
        step: u64,
        loss: f32,
        lr: f32,
        grad_norm: f32,
        phase: TrainingPhaseLog,
        batch_time_ms: f32,
    ) {
        // Log basic metrics
        self.log_step(step, loss, lr, grad_norm);

        // Log phase as ordinal
        let _ = self.rec.log(
            "training/phase",
            &rerun::Scalar::new(phase.as_ordinal() as f64),
        );

        // Log phase as text annotation
        let _ = self
            .rec
            .log("training/phase_name", &rerun::TextLog::new(phase.as_str()));

        // Log batch timing
        let _ = self.rec.log(
            "training/batch_time_ms",
            &rerun::Scalar::new(batch_time_ms as f64),
        );

        // Log throughput (steps per second)
        let throughput = if batch_time_ms > 0.0 {
            1000.0 / batch_time_ms
        } else {
            0.0
        };
        let _ = self.rec.log(
            "training/steps_per_second",
            &rerun::Scalar::new(throughput as f64),
        );
    }

    /// Log validation metrics.
    ///
    /// # Arguments
    ///
    /// * `step` - Training step when validation was performed
    /// * `val_loss` - Validation loss
    /// * `val_accuracy` - Optional validation accuracy (0.0-1.0)
    /// * `val_perplexity` - Optional perplexity value
    pub fn log_validation(
        &self,
        step: u64,
        val_loss: f32,
        val_accuracy: Option<f32>,
        val_perplexity: Option<f32>,
    ) {
        self.rec.set_time_sequence("step", step as i64);

        let _ = self
            .rec
            .log("validation/loss", &rerun::Scalar::new(val_loss as f64));

        if let Some(acc) = val_accuracy {
            let _ = self
                .rec
                .log("validation/accuracy", &rerun::Scalar::new(acc as f64));
        }

        if let Some(ppl) = val_perplexity {
            let _ = self
                .rec
                .log("validation/perplexity", &rerun::Scalar::new(ppl as f64));

            // Log log-perplexity for better visualization
            let log_ppl = if ppl > 0.0 { ppl.ln() } else { 0.0 };
            let _ = self.rec.log(
                "validation/log_perplexity",
                &rerun::Scalar::new(log_ppl as f64),
            );
        }
    }

    /// Log gradient statistics per layer.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `layer_name` - Name of the layer
    /// * `mean` - Mean gradient value
    /// * `std` - Standard deviation
    /// * `max_abs` - Maximum absolute gradient value
    pub fn log_layer_gradients(
        &self,
        step: u64,
        layer_name: &str,
        mean: f32,
        std: f32,
        max_abs: f32,
    ) {
        self.rec.set_time_sequence("step", step as i64);

        let path = format!("gradients/{}", layer_name);

        let _ = self
            .rec
            .log(format!("{}/mean", path), &rerun::Scalar::new(mean as f64));
        let _ = self
            .rec
            .log(format!("{}/std", path), &rerun::Scalar::new(std as f64));
        let _ = self.rec.log(
            format!("{}/max_abs", path),
            &rerun::Scalar::new(max_abs as f64),
        );
    }

    /// Log memory usage statistics.
    ///
    /// # Arguments
    ///
    /// * `step` - Current step
    /// * `gpu_memory_used_mb` - GPU memory used in MB
    /// * `gpu_memory_total_mb` - Total GPU memory in MB
    /// * `cpu_memory_used_mb` - Optional CPU memory used
    pub fn log_memory(
        &self,
        step: u64,
        gpu_memory_used_mb: f32,
        gpu_memory_total_mb: f32,
        cpu_memory_used_mb: Option<f32>,
    ) {
        self.rec.set_time_sequence("step", step as i64);

        let _ = self.rec.log(
            "memory/gpu_used_mb",
            &rerun::Scalar::new(gpu_memory_used_mb as f64),
        );
        let _ = self.rec.log(
            "memory/gpu_total_mb",
            &rerun::Scalar::new(gpu_memory_total_mb as f64),
        );

        let utilization = if gpu_memory_total_mb > 0.0 {
            gpu_memory_used_mb / gpu_memory_total_mb * 100.0
        } else {
            0.0
        };
        let _ = self.rec.log(
            "memory/gpu_utilization_pct",
            &rerun::Scalar::new(utilization as f64),
        );

        if let Some(cpu_mem) = cpu_memory_used_mb {
            let _ = self
                .rec
                .log("memory/cpu_used_mb", &rerun::Scalar::new(cpu_mem as f64));
        }
    }

    /// Log a milestone event (e.g., checkpoint saved, LR decay triggered).
    ///
    /// # Arguments
    ///
    /// * `step` - Step when event occurred
    /// * `event_type` - Type of event (e.g., "checkpoint", "lr_decay")
    /// * `description` - Human-readable description
    pub fn log_event(&self, step: u64, event_type: &str, description: &str) {
        self.rec.set_time_sequence("step", step as i64);

        let log_msg = format!("[{}] {}", event_type, description);
        let _ = self
            .rec
            .log("events/timeline", &rerun::TextLog::new(log_msg));
    }

    /// Log loss decomposition (if using multiple loss terms).
    ///
    /// # Arguments
    ///
    /// * `step` - Current step
    /// * `loss_components` - Named loss components (e.g., [("ce_loss", 0.5), ("kl_loss", 0.1)])
    pub fn log_loss_components(&self, step: u64, loss_components: &[(&str, f32)]) {
        self.rec.set_time_sequence("step", step as i64);

        let mut total = 0.0f32;
        for (name, value) in loss_components {
            let _ = self.rec.log(
                format!("training/loss_components/{}", name),
                &rerun::Scalar::new(*value as f64),
            );
            total += value;
        }

        // Also log the total
        let _ = self.rec.log(
            "training/loss_components/total",
            &rerun::Scalar::new(total as f64),
        );
    }

    /// Log exponential moving average of loss for smoothed visualization.
    ///
    /// # Arguments
    ///
    /// * `step` - Current step
    /// * `ema_loss` - Exponentially smoothed loss value
    /// * `raw_loss` - Raw unsmoothed loss
    pub fn log_smoothed_loss(&self, step: u64, ema_loss: f32, raw_loss: f32) {
        self.rec.set_time_sequence("step", step as i64);

        let _ = self
            .rec
            .log("training/loss_ema", &rerun::Scalar::new(ema_loss as f64));
        let _ = self
            .rec
            .log("training/loss_raw", &rerun::Scalar::new(raw_loss as f64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_ordinal_ordering() {
        // Verify phases have distinct ordinals
        let phases = [
            TrainingPhaseLog::Warmup,
            TrainingPhaseLog::Training,
            TrainingPhaseLog::Validation,
            TrainingPhaseLog::Cooldown,
            TrainingPhaseLog::Prediction,
            TrainingPhaseLog::Correction,
        ];

        let ordinals: Vec<f32> = phases.iter().map(|p| p.as_ordinal()).collect();

        // Check all ordinals are unique
        for (i, a) in ordinals.iter().enumerate() {
            for (j, b) in ordinals.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Phases {} and {} have same ordinal", i, j);
                }
            }
        }
    }

    #[test]
    fn test_phase_colors_are_valid_rgb() {
        let phases = [
            TrainingPhaseLog::Warmup,
            TrainingPhaseLog::Training,
            TrainingPhaseLog::Validation,
        ];

        for phase in phases {
            let [r, g, b] = phase.color();
            // RGB values should be 0-255 (guaranteed by u8, but sanity check)
            assert!(r <= 255 && g <= 255 && b <= 255);
        }
    }
}
