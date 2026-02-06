//! Learning rate schedulers.

use crate::optimizer::AdamWOptimizer;

/// Learning rate scheduler types.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,
    /// Linear warmup then linear decay
    Linear {
        /// Number of warmup steps
        warmup_steps: usize,
        /// Total number of training steps
        total_steps: usize,
    },
    /// Cosine annealing with warmup
    Cosine {
        /// Number of warmup steps
        warmup_steps: usize,
        /// Total number of training steps
        total_steps: usize,
    },
}

/// Learning rate scheduler.
pub struct LRScheduler {
    /// Scheduler type
    scheduler_type: SchedulerType,
    /// Base learning rate
    base_lr: f64,
    /// Current step
    current_step: usize,
}

impl LRScheduler {
    /// Create a new scheduler.
    pub fn new(scheduler_type: SchedulerType, base_lr: f64) -> Self {
        Self {
            scheduler_type,
            base_lr,
            current_step: 0,
        }
    }

    /// Get learning rate for current step.
    pub fn get_lr(&self) -> f64 {
        match &self.scheduler_type {
            SchedulerType::Constant => self.base_lr,

            SchedulerType::Linear {
                warmup_steps,
                total_steps,
            } => self.linear_schedule(*warmup_steps, *total_steps),

            SchedulerType::Cosine {
                warmup_steps,
                total_steps,
            } => self.cosine_schedule(*warmup_steps, *total_steps),
        }
    }

    /// Step the scheduler and update optimizer.
    pub fn step(&mut self, optimizer: &mut AdamWOptimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    /// Linear warmup then linear decay.
    fn linear_schedule(&self, warmup_steps: usize, total_steps: usize) -> f64 {
        if self.current_step < warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / warmup_steps as f64)
        } else {
            // Linear decay
            let progress =
                (self.current_step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
            self.base_lr * (1.0 - progress).max(0.0)
        }
    }

    /// Cosine annealing with linear warmup.
    fn cosine_schedule(&self, warmup_steps: usize, total_steps: usize) -> f64 {
        if self.current_step < warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / warmup_steps as f64)
        } else {
            // Cosine decay
            let progress =
                (self.current_step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.base_lr * cosine_decay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LRScheduler::new(SchedulerType::Constant, 1e-3);
        assert_eq!(scheduler.get_lr(), 1e-3);
    }

    #[test]
    fn test_linear_warmup() {
        let mut scheduler = LRScheduler::new(
            SchedulerType::Linear {
                warmup_steps: 100,
                total_steps: 1000,
            },
            1e-3,
        );

        // At step 0, should be 0
        assert_eq!(scheduler.get_lr(), 0.0);

        // At step 50, should be half of base_lr
        scheduler.current_step = 50;
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-10);

        // At step 100, should be base_lr
        scheduler.current_step = 100;
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-10);

        // At step 550 (halfway through decay), should be half of base_lr
        scheduler.current_step = 550;
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let mut scheduler = LRScheduler::new(
            SchedulerType::Cosine {
                warmup_steps: 100,
                total_steps: 1000,
            },
            1e-3,
        );

        // At step 50 (during warmup), should be half of base_lr
        scheduler.current_step = 50;
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-10);

        // At step 100, should be base_lr
        scheduler.current_step = 100;
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-10);

        // At end of training, should approach 0
        scheduler.current_step = 1000;
        assert!(scheduler.get_lr() < 1e-5);
    }
}
