//! Optimizer implementations (AdamW, SGD).

use candle_core::Tensor;
use candle_nn::{Optimizer, ParamsAdamW, VarMap};

use crate::error::{AxolotlError, Result};

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Beta1 for Adam
    pub beta1: f64,
    /// Beta2 for Adam
    pub beta2: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 5e-5,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
        }
    }
}

impl OptimizerConfig {
    /// Create AdamW optimizer with these parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the optimizer cannot be created.
    pub fn build_adamw(&self, varmap: &VarMap) -> Result<AdamWOptimizer> {
        let vars = varmap.all_vars();
        let params = ParamsAdamW {
            lr: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        };

        let opt = candle_nn::AdamW::new(vars, params)
            .map_err(|e| AxolotlError::Training(format!("Failed to create AdamW: {}", e)))?;

        Ok(AdamWOptimizer { inner: opt })
    }
}

/// AdamW optimizer wrapper.
pub struct AdamWOptimizer {
    inner: candle_nn::AdamW,
}

impl AdamWOptimizer {
    /// Perform a single optimization step.
    ///
    /// # Errors
    ///
    /// Returns an error if the step fails.
    pub fn step(&mut self, loss: &Tensor) -> Result<()> {
        self.inner
            .backward_step(loss)
            .map_err(|e| AxolotlError::Training(format!("Optimizer step failed: {}", e)))
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    /// Set learning rate (used by schedulers).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();
        assert_eq!(config.learning_rate, 5e-5);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_build_adamw() -> Result<()> {
        let config = OptimizerConfig::default();
        let varmap = VarMap::new();

        let optimizer = config.build_adamw(&varmap)?;
        assert_eq!(optimizer.learning_rate(), 5e-5);

        Ok(())
    }
}
