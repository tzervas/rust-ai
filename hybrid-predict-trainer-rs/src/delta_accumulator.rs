//! Delta accumulation for batched weight updates.
//!
//! This module implements delta accumulation to reduce memory overhead from
//! Burn's functional `.map()` API. Instead of applying weight deltas immediately
//! (which creates a full model copy each time), deltas are accumulated and
//! applied in batches at phase transitions.
//!
//! # Memory Savings
//!
//! Without accumulation:
//! - 35 weight delta applications per 50 steps
//! - 35 model copies × 496 MB = 17 GB allocations
//! - CUDA can't free fast enough → leak
//!
//! With accumulation:
//! - ~5 batch applications per 50 steps (at phase transitions)
//! - 5 model copies × 496 MB = 2.5 GB allocations
//! - **7× memory reduction**

use crate::state::WeightDelta;
use crate::HybridResult;
use std::collections::HashMap;

/// Accumulates weight deltas for batched application.
///
/// Weight deltas are accumulated by summing the delta vectors for each parameter.
/// When flushed, all accumulated deltas are applied in a single operation,
/// minimizing the number of model copies created by Burn's `.map()` API.
#[derive(Debug, Clone)]
pub struct DeltaAccumulator {
    /// Accumulated deltas per parameter
    accumulated: HashMap<String, Vec<f32>>,

    /// Cumulative scale factor
    cumulative_scale: f32,

    /// Number of deltas accumulated
    count: usize,
}

impl Default for DeltaAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAccumulator {
    /// Creates a new empty delta accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accumulated: HashMap::new(),
            cumulative_scale: 1.0,
            count: 0,
        }
    }

    /// Adds a weight delta to the accumulator.
    ///
    /// Deltas are summed element-wise for each parameter. If a parameter
    /// doesn't exist yet, it's initialized with the delta values.
    ///
    /// # Arguments
    ///
    /// * `delta` - The weight delta to accumulate
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut accumulator = DeltaAccumulator::new();
    /// accumulator.add(&prediction.weight_delta);
    /// accumulator.add(&correction.weight_delta);
    /// // Later: flush to apply both at once
    /// ```
    pub fn add(&mut self, delta: &WeightDelta) {
        // Update cumulative scale (multiplicative)
        self.cumulative_scale *= delta.scale;

        for (param_name, delta_vec) in &delta.deltas {
            // Get or create entry for this parameter
            let accumulated_vec = self.accumulated
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; delta_vec.len()]);

            // Sum deltas element-wise (scaled by delta.scale)
            for (acc, &d) in accumulated_vec.iter_mut().zip(delta_vec.iter()) {
                *acc += d * delta.scale;
            }
        }

        self.count += 1;
    }

    /// Checks if there are accumulated deltas to flush.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.accumulated.is_empty()
    }

    /// Returns the number of deltas accumulated.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Flushes accumulated deltas and returns a single merged delta.
    ///
    /// After flushing, the accumulator is reset to empty.
    ///
    /// # Returns
    ///
    /// A `WeightDelta` containing all accumulated deltas merged together,
    /// or `None` if no deltas were accumulated.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(merged_delta) = accumulator.flush() {
    ///     model.apply_weight_delta(&merged_delta)?;
    /// }
    /// ```
    #[must_use]
    pub fn flush(&mut self) -> Option<WeightDelta> {
        if self.accumulated.is_empty() {
            return None;
        }

        // Create merged delta with accumulated values
        let merged = WeightDelta {
            deltas: self.accumulated.clone(),
            scale: 1.0, // Scaling already applied during accumulation
            metadata: crate::state::WeightDeltaMetadata {
                is_predicted: false,
                confidence: None,
                source_phase: None,
                num_steps: self.count,
            },
        };

        // Reset accumulator
        self.accumulated.clear();
        self.cumulative_scale = 1.0;
        self.count = 0;

        Some(merged)
    }

    /// Clears all accumulated deltas without returning them.
    pub fn clear(&mut self) {
        self.accumulated.clear();
        self.cumulative_scale = 1.0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_empty() {
        let accumulator = DeltaAccumulator::new();
        assert!(accumulator.is_empty());
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_accumulator_add_single() {
        let mut accumulator = DeltaAccumulator::new();

        let mut deltas = HashMap::new();
        deltas.insert("param1".to_string(), vec![1.0, 2.0, 3.0]);

        let delta = WeightDelta {
            deltas,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta);
        assert!(!accumulator.is_empty());
        assert_eq!(accumulator.count(), 1);
    }

    #[test]
    fn test_accumulator_add_multiple() {
        let mut accumulator = DeltaAccumulator::new();

        // First delta
        let mut deltas1 = HashMap::new();
        deltas1.insert("param1".to_string(), vec![1.0, 2.0, 3.0]);
        let delta1 = WeightDelta {
            deltas: deltas1,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        // Second delta (same parameter)
        let mut deltas2 = HashMap::new();
        deltas2.insert("param1".to_string(), vec![0.5, 1.0, 1.5]);
        let delta2 = WeightDelta {
            deltas: deltas2,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta1);
        accumulator.add(&delta2);

        assert_eq!(accumulator.count(), 2);

        // Check accumulated values
        let merged = accumulator.flush().unwrap();
        let param1_deltas = &merged.deltas["param1"];
        assert_eq!(param1_deltas[0], 1.5); // 1.0 + 0.5
        assert_eq!(param1_deltas[1], 3.0); // 2.0 + 1.0
        assert_eq!(param1_deltas[2], 4.5); // 3.0 + 1.5
    }

    #[test]
    fn test_accumulator_flush_resets() {
        let mut accumulator = DeltaAccumulator::new();

        let mut deltas = HashMap::new();
        deltas.insert("param1".to_string(), vec![1.0, 2.0]);
        let delta = WeightDelta {
            deltas,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta);
        assert_eq!(accumulator.count(), 1);

        // Flush should reset
        let _ = accumulator.flush();
        assert!(accumulator.is_empty());
        assert_eq!(accumulator.count(), 0);
    }

    #[test]
    fn test_accumulator_with_scaling() {
        let mut accumulator = DeltaAccumulator::new();

        let mut deltas = HashMap::new();
        deltas.insert("param1".to_string(), vec![1.0, 2.0]);

        // Delta with scale factor
        let delta = WeightDelta {
            deltas,
            scale: 0.5,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta);

        let merged = accumulator.flush().unwrap();
        let param1_deltas = &merged.deltas["param1"];

        // Should be scaled during accumulation
        assert_eq!(param1_deltas[0], 0.5); // 1.0 * 0.5
        assert_eq!(param1_deltas[1], 1.0); // 2.0 * 0.5
        assert_eq!(merged.scale, 1.0); // Scale already applied
    }

    #[test]
    fn test_accumulator_multiple_parameters() {
        let mut accumulator = DeltaAccumulator::new();

        // Delta with two parameters
        let mut deltas = HashMap::new();
        deltas.insert("param1".to_string(), vec![1.0, 2.0]);
        deltas.insert("param2".to_string(), vec![3.0, 4.0]);

        let delta = WeightDelta {
            deltas,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta);

        let merged = accumulator.flush().unwrap();
        assert_eq!(merged.deltas.len(), 2);
        assert!(merged.deltas.contains_key("param1"));
        assert!(merged.deltas.contains_key("param2"));
    }

    #[test]
    fn test_accumulator_clear() {
        let mut accumulator = DeltaAccumulator::new();

        let mut deltas = HashMap::new();
        deltas.insert("param1".to_string(), vec![1.0, 2.0]);
        let delta = WeightDelta {
            deltas,
            scale: 1.0,
            metadata: crate::state::WeightDeltaMetadata::default(),
        };

        accumulator.add(&delta);
        assert!(!accumulator.is_empty());

        accumulator.clear();
        assert!(accumulator.is_empty());
        assert_eq!(accumulator.count(), 0);
    }
}
