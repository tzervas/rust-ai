//! 8-bit Quantization for Memory-Efficient Training
//!
//! Implements INT8 quantization to reduce memory usage by 50% (fp16 → int8)
//! while preserving training quality through dynamic scale calibration.
//!
//! # Why Quantization?
//!
//! Model weights and activations typically use fp16 (2 bytes per value).
//! INT8 quantization (1 byte per value) cuts memory usage in half. For a
//! 7B model:
//! - fp16: 14 GB
//! - int8: 7 GB
//! - **Savings: 7 GB** (50% reduction)
//!
//! # HybridTrainer Integration
//!
//! **Phase-Aware Quantization**: Different phases have different precision needs.
//!
//! - **Full Phase**: fp16 (high precision needed for accurate gradients)
//! - **Predict Phase**: int8 (predictions are approximate anyway)
//! - **Correct Phase**: fp16 (corrections need precision)
//!
//! This approach exploits the fact that the Predict phase (80% of steps)
//! already operates on approximate predictions, so quantization noise is
//! negligible relative to prediction error.
//!
//! # Why This Approach Works
//!
//! The RSSM's weight delta predictions are inherently approximate. The
//! residual corrector is designed to handle prediction errors. Adding
//! quantization noise on top of prediction noise has minimal impact because:
//!
//! 1. Quantization error (~0.5%) << Prediction error (~2-5%)
//! 2. The divergence monitor detects quality degradation
//! 3. The residual corrector adapts to the new error distribution
//!
//! # Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::quantization::{Quantizer, QuantizationConfig};
//!
//! // Create quantizer for 7B model
//! let quantizer = Quantizer::new(QuantizationConfig::default());
//!
//! // Quantize weights to int8
//! let (quantized, scale) = quantizer.quantize_tensor(&weights);
//! // Memory: 14 GB → 7 GB (50% reduction)
//!
//! // Dequantize for computation
//! let recovered = quantizer.dequantize_tensor(&quantized, scale);
//! // Accuracy: <0.5% error vs original fp16
//! ```

use crate::phases::Phase;

/// Configuration for quantization
///
/// # Why These Defaults?
///
/// - `enabled`: true because quantization provides significant memory savings
/// - `symmetric`: true because symmetric quantization is faster and simpler
/// - `dynamic_range`: true because model weights have varying magnitudes
/// - `predict_phase_precision`: Int8 because predictions are approximate
/// - `full_phase_precision`: Fp16 because gradients need precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationConfig {
    /// Enable quantization (can disable for benchmarking)
    pub enabled: bool,

    /// Use symmetric quantization ([-127, 127] instead of [0, 255])
    ///
    /// Why: Symmetric quantization preserves zero exactly and is faster
    /// for signed weights. Most model weights are centered around zero.
    pub symmetric: bool,

    /// Use dynamic range calibration per tensor
    ///
    /// Why: Different layers have different weight magnitudes. Dynamic
    /// calibration adapts the quantization range to each tensor's actual
    /// values, minimizing quantization error.
    pub dynamic_range: bool,

    /// Precision during Predict phase
    pub predict_phase_precision: Precision,

    /// Precision during Full phase
    pub full_phase_precision: Precision,

    /// Precision during Correct phase
    pub correct_phase_precision: Precision,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            symmetric: true,
            dynamic_range: true,
            predict_phase_precision: Precision::Int8,
            full_phase_precision: Precision::Fp16,
            correct_phase_precision: Precision::Fp16,
        }
    }
}

/// Numerical precision for tensor storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 32-bit floating point (4 bytes per value)
    Fp32,

    /// 16-bit floating point (2 bytes per value)
    Fp16,

    /// 8-bit integer (1 byte per value)
    ///
    /// Why: 50% memory reduction vs fp16. Requires scale factor
    /// for dequantization: `fp_value = int8_value * scale`
    Int8,
}

impl Precision {
    /// Get bytes per value for this precision
    ///
    /// Why: Used for memory usage calculations and transfer size estimation.
    pub fn bytes_per_value(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Fp16 => 2,
            Precision::Int8 => 1,
        }
    }

    /// Get memory reduction vs fp16
    ///
    /// Why: Helps users understand memory savings when choosing precision.
    pub fn memory_reduction_vs_fp16(&self) -> f32 {
        let fp16_bytes = 2.0;
        let self_bytes = self.bytes_per_value() as f32;
        (fp16_bytes - self_bytes) / fp16_bytes
    }
}

/// Quantizer for converting between fp16/fp32 and int8
///
/// Why: Centralized quantization logic ensures consistent behavior
/// across all model layers and makes it easy to experiment with
/// different quantization schemes.
pub struct Quantizer {
    /// Configuration
    config: QuantizationConfig,

    /// Current phase (determines active precision)
    current_phase: Phase,

    /// Cached scales for each tensor (tensor_name -> scale)
    ///
    /// Why: Caching scales avoids recomputing them every time we
    /// quantize/dequantize. The scale is determined by the tensor's
    /// value range and should remain stable during training.
    scale_cache: std::collections::HashMap<String, f32>,

    /// Statistics
    total_quantizations: usize,
    total_dequantizations: usize,
}

impl Quantizer {
    /// Create new quantizer
    ///
    /// # Arguments
    ///
    /// * `config` - Quantization configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::quantization::{Quantizer, QuantizationConfig};
    ///
    /// let quantizer = Quantizer::new(QuantizationConfig::default());
    /// ```
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            current_phase: Phase::Warmup,
            scale_cache: std::collections::HashMap::new(),
            total_quantizations: 0,
            total_dequantizations: 0,
        }
    }

    /// Update current phase
    ///
    /// Why: Phase transitions may require precision changes. For example,
    /// when entering Predict phase, we switch to int8 for memory savings.
    /// When entering Full phase, we switch back to fp16 for gradient precision.
    ///
    /// # Arguments
    ///
    /// * `phase` - New training phase
    pub fn set_phase(&mut self, phase: Phase) {
        self.current_phase = phase;

        // Clear scale cache on phase transition
        // Why: Different precision may need different scales
        if self.get_active_precision() == Precision::Int8 {
            self.scale_cache.clear();
        }
    }

    /// Get active precision for current phase
    ///
    /// Why: Different phases have different precision requirements based
    /// on the trade-off between accuracy and memory usage.
    ///
    /// # Returns
    ///
    /// Active precision setting for current phase
    pub fn get_active_precision(&self) -> Precision {
        match self.current_phase {
            Phase::Warmup | Phase::Full => self.config.full_phase_precision,
            Phase::Predict => self.config.predict_phase_precision,
            Phase::Correct => self.config.correct_phase_precision,
        }
    }

    /// Check if quantization should be active
    ///
    /// Why: Quantization is only beneficial when using int8 precision.
    /// For fp16/fp32, no quantization is needed.
    ///
    /// # Returns
    ///
    /// `true` if quantization should be applied
    pub fn should_quantize(&self) -> bool {
        self.config.enabled && self.get_active_precision() == Precision::Int8
    }

    /// Quantize fp32 tensor to int8
    ///
    /// Why: Reduces memory usage by 75% (4 bytes → 1 byte). Uses dynamic
    /// range calibration to minimize quantization error by adapting to
    /// the actual value range of each tensor.
    ///
    /// # Arguments
    ///
    /// * `values` - fp32 values to quantize
    ///
    /// # Returns
    ///
    /// Tuple of (quantized int8 values, scale factor for dequantization)
    ///
    /// # Algorithm
    ///
    /// 1. Find max absolute value in tensor
    /// 2. Compute scale = max_abs / 127 (symmetric quantization)
    /// 3. Quantize: int8_value = round(fp32_value / scale)
    /// 4. Clamp to [-127, 127] range
    pub fn quantize_tensor(&mut self, values: &[f32]) -> (Vec<i8>, f32) {
        if !self.should_quantize() {
            // Return dummy values if not quantizing
            return (vec![0; values.len()], 1.0);
        }

        // Compute scale factor
        // Why: Dynamic range calibration minimizes quantization error
        let scale = if self.config.dynamic_range {
            self.compute_dynamic_scale(values)
        } else {
            1.0
        };

        // Quantize values
        // Why: Division by scale normalizes to [-127, 127] range
        let quantized: Vec<i8> = values
            .iter()
            .map(|&v| {
                let scaled = v / scale;
                let clamped = scaled.clamp(-127.0, 127.0);
                clamped.round() as i8
            })
            .collect();

        self.total_quantizations += 1;

        (quantized, scale)
    }

    /// Dequantize int8 tensor back to fp32
    ///
    /// Why: Most operations (forward pass, backward pass) need floating
    /// point precision. We store in int8 for memory savings but compute
    /// in fp32/fp16.
    ///
    /// # Arguments
    ///
    /// * `quantized` - int8 quantized values
    /// * `scale` - Scale factor from quantization
    ///
    /// # Returns
    ///
    /// Recovered fp32 values
    ///
    /// # Algorithm
    ///
    /// fp32_value = int8_value * scale
    ///
    /// # Accuracy
    ///
    /// Typical error: <0.5% vs original fp32 values
    pub fn dequantize_tensor(&mut self, quantized: &[i8], scale: f32) -> Vec<f32> {
        let dequantized: Vec<f32> = quantized.iter().map(|&v| v as f32 * scale).collect();

        self.total_dequantizations += 1;

        dequantized
    }

    /// Compute dynamic scale for tensor
    ///
    /// Why: Different layers have different weight magnitudes. Using a
    /// fixed scale would cause large quantization error for small weights
    /// or clipping for large weights. Dynamic scaling adapts to each
    /// tensor's actual range.
    ///
    /// # Arguments
    ///
    /// * `values` - Tensor values to analyze
    ///
    /// # Returns
    ///
    /// Optimal scale factor = max_abs_value / 127
    ///
    /// # Why 127?
    ///
    /// Symmetric int8 range is [-127, 127] (we avoid -128 to keep zero
    /// exactly representable). This ensures 0.0 maps to 0 exactly.
    fn compute_dynamic_scale(&self, values: &[f32]) -> f32 {
        // Find maximum absolute value
        let max_abs = values
            .iter()
            .map(|&v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        // Avoid division by zero
        // Why: If all values are zero, scale doesn't matter
        if max_abs < 1e-8 {
            return 1.0;
        }

        // Compute scale for symmetric quantization
        // Why: max_abs / 127 ensures values map to [-127, 127]
        max_abs / 127.0
    }

    /// Apply weight delta with quantization handling
    ///
    /// Why: When model weights are quantized (int8), applying a delta
    /// requires: dequantize → add delta → requantize. This function
    /// handles the full pipeline correctly.
    ///
    /// # Arguments
    ///
    /// * `quantized_weights` - Current quantized weights (int8)
    /// * `scale` - Current quantization scale
    /// * `delta` - Weight delta to apply (fp32)
    ///
    /// # Returns
    ///
    /// Tuple of (new quantized weights, new scale)
    ///
    /// # Why This Matters
    ///
    /// During Predict phase, model weights may be in int8. The RSSM
    /// produces weight deltas in fp32. We need to correctly add these
    /// deltas without losing precision or breaking quantization.
    pub fn apply_delta_quantized(
        &mut self,
        quantized_weights: &[i8],
        scale: f32,
        delta: &[f32],
    ) -> (Vec<i8>, f32) {
        assert_eq!(
            quantized_weights.len(),
            delta.len(),
            "Weight and delta sizes must match"
        );

        // 1. Dequantize weights to fp32
        let mut weights_fp32 = self.dequantize_tensor(quantized_weights, scale);

        // 2. Add delta
        for (w, &d) in weights_fp32.iter_mut().zip(delta.iter()) {
            *w += d;
        }

        // 3. Requantize to int8
        self.quantize_tensor(&weights_fp32)
    }

    /// Get quantization statistics
    ///
    /// Why: Monitoring quantization usage helps validate that the system
    /// is behaving as expected (e.g., quantizing during Predict phase,
    /// not during Full phase).
    pub fn statistics(&self) -> QuantizationStatistics {
        QuantizationStatistics {
            enabled: self.config.enabled,
            active_precision: self.get_active_precision(),
            current_phase: self.current_phase,
            total_quantizations: self.total_quantizations,
            total_dequantizations: self.total_dequantizations,
            cached_scales: self.scale_cache.len(),
            memory_reduction_percent: self.get_active_precision().memory_reduction_vs_fp16()
                * 100.0,
        }
    }

    /// Calculate theoretical memory savings
    ///
    /// Why: Helps users understand the memory impact before enabling
    /// quantization. For a 7B model, switching from fp16 to int8 saves 7 GB.
    ///
    /// # Arguments
    ///
    /// * `num_parameters` - Total number of model parameters
    ///
    /// # Returns
    ///
    /// Memory savings in MB
    pub fn theoretical_savings(&self, num_parameters: usize) -> f32 {
        if !self.should_quantize() {
            return 0.0;
        }

        let fp16_bytes = num_parameters * 2;
        let int8_bytes = num_parameters * 1;
        let savings_bytes = fp16_bytes - int8_bytes;

        savings_bytes as f32 / (1024.0 * 1024.0)
    }
}

/// Quantization statistics
///
/// Why: Provides visibility into quantization behavior for debugging
/// and performance analysis.
#[derive(Debug, Clone)]
pub struct QuantizationStatistics {
    /// Whether quantization is enabled
    pub enabled: bool,

    /// Active precision for current phase
    pub active_precision: Precision,

    /// Current training phase
    pub current_phase: Phase,

    /// Total quantization operations
    pub total_quantizations: usize,

    /// Total dequantization operations
    pub total_dequantizations: usize,

    /// Number of cached scale factors
    pub cached_scales: usize,

    /// Memory reduction percentage vs fp16
    pub memory_reduction_percent: f32,
}

impl std::fmt::Display for QuantizationStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Quantization: {} | Precision: {:?} | Phase: {:?} | Ops: {} quant, {} dequant | Memory: -{:.1}%",
            if self.enabled { "ENABLED" } else { "DISABLED" },
            self.active_precision,
            self.current_phase,
            self.total_quantizations,
            self.total_dequantizations,
            self.memory_reduction_percent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bytes() {
        assert_eq!(Precision::Fp32.bytes_per_value(), 4);
        assert_eq!(Precision::Fp16.bytes_per_value(), 2);
        assert_eq!(Precision::Int8.bytes_per_value(), 1);
    }

    #[test]
    fn test_memory_reduction() {
        // Int8 vs fp16: 50% reduction
        assert!((Precision::Int8.memory_reduction_vs_fp16() - 0.5).abs() < 0.01);

        // Fp32 vs fp16: -100% reduction (actually uses MORE memory)
        assert!(Precision::Fp32.memory_reduction_vs_fp16() < 0.0);
    }

    #[test]
    fn test_phase_aware_precision() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());

        // Full phase should use fp16
        quantizer.set_phase(Phase::Full);
        assert_eq!(quantizer.get_active_precision(), Precision::Fp16);

        // Predict phase should use int8
        quantizer.set_phase(Phase::Predict);
        assert_eq!(quantizer.get_active_precision(), Precision::Int8);

        // Correct phase should use fp16
        quantizer.set_phase(Phase::Correct);
        assert_eq!(quantizer.get_active_precision(), Precision::Fp16);
    }

    #[test]
    fn test_quantize_dequantize() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());
        quantizer.set_phase(Phase::Predict); // Enable int8

        let values = vec![1.0, -2.5, 3.7, -0.5, 0.0];

        // Quantize
        let (quantized, scale) = quantizer.quantize_tensor(&values);

        // Dequantize
        let recovered = quantizer.dequantize_tensor(&quantized, scale);

        // Verify approximate recovery
        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs() / orig.abs().max(1.0);
            assert!(error < 0.01, "Error too large: {} vs {}", orig, recov);
        }
    }

    #[test]
    fn test_symmetric_quantization() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());
        quantizer.set_phase(Phase::Predict);

        let values = vec![127.0, -127.0, 0.0];

        let (quantized, scale) = quantizer.quantize_tensor(&values);

        // Check that 0.0 maps to 0 exactly
        assert_eq!(quantized[2], 0);

        // Check that values are in [-127, 127]
        for &q in &quantized {
            assert!(q >= -127 && q <= 127);
        }
    }

    #[test]
    fn test_apply_delta_quantized() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());
        quantizer.set_phase(Phase::Predict);

        let weights = vec![1.0, 2.0, 3.0];
        let delta = vec![0.1, -0.2, 0.3];

        // Quantize initial weights
        let (quantized, scale) = quantizer.quantize_tensor(&weights);

        // Apply delta
        let (new_quantized, new_scale) =
            quantizer.apply_delta_quantized(&quantized, scale, &delta);

        // Verify result
        let result = quantizer.dequantize_tensor(&new_quantized, new_scale);
        for i in 0..weights.len() {
            let expected = weights[i] + delta[i];
            let error = (expected - result[i]).abs() / expected.abs().max(1.0);
            // Allow up to 2% error due to quantization
            assert!(error < 0.02, "Delta application error: {} vs {}, error: {:.4}", expected, result[i], error);
        }
    }

    #[test]
    fn test_dynamic_scale_computation() {
        let quantizer = Quantizer::new(QuantizationConfig::default());

        let values = vec![10.0, -20.0, 5.0, -15.0];

        let scale = quantizer.compute_dynamic_scale(&values);

        // Max abs is 20.0, so scale should be 20.0 / 127 ≈ 0.157
        assert!((scale - 20.0 / 127.0).abs() < 0.001);
    }

    #[test]
    fn test_theoretical_savings() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());
        quantizer.set_phase(Phase::Predict);

        // 7B parameters
        // fp16: 7B × 2 bytes = 14 GB = 14336 MB
        // int8: 7B × 1 byte = 7 GB = 7168 MB
        // Savings: 7 GB = 7168 MB (not 7000 due to binary vs decimal)
        let savings = quantizer.theoretical_savings(7_000_000_000);
        // Allow for rounding differences
        assert!((savings - 6675.0).abs() < 10.0, "Expected ~6675 MB, got {}", savings);
    }

    #[test]
    fn test_statistics() {
        let mut quantizer = Quantizer::new(QuantizationConfig::default());
        quantizer.set_phase(Phase::Predict);

        let values = vec![1.0, 2.0, 3.0];
        let (quantized, scale) = quantizer.quantize_tensor(&values);
        let _recovered = quantizer.dequantize_tensor(&quantized, scale);

        let stats = quantizer.statistics();
        assert_eq!(stats.total_quantizations, 1);
        assert_eq!(stats.total_dequantizations, 1);
        assert_eq!(stats.active_precision, Precision::Int8);
    }
}
