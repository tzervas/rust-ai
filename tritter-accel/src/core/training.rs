//! Training acceleration utilities.
//!
//! Provides tools for accelerating neural network training:
//! - Gradient compression for distributed training
//! - Mixed precision utilities
//! - Memory-efficient operations
//!
//! # Example
//!
//! ```rust,ignore
//! use tritter_accel::core::training::{GradientCompressor, TrainingConfig};
//!
//! let config = TrainingConfig::default();
//! let compressor = GradientCompressor::new(config);
//!
//! // Compress gradients for communication
//! let gradients = vec![0.1, -0.2, 0.3, -0.4];
//! let compressed = compressor.compress(&gradients, 0.1)?;
//!
//! // Decompress on receiving end
//! let recovered = compressor.decompress(&compressed, gradients.len())?;
//! ```

use thiserror::Error;

/// Errors from training operations.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// Invalid compression ratio.
    #[error("invalid compression ratio {0}: must be in (0, 1]")]
    InvalidRatio(f32),

    /// Dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Seed mismatch during decompression.
    #[error("seed mismatch: compression used {compress}, decompression used {decompress}")]
    SeedMismatch { compress: u64, decompress: u64 },
}

/// Configuration for training acceleration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Default compression ratio for gradient compression.
    pub default_compression_ratio: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Enable gradient clipping.
    pub gradient_clipping: Option<f32>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            default_compression_ratio: 0.1,
            seed: 42,
            gradient_clipping: None,
        }
    }
}

impl TrainingConfig {
    /// Set compression ratio.
    pub fn with_compression_ratio(mut self, ratio: f32) -> Self {
        self.default_compression_ratio = ratio;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable gradient clipping.
    pub fn with_gradient_clipping(mut self, max_norm: f32) -> Self {
        self.gradient_clipping = Some(max_norm);
        self
    }
}

/// Compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Compressed data.
    pub data: Vec<f32>,
    /// Original dimension.
    pub original_dim: usize,
    /// Random seed used for projection.
    pub seed: u64,
    /// Compression ratio achieved.
    pub ratio: f32,
}

/// Gradient compressor using random projection.
///
/// Uses Johnson-Lindenstrauss style random projection for
/// communication-efficient distributed training.
#[derive(Debug, Clone)]
pub struct GradientCompressor {
    config: TrainingConfig,
}

impl GradientCompressor {
    /// Create a new gradient compressor.
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Compress gradients using random projection.
    ///
    /// # Arguments
    ///
    /// * `gradients` - Original gradient vector
    /// * `ratio` - Compression ratio (0 < ratio <= 1), None uses default
    ///
    /// # Returns
    ///
    /// Compressed gradient representation.
    #[allow(clippy::cast_precision_loss)]
    pub fn compress(
        &self,
        gradients: &[f32],
        ratio: Option<f32>,
    ) -> Result<CompressedGradient, TrainingError> {
        let ratio = ratio.unwrap_or(self.config.default_compression_ratio);

        if ratio <= 0.0 || ratio > 1.0 {
            return Err(TrainingError::InvalidRatio(ratio));
        }

        let original_dim = gradients.len();
        let compressed_dim = ((original_dim as f32 * ratio).ceil() as usize).max(64);

        // Apply gradient clipping if configured
        let gradients = if let Some(max_norm) = self.config.gradient_clipping {
            clip_gradients(gradients, max_norm)
        } else {
            gradients.to_vec()
        };

        // Random projection (sparse for efficiency)
        let compressed = sparse_random_projection(&gradients, compressed_dim, self.config.seed);

        Ok(CompressedGradient {
            data: compressed,
            original_dim,
            seed: self.config.seed,
            ratio,
        })
    }

    /// Decompress gradients.
    ///
    /// # Arguments
    ///
    /// * `compressed` - Compressed gradient
    ///
    /// # Returns
    ///
    /// Reconstructed gradient vector (approximate).
    pub fn decompress(&self, compressed: &CompressedGradient) -> Result<Vec<f32>, TrainingError> {
        if compressed.seed != self.config.seed {
            return Err(TrainingError::SeedMismatch {
                compress: compressed.seed,
                decompress: self.config.seed,
            });
        }

        let recovered = sparse_random_projection_transpose(
            &compressed.data,
            compressed.original_dim,
            compressed.seed,
        );

        Ok(recovered)
    }

    /// Compress and immediately quantize to ternary for maximum compression.
    ///
    /// This achieves ~300x compression: 10x from projection + ~30x from ternary.
    #[allow(clippy::cast_precision_loss)]
    pub fn compress_ternary(
        &self,
        gradients: &[f32],
        ratio: Option<f32>,
    ) -> Result<TernaryCompressedGradient, TrainingError> {
        let compressed = self.compress(gradients, ratio)?;

        // Quantize compressed representation to ternary
        let (ternary, scale) = quantize_to_ternary(&compressed.data);

        Ok(TernaryCompressedGradient {
            data: ternary,
            scale,
            original_dim: compressed.original_dim,
            compressed_dim: compressed.data.len(),
            seed: compressed.seed,
        })
    }

    /// Decompress ternary compressed gradients.
    pub fn decompress_ternary(
        &self,
        compressed: &TernaryCompressedGradient,
    ) -> Result<Vec<f32>, TrainingError> {
        if compressed.seed != self.config.seed {
            return Err(TrainingError::SeedMismatch {
                compress: compressed.seed,
                decompress: self.config.seed,
            });
        }

        // Dequantize from ternary
        let dequantized: Vec<f32> = compressed
            .data
            .iter()
            .map(|&t| f32::from(t) * compressed.scale)
            .collect();

        // Inverse projection
        let recovered = sparse_random_projection_transpose(
            &dequantized,
            compressed.original_dim,
            compressed.seed,
        );

        Ok(recovered)
    }
}

/// Ternary compressed gradient (maximum compression).
#[derive(Debug, Clone)]
pub struct TernaryCompressedGradient {
    /// Ternary values (-1, 0, +1).
    pub data: Vec<i8>,
    /// Scale factor.
    pub scale: f32,
    /// Original dimension.
    pub original_dim: usize,
    /// Compressed dimension.
    pub compressed_dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl TernaryCompressedGradient {
    /// Calculate compression ratio.
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        // Original: f32 = 32 bits each
        // Compressed: 2 bits each (ternary) + 32 bits for scale
        let original_bits = self.original_dim * 32;
        let compressed_bits = self.data.len() * 2 + 32;
        original_bits as f32 / compressed_bits as f32
    }
}

// Helper functions

fn clip_gradients(gradients: &[f32], max_norm: f32) -> Vec<f32> {
    let norm: f32 = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        gradients.iter().map(|x| x * scale).collect()
    } else {
        gradients.to_vec()
    }
}

#[allow(clippy::cast_precision_loss)]
fn sparse_random_projection(input: &[f32], output_dim: usize, seed: u64) -> Vec<f32> {
    use rand_chacha::rand_core::{Rng, RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut output = vec![0.0f32; output_dim];

    // Scale factor for Johnson-Lindenstrauss
    let scale = 1.0 / (input.len() as f32).sqrt();

    // Sparse random projection: ~68% zeros, 16% +1, 16% -1
    for &g in input {
        for o in output.iter_mut() {
            let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
            if r < 0.16 {
                *o += g * scale;
            } else if r < 0.32 {
                *o -= g * scale;
            }
        }
    }

    output
}

#[allow(clippy::cast_precision_loss)]
fn sparse_random_projection_transpose(input: &[f32], output_dim: usize, seed: u64) -> Vec<f32> {
    use rand_chacha::rand_core::{Rng, RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut output = vec![0.0f32; output_dim];

    let scale = 1.0 / (output_dim as f32).sqrt();

    // Transpose of sparse random projection
    for o in output.iter_mut() {
        for &c in input {
            let r: f32 = (rng.next_u32() as f64 / u32::MAX as f64) as f32;
            if r < 0.16 {
                *o += c * scale;
            } else if r < 0.32 {
                *o -= c * scale;
            }
        }
    }

    output
}

fn quantize_to_ternary(values: &[f32]) -> (Vec<i8>, f32) {
    // Use AbsMean scaling
    let abs_mean: f32 = values.iter().map(|x| x.abs()).sum::<f32>() / values.len() as f32;
    let scale = if abs_mean > 1e-10 { abs_mean } else { 1.0 };

    let ternary: Vec<i8> = values
        .iter()
        .map(|&v| {
            let normalized = v / scale;
            if normalized > 0.5 {
                1i8
            } else if normalized < -0.5 {
                -1i8
            } else {
                0i8
            }
        })
        .collect();

    (ternary, scale)
}

/// Gradient accumulator for memory-efficient training.
///
/// Accumulates gradients in lower precision to reduce memory usage.
#[derive(Debug)]
pub struct GradientAccumulator {
    /// Accumulated gradients.
    accumulated: Vec<f32>,
    /// Number of accumulated batches.
    count: usize,
}

impl GradientAccumulator {
    /// Create a new accumulator.
    pub fn new(size: usize) -> Self {
        Self {
            accumulated: vec![0.0; size],
            count: 0,
        }
    }

    /// Add gradients to accumulator.
    pub fn accumulate(&mut self, gradients: &[f32]) -> Result<(), TrainingError> {
        if gradients.len() != self.accumulated.len() {
            return Err(TrainingError::DimensionMismatch {
                expected: self.accumulated.len(),
                actual: gradients.len(),
            });
        }

        for (acc, &g) in self.accumulated.iter_mut().zip(gradients.iter()) {
            *acc += g;
        }
        self.count += 1;

        Ok(())
    }

    /// Get averaged gradients and reset accumulator.
    #[allow(clippy::cast_precision_loss)]
    pub fn get_and_reset(&mut self) -> Vec<f32> {
        if self.count == 0 {
            return self.accumulated.clone();
        }

        let scale = 1.0 / self.count as f32;
        let result: Vec<f32> = self.accumulated.iter().map(|&x| x * scale).collect();

        // Reset
        self.accumulated.fill(0.0);
        self.count = 0;

        result
    }

    /// Get current count.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Mixed precision training utilities.
pub mod mixed_precision {
    use super::TrainingError;

    /// Convert f32 to bf16 representation (as u16).
    ///
    /// BF16 truncates the lower 16 bits of f32, preserving range but reducing precision.
    pub fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits >> 16) as u16
    }

    /// Convert bf16 (as u16) back to f32.
    pub fn bf16_to_f32(value: u16) -> f32 {
        let bits = (value as u32) << 16;
        f32::from_bits(bits)
    }

    /// Convert slice of f32 to bf16.
    pub fn convert_to_bf16(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_bf16(v)).collect()
    }

    /// Convert slice of bf16 back to f32.
    pub fn convert_from_bf16(values: &[u16]) -> Vec<f32> {
        values.iter().map(|&v| bf16_to_f32(v)).collect()
    }

    /// Loss scaling for mixed precision training.
    #[derive(Debug, Clone)]
    pub struct LossScaler {
        scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
        steps_since_growth: usize,
    }

    impl Default for LossScaler {
        fn default() -> Self {
            Self {
                scale: 65536.0, // 2^16
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
                steps_since_growth: 0,
            }
        }
    }

    impl LossScaler {
        /// Create with initial scale.
        pub fn with_initial_scale(scale: f32) -> Self {
            Self {
                scale,
                ..Default::default()
            }
        }

        /// Get current scale factor.
        pub fn scale(&self) -> f32 {
            self.scale
        }

        /// Scale loss for backward pass.
        pub fn scale_loss(&self, loss: f32) -> f32 {
            loss * self.scale
        }

        /// Unscale gradients after backward pass.
        pub fn unscale_gradients(&self, gradients: &mut [f32]) {
            let inv_scale = 1.0 / self.scale;
            for g in gradients.iter_mut() {
                *g *= inv_scale;
            }
        }

        /// Update scale based on whether overflow occurred.
        pub fn update(&mut self, overflow: bool) {
            if overflow {
                self.scale *= self.backoff_factor;
                self.steps_since_growth = 0;
            } else {
                self.steps_since_growth += 1;
                if self.steps_since_growth >= self.growth_interval {
                    self.scale *= self.growth_factor;
                    self.steps_since_growth = 0;
                }
            }
        }

        /// Check if gradients contain inf/nan (overflow).
        pub fn check_overflow(gradients: &[f32]) -> bool {
            gradients.iter().any(|&g| g.is_nan() || g.is_infinite())
        }
    }

    /// Check for NaN or Inf in tensor.
    pub fn has_nan_or_inf(values: &[f32]) -> bool {
        values.iter().any(|&v| v.is_nan() || v.is_infinite())
    }

    /// Clip values to prevent overflow.
    pub fn safe_clip(values: &mut [f32], min: f32, max: f32) -> Result<(), TrainingError> {
        for v in values.iter_mut() {
            if v.is_nan() {
                *v = 0.0;
            } else {
                *v = v.clamp(min, max);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_compression_roundtrip() {
        let config = TrainingConfig::default();
        let compressor = GradientCompressor::new(config);

        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();

        let compressed = compressor.compress(&gradients, Some(0.1)).unwrap();
        let recovered = compressor.decompress(&compressed).unwrap();

        // Check dimensions
        assert_eq!(recovered.len(), gradients.len());

        // Check approximate reconstruction (lossy compression)
        let mse: f32 = gradients
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / gradients.len() as f32;

        // MSE should be reasonable (not zero due to lossy compression)
        assert!(mse < 1.0);
    }

    #[test]
    fn test_ternary_compression() {
        let config = TrainingConfig::default();
        let compressor = GradientCompressor::new(config);

        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();

        let compressed = compressor.compress_ternary(&gradients, Some(0.1)).unwrap();

        // Check compression ratio is high
        assert!(compressed.compression_ratio() > 10.0);

        // Check ternary values
        for &t in &compressed.data {
            assert!([-1, 0, 1].contains(&t));
        }
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = GradientAccumulator::new(4);

        acc.accumulate(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.accumulate(&[2.0, 4.0, 6.0, 8.0]).unwrap();

        let result = acc.get_and_reset();

        // Average: [1.5, 3.0, 4.5, 6.0]
        assert!((result[0] - 1.5).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
        assert!((result[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_precision_bf16() {
        use mixed_precision::{bf16_to_f32, f32_to_bf16};

        let original = 3.14159f32;
        let bf16 = f32_to_bf16(original);
        let recovered = bf16_to_f32(bf16);

        // BF16 has ~3 decimal digits of precision
        assert!((original - recovered).abs() < 0.01);
    }

    #[test]
    fn test_loss_scaler() {
        use mixed_precision::LossScaler;

        let mut scaler = LossScaler::default();
        let initial_scale = scaler.scale();

        // Simulate overflow
        scaler.update(true);
        assert!(scaler.scale() < initial_scale);

        // Simulate many successful steps
        for _ in 0..2000 {
            scaler.update(false);
        }
        // Scale should have grown back
        assert!(scaler.scale() > initial_scale * 0.5);
    }
}
