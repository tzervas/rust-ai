//! Model configuration for Tritter transformer.
//!
//! Provides configuration for different model sizes from 100M to 70B parameters.

use serde::{Deserialize, Serialize};

/// Configuration for the Tritter model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritterConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA, None = MHA)
    pub num_kv_heads: Option<usize>,
    /// FFN intermediate dimension (typically 2.7x hidden_size)
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Use BitNet ternary quantization
    pub use_bitnet: bool,
    /// Use QK-norm in attention
    pub use_qk_norm: bool,
    /// RoPE theta for positional encoding
    pub rope_theta: f32,
    /// Gradient checkpointing (reduces memory by ~50% at cost of ~33% extra compute)
    /// Note: Currently a documentation flag - Candle does not support native checkpointing.
    /// When enabled, memory estimates will reflect checkpointed usage.
    pub gradient_checkpointing: bool,
    /// Checkpoint every N layers (only used when gradient_checkpointing is true)
    /// Lower values = more memory savings but more recomputation
    pub checkpoint_every_n_layers: usize,
}

impl Default for TritterConfig {
    fn default() -> Self {
        Self::small_100m()
    }
}

impl TritterConfig {
    /// 100M parameter configuration (for testing/validation)
    pub fn small_100m() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: None,      // MHA
            intermediate_size: 2048, // ~2.7x hidden
            vocab_size: 65536,
            max_seq_length: 2048,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: false,
            checkpoint_every_n_layers: 1,
        }
    }

    /// 500M parameter configuration
    pub fn medium_500m() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: None,      // MHA
            intermediate_size: 2816, // ~2.75x hidden
            vocab_size: 65536,
            max_seq_length: 4096,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 4,
        }
    }

    /// 1B parameter configuration
    pub fn large_1b() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: Some(8),   // GQA 2:1
            intermediate_size: 5632, // ~2.75x hidden
            vocab_size: 65536,
            max_seq_length: 8192,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 4,
        }
    }

    /// 1B parameter configuration for aNa model
    ///
    /// Optimized for the aNa (Autonomous Networked Assistant) model:
    /// - 2048 hidden dimensions
    /// - 24 transformer layers
    /// - 16 attention heads
    /// - 32K vocabulary (for efficiency with tool usage tokens)
    /// - 4096 max sequence length (balances context with memory)
    /// - 8192 intermediate size (4x hidden for MLP)
    ///
    /// # Example
    /// ```
    /// use tritter_model_rs::TritterConfig;
    /// let config = TritterConfig::preset_1b();
    /// assert_eq!(config.hidden_size, 2048);
    /// assert_eq!(config.num_layers, 24);
    /// ```
    pub fn preset_1b() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: Some(8),   // GQA 2:1
            intermediate_size: 8192, // 4x hidden for better MLP capacity
            vocab_size: 32000,
            max_seq_length: 4096,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 4,
        }
    }

    /// 3B parameter configuration
    pub fn xlarge_3b() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 26,
            num_heads: 20,
            num_kv_heads: Some(5),   // GQA 4:1
            intermediate_size: 6912, // ~2.7x hidden
            vocab_size: 65536,
            max_seq_length: 131072, // 128K context
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 4,
        }
    }

    /// 7B parameter configuration
    pub fn huge_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: Some(8),    // GQA 4:1
            intermediate_size: 11008, // ~2.7x hidden
            vocab_size: 65536,
            max_seq_length: 131072, // 128K context
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bitnet: true,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: true,
            checkpoint_every_n_layers: 8,
        }
    }

    /// Test configuration (minimal for unit tests)
    pub fn test() -> Self {
        Self {
            hidden_size: 64,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: None,
            intermediate_size: 128,
            vocab_size: 256,
            max_seq_length: 64,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            use_bitnet: false,
            use_qk_norm: true,
            rope_theta: 10000.0,
            gradient_checkpointing: false,
            checkpoint_every_n_layers: 1,
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Get number of KV heads (defaults to num_heads for MHA)
    pub fn kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    /// Estimate parameter count
    pub fn parameter_count(&self) -> usize {
        let embed = self.vocab_size * self.hidden_size;
        let attn_qkv =
            self.hidden_size * (self.hidden_size + 2 * self.kv_heads() * self.head_dim());
        let attn_out = self.hidden_size * self.hidden_size;
        let ffn = self.hidden_size * self.intermediate_size * 3; // gate, up, down
        let layer = attn_qkv + attn_out + ffn + 4 * self.hidden_size; // + norms
        let lm_head = self.hidden_size * self.vocab_size;

        embed + self.num_layers * layer + lm_head
    }

    /// Estimate memory usage in bytes (FP32)
    pub fn memory_estimate_fp32(&self) -> usize {
        self.parameter_count() * 4
    }

    /// Estimate memory usage in bytes (BitNet ternary + scales)
    pub fn memory_estimate_bitnet(&self) -> usize {
        // 2 bits per weight + f32 scale per 64 weights
        let bits_per_weight = 2.0 + 32.0 / 64.0;
        ((self.parameter_count() as f64) * bits_per_weight / 8.0) as usize
    }

    /// Estimate activation memory per layer in bytes (FP32)
    ///
    /// Includes:
    /// - Hidden states: batch * seq * hidden * 4 bytes
    /// - Attention scores: batch * heads * seq * seq * 4 bytes
    /// - MLP intermediates: batch * seq * intermediate * 4 bytes (gate + up)
    pub fn activation_memory_per_layer(&self, batch_size: usize, seq_len: usize) -> usize {
        let bytes_per_elem = 4; // FP32

        // Hidden states (input to layer + output)
        let hidden_mem = 2 * batch_size * seq_len * self.hidden_size * bytes_per_elem;

        // Attention: Q, K, V projections + attention scores
        let qkv_mem = 3 * batch_size * seq_len * self.hidden_size * bytes_per_elem;
        let attn_scores = batch_size * self.num_heads * seq_len * seq_len * bytes_per_elem;

        // MLP: gate and up projections (intermediate size)
        let mlp_mem = 2 * batch_size * seq_len * self.intermediate_size * bytes_per_elem;

        hidden_mem + qkv_mem + attn_scores + mlp_mem
    }

    /// Estimate total activation memory for training in bytes
    ///
    /// # Arguments
    /// * `batch_size` - Training batch size
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Estimated activation memory. If `gradient_checkpointing` is enabled,
    /// returns reduced estimate based on `checkpoint_every_n_layers`.
    pub fn activation_memory_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        let per_layer = self.activation_memory_per_layer(batch_size, seq_len);

        if self.gradient_checkpointing {
            // With checkpointing, we only store activations every N layers
            // Other layers are recomputed during backward pass
            let stored_layers = self.num_layers.div_ceil(self.checkpoint_every_n_layers);
            stored_layers * per_layer
        } else {
            // Without checkpointing, store all layer activations
            self.num_layers * per_layer
        }
    }

    /// Estimate total training memory in bytes
    ///
    /// Includes:
    /// - Model parameters (FP32 or BitNet depending on use_bitnet)
    /// - Optimizer states (2x parameters for Adam m and v)
    /// - Gradients (1x parameters)
    /// - Activations (depends on checkpointing)
    ///
    /// # Arguments
    /// * `batch_size` - Training batch size
    /// * `seq_len` - Sequence length
    pub fn total_training_memory_estimate(
        &self,
        batch_size: usize,
        seq_len: usize,
    ) -> TrainingMemoryEstimate {
        let params = if self.use_bitnet {
            self.memory_estimate_bitnet()
        } else {
            self.memory_estimate_fp32()
        };

        // Adam optimizer: m (first moment) + v (second moment) - both FP32
        let optimizer_states = 2 * self.parameter_count() * 4;

        // Gradients (FP32)
        let gradients = self.parameter_count() * 4;

        // Activations
        let activations = self.activation_memory_estimate(batch_size, seq_len);
        let activations_no_checkpoint =
            self.num_layers * self.activation_memory_per_layer(batch_size, seq_len);

        TrainingMemoryEstimate {
            parameters: params,
            optimizer_states,
            gradients,
            activations,
            activations_without_checkpointing: activations_no_checkpoint,
            total: params + optimizer_states + gradients + activations,
            checkpointing_enabled: self.gradient_checkpointing,
            checkpoint_every_n_layers: self.checkpoint_every_n_layers,
        }
    }

    /// Calculate memory reduction factor from checkpointing
    ///
    /// Returns a value between 0 and 1, where lower means more memory saved.
    pub fn checkpointing_memory_factor(&self) -> f64 {
        if !self.gradient_checkpointing || self.num_layers == 0 {
            1.0
        } else {
            let stored = self.num_layers.div_ceil(self.checkpoint_every_n_layers);
            stored as f64 / self.num_layers as f64
        }
    }
}

/// Detailed breakdown of training memory requirements
#[derive(Debug, Clone)]
pub struct TrainingMemoryEstimate {
    /// Model parameter memory in bytes
    pub parameters: usize,
    /// Adam optimizer state memory (m + v) in bytes
    pub optimizer_states: usize,
    /// Gradient memory in bytes
    pub gradients: usize,
    /// Activation memory in bytes (with checkpointing if enabled)
    pub activations: usize,
    /// Activation memory without checkpointing (for comparison)
    pub activations_without_checkpointing: usize,
    /// Total estimated memory in bytes
    pub total: usize,
    /// Whether gradient checkpointing is enabled
    pub checkpointing_enabled: bool,
    /// Checkpoint frequency (layers between checkpoints)
    pub checkpoint_every_n_layers: usize,
}

impl TrainingMemoryEstimate {
    /// Format the estimate as a human-readable string
    pub fn format(&self) -> String {
        let format_bytes = |b: usize| -> String {
            if b >= 1024 * 1024 * 1024 {
                format!("{:.2} GB", b as f64 / (1024.0 * 1024.0 * 1024.0))
            } else if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / (1024.0 * 1024.0))
            } else if b >= 1024 {
                format!("{:.2} KB", b as f64 / 1024.0)
            } else {
                format!("{} bytes", b)
            }
        };

        let activation_savings = if self.checkpointing_enabled {
            let saved = self
                .activations_without_checkpointing
                .saturating_sub(self.activations);
            let pct = if self.activations_without_checkpointing > 0 {
                100.0 * saved as f64 / self.activations_without_checkpointing as f64
            } else {
                0.0
            };
            format!(" (saved {}, {:.1}% reduction)", format_bytes(saved), pct)
        } else {
            String::new()
        };

        format!(
            "Training Memory Estimate:\n\
             - Parameters:      {}\n\
             - Optimizer:       {}\n\
             - Gradients:       {}\n\
             - Activations:     {}{}\n\
             - Total:           {}\n\
             - Checkpointing:   {} (every {} layers)",
            format_bytes(self.parameters),
            format_bytes(self.optimizer_states),
            format_bytes(self.gradients),
            format_bytes(self.activations),
            activation_savings,
            format_bytes(self.total),
            if self.checkpointing_enabled {
                "enabled"
            } else {
                "disabled"
            },
            self.checkpoint_every_n_layers,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let c100m = TritterConfig::small_100m();
        let c500m = TritterConfig::medium_500m();
        let c1b = TritterConfig::large_1b();

        // Parameter counts should scale approximately
        let p100m = c100m.parameter_count();
        let p500m = c500m.parameter_count();
        let p1b = c1b.parameter_count();

        assert!(p100m < p500m);
        assert!(p500m < p1b);
        assert!(p100m > 50_000_000); // At least 50M
        assert!(p1b > 500_000_000); // At least 500M
    }

    #[test]
    fn test_preset_1b() {
        let config = TritterConfig::preset_1b();

        // Verify aNa model specifications
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.max_seq_length, 4096);
        assert_eq!(config.intermediate_size, 8192);

        // Verify BitNet and other features
        assert!(config.use_bitnet);
        assert!(config.use_qk_norm);
        assert!(config.gradient_checkpointing);

        // Parameter count should be in ~1B range (many "1B" models are 1-2B)
        let params = config.parameter_count();
        assert!(
            params > 800_000_000,
            "Expected > 800M params, got {}",
            params
        );
        assert!(
            params < 2_000_000_000,
            "Expected < 2B params, got {}",
            params
        );
    }

    #[test]
    fn test_memory_estimate() {
        let config = TritterConfig::small_100m();
        let fp32 = config.memory_estimate_fp32();
        let bitnet = config.memory_estimate_bitnet();

        // BitNet should be much smaller
        assert!(bitnet < fp32 / 4);
    }

    #[test]
    fn test_activation_memory_checkpointing() {
        let batch_size = 4;
        let seq_len = 2048;

        // Config with checkpointing disabled
        let mut config_no_cp = TritterConfig::small_100m();
        config_no_cp.gradient_checkpointing = false;

        // Config with checkpointing enabled (every 4 layers)
        let mut config_cp = TritterConfig::small_100m();
        config_cp.gradient_checkpointing = true;
        config_cp.checkpoint_every_n_layers = 4;

        let mem_no_cp = config_no_cp.activation_memory_estimate(batch_size, seq_len);
        let mem_cp = config_cp.activation_memory_estimate(batch_size, seq_len);

        // With checkpointing every 4 layers on a 12-layer model:
        // We store ceil(12/4) = 3 layers instead of 12
        // So memory should be reduced to ~25%
        assert!(mem_cp < mem_no_cp);
        assert!(mem_cp <= mem_no_cp / 3); // Should be roughly 1/4
    }

    #[test]
    fn test_checkpointing_memory_factor() {
        let mut config = TritterConfig::small_100m();
        config.num_layers = 32;
        config.checkpoint_every_n_layers = 8;

        // Without checkpointing
        config.gradient_checkpointing = false;
        assert!((config.checkpointing_memory_factor() - 1.0).abs() < 0.01);

        // With checkpointing: ceil(32/8) = 4 stored, factor = 4/32 = 0.125
        config.gradient_checkpointing = true;
        let factor = config.checkpointing_memory_factor();
        assert!((factor - 0.125).abs() < 0.01);
    }

    #[test]
    fn test_total_training_memory_estimate() {
        let config = TritterConfig::test();
        let estimate = config.total_training_memory_estimate(2, 16);

        // Basic sanity checks
        assert!(estimate.parameters > 0);
        assert!(estimate.optimizer_states > 0);
        assert!(estimate.gradients > 0);
        assert!(estimate.activations > 0);
        assert!(estimate.total > estimate.parameters);

        // Verify breakdown adds up
        let expected_total = estimate.parameters
            + estimate.optimizer_states
            + estimate.gradients
            + estimate.activations;
        assert_eq!(estimate.total, expected_total);
    }

    #[test]
    fn test_memory_estimate_format() {
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;

        let estimate = config.total_training_memory_estimate(2, 16);
        let formatted = estimate.format();

        // Should contain key sections
        assert!(formatted.contains("Parameters:"));
        assert!(formatted.contains("Optimizer:"));
        assert!(formatted.contains("Activations:"));
        assert!(formatted.contains("Checkpointing:"));
    }
}
