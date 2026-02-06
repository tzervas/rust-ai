//! Shared utilities for LLaMA model implementations.
//!
//! This module contains common code shared between `LoraLlama` and `QLoraLlama`:
//! - Cache structure for KV cache and rotary embeddings
//! - Rotary embedding application helpers
//! - Model preparation utilities for k-bit training

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::RmsNorm;
use candle_transformers::models::llama::Config;
use std::collections::HashMap;

/// Cache for KV cache and rotary embeddings.
///
/// Precomputes rotary position embeddings and manages KV cache per layer.
pub struct Cache {
    /// Precomputed cosine values for rotary embeddings.
    pub cos: Tensor,
    /// Precomputed sine values for rotary embeddings.
    pub sin: Tensor,
    /// KV cache per layer: (key, value) tensors.
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    /// Cached attention masks by sequence length.
    pub masks: HashMap<usize, Tensor>,
    /// Whether to use KV caching (for inference).
    pub use_kv_cache: bool,
    /// Device where cache is stored.
    pub device: Device,
}

impl Cache {
    /// Create a new cache with precomputed rotary embeddings.
    ///
    /// # Errors
    /// Returns error if tensor creation fails.
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> CandleResult<Self> {
        let num_layers = config.num_hidden_layers;
        let rope_theta = config.rope_theta;
        let head_dim = config.hidden_size / config.num_attention_heads;

        // Precompute rotary embeddings - use step_by(2) like candle-transformers
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
            .collect();

        let theta = Tensor::from_vec(theta, (head_dim / 2,), device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            kvs: vec![None; num_layers],
            masks: HashMap::new(),
            use_kv_cache,
            device: device.clone(),
        })
    }

    /// Get or create a causal attention mask for the given sequence length.
    ///
    /// # Errors
    /// Returns error if tensor creation fails.
    pub fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Reset KV cache (useful when starting a new sequence).
    pub fn reset(&mut self) {
        for kv in &mut self.kvs {
            *kv = None;
        }
    }
}

/// Apply rotary position embeddings to query/key tensors.
///
/// # Arguments
/// * `x` - Tensor of shape `[batch, heads, seq_len, head_dim]`
/// * `index_pos` - Starting position index
/// * `cache` - Cache containing precomputed cos/sin
///
/// # Errors
/// Returns error if tensor operations fail.
pub fn apply_rotary_emb(x: &Tensor, index_pos: usize, cache: &Cache) -> CandleResult<Tensor> {
    let (_b_sz, _num_heads, seq_len, _head_dim) = x.dims4()?;
    let cos = cache.cos.narrow(0, index_pos, seq_len)?;
    let sin = cache.sin.narrow(0, index_pos, seq_len)?;
    candle_nn::rotary_emb::rope(x, &cos, &sin)
}

/// Repeat KV heads for grouped-query attention.
///
/// # Arguments
/// * `x` - Tensor of shape `[batch, kv_heads, seq_len, head_dim]`
/// * `n_rep` - Number of times to repeat each KV head
///
/// # Errors
/// Returns error if tensor operations fail.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    candle_transformers::utils::repeat_kv(x, n_rep)
}

/// Apply masked fill operation for attention.
///
/// # Arguments
/// * `on_false` - Values when mask is false
/// * `mask` - Boolean mask tensor
/// * `on_true` - Value to fill when mask is true (typically `-inf`)
///
/// # Errors
/// Returns error if tensor operations fail.
pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CandleResult<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// Trait for modules that can be upcasted to FP32.
///
/// Used by `prepare_model_for_qlora_training` to upcast non-quantized modules.
pub trait UpcastToFp32 {
    /// Upcast module weights to FP32.
    ///
    /// # Errors
    /// Returns error if dtype conversion fails.
    fn upcast_to_fp32(&mut self) -> CandleResult<()>;
}

/// Model preparation configuration for QLoRA training.
#[derive(Debug, Clone)]
pub struct PrepareForTrainingConfig {
    /// Whether to enable gradient checkpointing (future work).
    pub enable_gradient_checkpointing: bool,
    /// Whether to upcast layer norms to FP32 (recommended for stability).
    pub upcast_layernorms: bool,
    /// Whether to upcast embeddings to FP32.
    pub upcast_embeddings: bool,
    /// Whether to upcast lm_head to FP32.
    pub upcast_lm_head: bool,
}

impl Default for PrepareForTrainingConfig {
    fn default() -> Self {
        Self {
            enable_gradient_checkpointing: false,
            upcast_layernorms: true,
            upcast_embeddings: true,
            upcast_lm_head: true,
        }
    }
}

/// Upcast an RmsNorm layer to FP32.
///
/// Layer norms require FP32 for numerical stability during QLoRA training.
/// Using FP16/BF16 for layer norms can cause NaN losses.
///
/// In this implementation, the effective dtype of the `RmsNorm` parameters
/// is controlled at construction time via the `VarBuilder`'s `DType`
/// (e.g. using `DType::F32` for layer norms). Because the `RmsNorm` type
/// does not currently expose its internal weight tensor, this helper
/// cannot safely recreate the layer with a different dtype.
///
/// As a result, this function is currently a no-op "upcast": it returns a
/// cloned `RmsNorm` and assumes that the caller has already constructed
/// the layer with an appropriate dtype.
///
/// # References
/// - PEFT: `prepare_model_for_kbit_training` upcasts all 1D params to FP32
/// - QLoRA paper Section 4.1: FP16 compute causes 20% training failure rate
///
/// This function does not return errors in the current implementation.
pub fn upcast_rms_norm(norm: &RmsNorm, _device: &Device) -> CandleResult<RmsNorm> {
    // NOTE: Real upcasting should be handled when creating `RmsNorm`
    // via the `VarBuilder` dtype. Here we simply return a clone to
    // provide a stable, non-erroring helper.
    Ok(norm.clone())
}

/// Helper to create linear layer without bias (LLaMA models don't use bias).
pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> CandleResult<candle_nn::Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(candle_nn::Linear::new(weight, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 1000,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
            use_flash_attn: false,
            bos_token_id: Some(1),
            eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(2)),
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_cache_creation() {
        let config = create_test_config();
        let device = Device::Cpu;
        let cache = Cache::new(false, DType::F32, &config, &device);
        assert!(cache.is_ok());

        let cache = cache.unwrap();
        assert_eq!(cache.kvs.len(), config.num_hidden_layers);
        assert!(!cache.use_kv_cache);
    }

    #[test]
    fn test_cache_mask() {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut cache = Cache::new(false, DType::F32, &config, &device).unwrap();

        let mask = cache.mask(4).unwrap();
        assert_eq!(mask.dims(), &[4, 4]);
    }

    #[test]
    fn test_rotary_emb_shapes() {
        let config = create_test_config();
        let device = Device::Cpu;
        let cache = Cache::new(false, DType::F32, &config, &device).unwrap();

        // cos/sin should be [max_pos, head_dim/2]
        let head_dim = config.hidden_size / config.num_attention_heads;
        assert_eq!(
            cache.cos.dims(),
            &[config.max_position_embeddings, head_dim / 2]
        );
        assert_eq!(
            cache.sin.dims(),
            &[config.max_position_embeddings, head_dim / 2]
        );
    }
}
