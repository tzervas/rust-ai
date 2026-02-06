//! Custom LLaMA implementation with per-layer LoRA injection.
//!
//! This module provides a LoRA-aware LLaMA model that properly integrates
//! adapter layers at each attention/MLP layer, ensuring proper gradient flow
//! through the adapter weights during training.
//!
//! Key differences from standard candle-transformers Llama:
//! - LoRA layers injected at Q, K, V, O projections in attention
//! - LoRA layers injected at gate, up, down projections in MLP
//! - LoRA forward returns base + delta for proper gradient flow
//!
//! Based on candle-transformers 0.9.1 Llama implementation.

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, VarMap};
use candle_transformers::models::llama::Config;
use std::collections::HashMap;

#[cfg(feature = "peft")]
use peft_rs::{Adapter, LoraConfig, LoraLayer};

/// Helper to create linear layer without bias (LLaMA models don't use bias)
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CandleResult<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// Cache for KV cache and rotary embeddings
pub struct Cache {
    /// Precomputed cosine values for rotary embeddings.
    pub cos: Tensor,
    /// Precomputed sine values for rotary embeddings.
    pub sin: Tensor,
    /// KV cache per layer: (key, value) tensors.
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    masks: HashMap<usize, Tensor>,
    /// Whether to use KV caching (for inference).
    pub use_kv_cache: bool,
    device: Device,
}

impl Cache {
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

    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
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
}

/// LoRA-aware attention layer that injects adapters into Q, K, V, and O projections.
pub struct LoraAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[cfg(feature = "peft")]
    q_lora: Option<LoraLayer>,
    #[cfg(feature = "peft")]
    k_lora: Option<LoraLayer>,
    #[cfg(feature = "peft")]
    v_lora: Option<LoraLayer>,
    #[cfg(feature = "peft")]
    o_lora: Option<LoraLayer>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    max_position_embeddings: usize,
}

impl LoraAttention {
    /// Create a new LoRA-aware attention layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        max_position_embeddings: usize,
        vb: VarBuilder,
        lora_config: Option<&LoraConfig>,
        lora_vb: Option<VarBuilder>,
        layer_idx: usize,
    ) -> CandleResult<Self> {
        let head_dim = hidden_size / num_attention_heads;
        let size_q = head_dim * num_attention_heads;
        let size_kv = head_dim * num_key_value_heads;

        // Load base model weights (no bias in LLaMA)
        let q_proj = linear_no_bias(hidden_size, size_q, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, hidden_size, vb.pp("o_proj"))?;

        // Create LoRA adapters if config provided
        #[cfg(feature = "peft")]
        let (q_lora, k_lora, v_lora, o_lora) = if let (Some(cfg), Some(vb)) = (lora_config, lora_vb)
        {
            let mut q_lora = None;
            let mut k_lora = None;
            let mut v_lora = None;
            let mut o_lora = None;

            if cfg.target_modules.contains(&"q_proj".to_string()) {
                let name = format!("model.layers.{}.self_attn.q_proj", layer_idx);
                q_lora = Some(
                    LoraLayer::new(hidden_size, size_q, cfg.clone(), vb.pp(&name)).map_err(
                        |e| candle_core::Error::Msg(format!("Failed to create q_lora: {}", e)),
                    )?,
                );
            }
            if cfg.target_modules.contains(&"k_proj".to_string()) {
                let name = format!("model.layers.{}.self_attn.k_proj", layer_idx);
                k_lora = Some(
                    LoraLayer::new(hidden_size, size_kv, cfg.clone(), vb.pp(&name)).map_err(
                        |e| candle_core::Error::Msg(format!("Failed to create k_lora: {}", e)),
                    )?,
                );
            }
            if cfg.target_modules.contains(&"v_proj".to_string()) {
                let name = format!("model.layers.{}.self_attn.v_proj", layer_idx);
                v_lora = Some(
                    LoraLayer::new(hidden_size, size_kv, cfg.clone(), vb.pp(&name)).map_err(
                        |e| candle_core::Error::Msg(format!("Failed to create v_lora: {}", e)),
                    )?,
                );
            }
            if cfg.target_modules.contains(&"o_proj".to_string()) {
                let name = format!("model.layers.{}.self_attn.o_proj", layer_idx);
                o_lora = Some(
                    LoraLayer::new(size_q, hidden_size, cfg.clone(), vb.pp(&name)).map_err(
                        |e| candle_core::Error::Msg(format!("Failed to create o_lora: {}", e)),
                    )?,
                );
            }

            (q_lora, k_lora, v_lora, o_lora)
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            #[cfg(feature = "peft")]
            q_lora,
            #[cfg(feature = "peft")]
            k_lora,
            #[cfg(feature = "peft")]
            v_lora,
            #[cfg(feature = "peft")]
            o_lora,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            max_position_embeddings,
        })
    }

    /// Apply rotary embeddings using precomputed cos/sin from cache.
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
    ) -> CandleResult<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// Forward pass with LoRA injection at each projection.
    /// Matches candle-transformers signature: (x, index_pos, block_idx, cache)
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        // Q projection with optional LoRA
        let base_q = self.q_proj.forward(x)?;
        #[cfg(feature = "peft")]
        let q = if let Some(lora) = &self.q_lora {
            lora.forward(x, Some(&base_q)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA q_proj forward failed: {}", e))
            })?
        } else {
            base_q
        };
        #[cfg(not(feature = "peft"))]
        let q = base_q;

        // K projection with optional LoRA
        let base_k = self.k_proj.forward(x)?;
        #[cfg(feature = "peft")]
        let k = if let Some(lora) = &self.k_lora {
            lora.forward(x, Some(&base_k)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA k_proj forward failed: {}", e))
            })?
        } else {
            base_k
        };
        #[cfg(not(feature = "peft"))]
        let k = base_k;

        // V projection with optional LoRA
        let base_v = self.v_proj.forward(x)?;
        #[cfg(feature = "peft")]
        let v = if let Some(lora) = &self.v_lora {
            lora.forward(x, Some(&base_v)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA v_proj forward failed: {}", e))
            })?
        } else {
            base_v
        };
        #[cfg(not(feature = "peft"))]
        let v = base_v;

        // Reshape for multi-head attention
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // Update KV cache
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // Grouped-query attention: repeat KV heads
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Attention computation
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;

        // Reshape back
        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, self.hidden_size])?;

        // O projection with optional LoRA
        let base_output = self.o_proj.forward(&y)?;
        #[cfg(feature = "peft")]
        let output = if let Some(lora) = &self.o_lora {
            lora.forward(&y, Some(&base_output)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA o_proj forward failed: {}", e))
            })?
        } else {
            base_output
        };
        #[cfg(not(feature = "peft"))]
        let output = base_output;

        Ok(output)
    }

    /// Repeat KV heads for grouped-query attention.
    fn repeat_kv(&self, x: Tensor) -> CandleResult<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CandleResult<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// LoRA-aware MLP layer that injects adapters into gate, up, and down projections.
pub struct LoraMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    #[cfg(feature = "peft")]
    gate_lora: Option<LoraLayer>,
    #[cfg(feature = "peft")]
    up_lora: Option<LoraLayer>,
    #[cfg(feature = "peft")]
    down_lora: Option<LoraLayer>,
}

impl LoraMlp {
    /// Create a new LoRA-aware MLP layer.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        lora_config: Option<&LoraConfig>,
        lora_vb: Option<VarBuilder>,
        layer_idx: usize,
    ) -> CandleResult<Self> {
        // Load base model weights (no bias in LLaMA)
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        // Create LoRA adapters if config provided
        #[cfg(feature = "peft")]
        let (gate_lora, up_lora, down_lora) = if let (Some(cfg), Some(vb)) = (lora_config, lora_vb)
        {
            let mut gate_lora = None;
            let mut up_lora = None;
            let mut down_lora = None;

            if cfg.target_modules.contains(&"gate_proj".to_string()) {
                let name = format!("model.layers.{}.mlp.gate_proj", layer_idx);
                gate_lora = Some(
                    LoraLayer::new(hidden_size, intermediate_size, cfg.clone(), vb.pp(&name))
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("Failed to create gate_lora: {}", e))
                        })?,
                );
            }
            if cfg.target_modules.contains(&"up_proj".to_string()) {
                let name = format!("model.layers.{}.mlp.up_proj", layer_idx);
                up_lora = Some(
                    LoraLayer::new(hidden_size, intermediate_size, cfg.clone(), vb.pp(&name))
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("Failed to create up_lora: {}", e))
                        })?,
                );
            }
            if cfg.target_modules.contains(&"down_proj".to_string()) {
                let name = format!("model.layers.{}.mlp.down_proj", layer_idx);
                down_lora = Some(
                    LoraLayer::new(intermediate_size, hidden_size, cfg.clone(), vb.pp(&name))
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("Failed to create down_lora: {}", e))
                        })?,
                );
            }

            (gate_lora, up_lora, down_lora)
        } else {
            (None, None, None)
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            #[cfg(feature = "peft")]
            gate_lora,
            #[cfg(feature = "peft")]
            up_lora,
            #[cfg(feature = "peft")]
            down_lora,
        })
    }

    /// Forward pass with LoRA injection at each projection.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Gate projection with optional LoRA
        let base_gate = self.gate_proj.forward(x)?;
        #[cfg(feature = "peft")]
        let gate = if let Some(lora) = &self.gate_lora {
            lora.forward(x, Some(&base_gate)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA gate_proj forward failed: {}", e))
            })?
        } else {
            base_gate
        };
        #[cfg(not(feature = "peft"))]
        let gate = base_gate;
        let gate = candle_nn::ops::silu(&gate)?;

        // Up projection with optional LoRA
        let base_up = self.up_proj.forward(x)?;
        #[cfg(feature = "peft")]
        let up = if let Some(lora) = &self.up_lora {
            lora.forward(x, Some(&base_up)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA up_proj forward failed: {}", e))
            })?
        } else {
            base_up
        };
        #[cfg(not(feature = "peft"))]
        let up = base_up;

        // Gated activation
        let hidden = (gate * up)?;

        // Down projection with optional LoRA
        let base_output = self.down_proj.forward(&hidden)?;
        #[cfg(feature = "peft")]
        let output = if let Some(lora) = &self.down_lora {
            lora.forward(&hidden, Some(&base_output)).map_err(|e| {
                candle_core::Error::Msg(format!("LoRA down_proj forward failed: {}", e))
            })?
        } else {
            base_output
        };
        #[cfg(not(feature = "peft"))]
        let output = base_output;

        Ok(output)
    }
}

/// Single transformer layer with LoRA-aware attention and MLP.
pub struct LoraTransformerBlock {
    attention: LoraAttention,
    mlp: LoraMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LoraTransformerBlock {
    /// Create a new LoRA-aware transformer block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rms_norm_eps: f64,
        max_position_embeddings: usize,
        vb: VarBuilder,
        lora_config: Option<&LoraConfig>,
        lora_vb: Option<VarBuilder>,
    ) -> CandleResult<Self> {
        let attention = LoraAttention::new(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            vb.pp("self_attn"),
            lora_config,
            lora_vb.as_ref().map(|v| v.pp("self_attn")),
            layer_idx,
        )?;

        let mlp = LoraMlp::new(
            hidden_size,
            intermediate_size,
            vb.pp("mlp"),
            lora_config,
            lora_vb.as_ref().map(|v| v.pp("mlp")),
            layer_idx,
        )?;

        let input_layernorm =
            candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass through the transformer block.
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> CandleResult<Tensor> {
        // Pre-norm for attention
        // Use forward_diff to preserve gradient tracking (regular forward uses no-grad kernel)
        let normed = self.input_layernorm.forward_diff(x)?;
        let attn_output = self
            .attention
            .forward(&normed, index_pos, block_idx, cache)?;
        let x = (x + attn_output)?;

        // Pre-norm for MLP
        // Use forward_diff to preserve gradient tracking
        let normed = self.post_attention_layernorm.forward_diff(&x)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let x = (x + mlp_output)?;

        Ok(x)
    }
}

/// Full LoRA-aware LLaMA model with per-layer injection.
pub struct LoraLlama {
    embed_tokens: Embedding,
    layers: Vec<LoraTransformerBlock>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Config,
    device: Device,
}

impl LoraLlama {
    /// Create a new LoRA-aware LLaMA model with LoRA layers created internally.
    #[cfg(feature = "peft")]
    pub fn new_with_lora(
        config: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
        lora_varmap: &VarMap,
    ) -> CandleResult<Self> {
        let device = vb.device().clone();

        // Create VarBuilder from VarMap for trainable LoRA parameters
        let lora_vb = VarBuilder::from_varmap(lora_varmap, DType::F32, &device);

        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model").pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("model").pp("layers");
        let lora_vb_layers = lora_vb.pp("model").pp("layers");

        for layer_idx in 0..config.num_hidden_layers {
            let layer = LoraTransformerBlock::new(
                layer_idx,
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.rms_norm_eps,
                config.max_position_embeddings,
                vb_layers.pp(layer_idx),
                Some(lora_config),
                Some(lora_vb_layers.pp(layer_idx)),
            )?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("model").pp("norm"),
        )?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: config.clone(),
            device,
        })
    }

    /// Forward pass through the model, returning logits for all positions.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> CandleResult<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embed input tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through each transformer layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, index_pos, layer_idx, cache)?;
        }

        // Final layer norm
        // CRITICAL: Use forward_diff() instead of forward() because the optimized rms_norm
        // kernel uses apply_op_no_bwd which doesn't track gradients!
        // forward_diff() falls back to the slower but gradient-tracking implementation.
        // For background, see the Candle repository discussions/issues on `apply_op_no_bwd`
        // and rms_norm gradient tracking in candle-core/candle-transformers.
        hidden_states = self.norm.forward_diff(&hidden_states)?;

        // Project to vocabulary logits for ALL positions
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }
}

impl Module for LoraLlama {
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        // Create cache using the model's actual config
        let mut cache = Cache::new(false, DType::F32, &self.config, &self.device)?;
        self.forward(input_ids, 0, &mut cache)
    }
}
