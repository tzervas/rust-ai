//! Multi-head attention with QK-Norm and optional BitNet quantization.
//!
//! Implements the attention mechanism with:
//! - QK-Norm for query-key normalization (per Chameleon/BitNet papers)
//! - Rotary Position Embeddings (RoPE)
//! - Grouped Query Attention (GQA) for memory efficiency
//! - BitNet ternary weight quantization

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::bitnet::TritterLinear;
use crate::config::TritterConfig;

/// Rotary position embedding cache
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    /// Create RoPE cache for given max sequence length
    pub fn new(config: &TritterConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let theta = config.rope_theta;

        // Compute inverse frequencies: 1 / (theta^(2i/dim))
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        // Position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?;

        // Compute angles: outer product of positions and inv_freq
        let angles = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let angles = Tensor::cat(&[&angles, &angles], D::Minus1)?;

        let cos = angles.cos()?;
        let sin = angles.sin()?;

        Ok(Self { cos, sin, dim })
    }

    /// Apply RoPE to query and key tensors
    /// Input shape: (batch, seq_len, num_heads, head_dim)
    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.i(start_pos..start_pos + seq_len)?;
        let sin = self.sin.i(start_pos..start_pos + seq_len)?;

        // Reshape for broadcasting: (seq_len, 1, head_dim)
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        let q_rot = rotate_half(q, &cos, &sin)?;
        let k_rot = rotate_half(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

/// Rotate half of the tensor for RoPE
fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    // Combine: [x1, x2] * cos + [-x2, x1] * sin
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)?.broadcast_add(&rotated.broadcast_mul(sin)?)
}

/// QK-Norm layer for attention stability
pub struct QKNorm {
    weight: Tensor,
    eps: f64,
}

impl QKNorm {
    /// Create QK-Norm with learnable scale
    pub fn new(head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(head_dim, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Apply normalization per head: x * weight / rms(x)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: (batch, seq, num_heads, head_dim)
        // Compute RMS norm over head_dim
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_norm.broadcast_mul(&self.weight)
    }
}

/// Multi-head attention with QK-Norm and optional GQA
pub struct TritterAttention {
    q_proj: TritterLinear,
    k_proj: TritterLinear,
    v_proj: TritterLinear,
    o_proj: TritterLinear,
    q_norm: Option<QKNorm>,
    k_norm: Option<QKNorm>,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dropout: f32,
    use_bitnet: bool,
}

impl TritterAttention {
    /// Create attention module
    ///
    /// If `config.use_bitnet` is true, the projection layers will use
    /// BitNet ternary quantization for ~16x memory reduction.
    pub fn new(config: &TritterConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.kv_heads();
        let head_dim = config.head_dim();
        let use_bitnet = config.use_bitnet;

        // Projections (optionally BitNet quantized)
        let q_proj = TritterLinear::new(hidden, num_heads * head_dim, use_bitnet, vb.pp("q_proj"), device)?;
        let k_proj = TritterLinear::new(hidden, num_kv_heads * head_dim, use_bitnet, vb.pp("k_proj"), device)?;
        let v_proj = TritterLinear::new(hidden, num_kv_heads * head_dim, use_bitnet, vb.pp("v_proj"), device)?;
        let o_proj = TritterLinear::new(num_heads * head_dim, hidden, use_bitnet, vb.pp("o_proj"), device)?;

        // QK-Norm (optional)
        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(QKNorm::new(head_dim, config.layer_norm_eps, vb.pp("q_norm"))?),
                Some(QKNorm::new(head_dim, config.layer_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        // RoPE
        let rotary = RotaryEmbedding::new(config, config.max_seq_length, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            dropout: config.dropout,
            use_bitnet,
        })
    }

    /// Check if this attention layer is using BitNet quantization.
    #[must_use]
    pub const fn is_bitnet(&self) -> bool {
        self.use_bitnet
    }

    /// Get the average compression ratio across all projections.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        (self.q_proj.compression_ratio()
            + self.k_proj.compression_ratio()
            + self.v_proj.compression_ratio()
            + self.o_proj.compression_ratio())
            / 4.0
    }

    /// Forward pass
    /// Input: (batch, seq_len, hidden_size)
    /// Output: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim)
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK-Norm if enabled
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // Apply RoPE
        let (q, k) = self.rotary.apply(&q, &k, 0)?;

        // Transpose to (batch, num_heads, seq, head_dim) for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Handle GQA: repeat K, V for each query head group
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, n_rep)?;
            let v = repeat_kv(&v, n_rep)?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.t()?.contiguous()?;
        let attn = q.matmul(&k_t)?;
        let attn = (attn / scale)?;

        // Apply causal mask if provided
        let attn = if let Some(m) = mask {
            attn.broadcast_add(m)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Attention output
        let out = attn.matmul(&v)?;

        // Transpose back and reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden)
        let out = out.transpose(1, 2)?;
        let out = out.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&out)
    }
}

/// Repeat KV heads for GQA
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, s, d) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, n_kv, n_rep, s, d))?;
    x.reshape((b, n_kv * n_rep, s, d))
}

/// Create causal attention mask
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_rotary_embedding() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(&config, 64, &device).unwrap();

        let batch = 2;
        let seq = 16;
        let heads = config.num_heads;
        let dim = config.head_dim();

        let q = Tensor::randn(0.0f32, 1.0, (batch, seq, heads, dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch, seq, heads, dim), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device).unwrap();
        let mask_vec: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Check pattern: lower triangular is 0, upper is -inf
        assert_eq!(mask_vec[0], 0.0);  // (0,0)
        assert!(mask_vec[1].is_infinite()); // (0,1)
        assert_eq!(mask_vec[5], 0.0);  // (1,1)
    }

    #[test]
    fn test_attention_standard() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let attn = TritterAttention::new(&config, vb, &device).unwrap();
        assert!(!attn.is_bitnet());
        assert_eq!(attn.compression_ratio(), 1.0);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, config.hidden_size), &device).unwrap();
        let out = attn.forward(&x, None).unwrap();

        assert_eq!(out.dims(), &[2, 8, config.hidden_size]);
    }

    #[test]
    fn test_attention_bitnet() {
        let mut config = TritterConfig::test();
        config.use_bitnet = true;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let attn = TritterAttention::new(&config, vb, &device).unwrap();
        assert!(attn.is_bitnet());
        assert!(attn.compression_ratio() > 1.0);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, config.hidden_size), &device).unwrap();
        let out = attn.forward(&x, None).unwrap();

        assert_eq!(out.dims(), &[2, 8, config.hidden_size]);
    }
}
