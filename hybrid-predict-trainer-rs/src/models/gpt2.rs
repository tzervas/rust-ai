//! GPT-2 model implementation for validation.
//!
//! This is a reference implementation of GPT-2 used to validate memory optimizations
//! on real transformer models. Follows the original GPT-2 architecture with pre-norm
//! and causal self-attention.

use burn::{
    module::Module,
    nn::{
        attention::{generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};
use serde::{Deserialize, Serialize};

/// GPT-2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt2Config {
    /// Vocabulary size (default: 50257 for GPT-2)
    pub vocab_size: usize,
    /// Maximum sequence length (default: 1024)
    pub n_positions: usize,
    /// Embedding dimension (hidden size)
    pub n_embd: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Gpt2Config {
    /// GPT-2 Small configuration (124M parameters with weight tying).
    #[must_use]
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            dropout: 0.1,
        }
    }

    /// GPT-2 Medium configuration (350M parameters).
    #[must_use]
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            dropout: 0.1,
        }
    }

    /// GPT-2 XL configuration (~1B parameters).
    #[must_use]
    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            dropout: 0.1,
        }
    }
}

/// GPT-2 MLP (feedforward) block.
///
/// Implements: Linear(d, 4d) -> GELU -> Linear(4d, d) -> Dropout
#[derive(Module, Debug)]
pub struct Gpt2Mlp<B: Backend> {
    c_fc: Linear<B>,   // Expand: [n_embd, 4*n_embd]
    c_proj: Linear<B>, // Contract: [4*n_embd, n_embd]
    gelu: Gelu,
    dropout: Dropout,
}

impl<B: Backend> Gpt2Mlp<B> {
    /// Creates a new GPT-2 MLP block.
    pub fn new(n_embd: usize, dropout: f64, device: &B::Device) -> Self {
        let expansion_size = 4 * n_embd;

        Self {
            c_fc: LinearConfig::new(n_embd, expansion_size).init(device),
            c_proj: LinearConfig::new(expansion_size, n_embd).init(device),
            gelu: Gelu::new(),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// Forward pass through the MLP.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.gelu.forward(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

/// Causal self-attention using Burn's MultiHeadAttention.
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    mha: MultiHeadAttention<B>,
    resid_dropout: Dropout,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Creates a new causal self-attention block.
    pub fn new(n_embd: usize, n_head: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            mha: MultiHeadAttentionConfig::new(n_embd, n_head).init(device),
            resid_dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// Forward pass with causal masking.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();

        // Generate causal mask (upper triangle is true = masked out)
        let mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);

        // Self-attention with causal mask
        let input = MhaInput::self_attn(x).mask_attn(mask);
        let output = self.mha.forward(input);

        self.resid_dropout.forward(output.context)
    }
}

/// GPT-2 transformer block with pre-norm architecture.
///
/// Structure: x + Attention(LN(x)) -> x + MLP(LN(x))
#[derive(Module, Debug)]
pub struct Gpt2Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: Gpt2Mlp<B>,
}

impl<B: Backend> Gpt2Block<B> {
    /// Creates a new GPT-2 transformer block.
    pub fn new(n_embd: usize, n_head: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            ln_1: LayerNormConfig::new(n_embd).init(device),
            attn: CausalSelfAttention::new(n_embd, n_head, dropout, device),
            ln_2: LayerNormConfig::new(n_embd).init(device),
            mlp: Gpt2Mlp::new(n_embd, dropout, device),
        }
    }

    /// Forward pass with pre-norm and residual connections.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Attention block with residual
        let residual = x.clone();
        let x = self.ln_1.forward(x);
        let x = self.attn.forward(x);
        let x = residual + x;

        // MLP block with residual
        let residual = x.clone();
        let x = self.ln_2.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

/// GPT-2 model.
///
/// Implements the decoder-only transformer architecture with causal self-attention.
/// Uses weight tying between token embeddings and output projection.
#[derive(Module, Debug)]
pub struct Gpt2Model<B: Backend> {
    wte: Embedding<B>,        // Token embeddings
    wpe: Embedding<B>,        // Position embeddings
    drop: Dropout,
    h: Vec<Gpt2Block<B>>,    // Transformer blocks
    ln_f: LayerNorm<B>,       // Final layer norm
}

impl<B: Backend> Gpt2Model<B> {
    /// Creates a new GPT-2 model.
    pub fn new(config: &Gpt2Config, device: &B::Device) -> Self {
        // Create embeddings
        let wte = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let wpe = EmbeddingConfig::new(config.n_positions, config.n_embd).init(device);
        let drop = DropoutConfig::new(config.dropout).init();

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Gpt2Block::new(
                config.n_embd,
                config.n_head,
                config.dropout,
                device,
            ));
        }

        let ln_f = LayerNormConfig::new(config.n_embd).init(device);

        Self {
            wte,
            wpe,
            drop,
            h: blocks,
            ln_f,
        }
    }

    /// Forward pass through the model.
    ///
    /// Returns logits over vocabulary: [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Generate position IDs [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);

        // Embeddings: token + position
        let tok_emb = self.wte.forward(input_ids);
        let pos_emb = self.wpe.forward(positions);
        let mut x = self.drop.forward(tok_emb + pos_emb);

        // Transformer blocks
        for block in &self.h {
            x = block.forward(x);
        }

        // Final norm
        x = self.ln_f.forward(x);

        // Weight-tied language model head (use wte weights transposed)
        // [B, S, n_embd] @ [n_embd, vocab_size] = [B, S, vocab_size]
        // Note: wte.weight is [vocab_size, n_embd], we need [n_embd, vocab_size]
        let wte_weight = self.wte.weight.val().transpose();

        // Reshape x to [B*S, n_embd] for matmul, then reshape back
        let [batch_size, seq_len, n_embd] = x.dims();
        let x_flat = x.reshape([batch_size * seq_len, n_embd]);
        let logits_flat = x_flat.matmul(wte_weight);
        logits_flat.reshape([batch_size, seq_len, self.wte.weight.val().dims()[0]])
    }
}

/// Batch type for GPT-2 training.
#[derive(Debug, Clone)]
pub struct Gpt2Batch<B: Backend> {
    /// Input token IDs: [batch_size, seq_len]
    pub input_ids: Tensor<B, 2, Int>,
    /// Target token IDs: [batch_size, seq_len]
    /// In standard language modeling, targets are input_ids shifted right by 1
    pub targets: Tensor<B, 2, Int>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_gpt2_config() {
        let config = Gpt2Config::gpt2_small();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_positions, 1024);
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
    }

    #[test]
    fn test_gpt2_model_forward_shape() {
        let device = Default::default();
        let config = Gpt2Config::gpt2_small();
        let model = Gpt2Model::<TestBackend>::new(&config, &device);

        // Create dummy input: [2, 10] (batch=2, seq_len=10)
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);

        // Forward pass
        let logits = model.forward(input_ids);

        // Check output shape: [2, 10, 50257]
        assert_eq!(logits.dims(), [2, 10, 50257]);
    }

    #[test]
    fn test_gpt2_mlp_forward() {
        let device = Default::default();
        let mlp = Gpt2Mlp::<TestBackend>::new(768, 0.1, &device);

        // Input: [2, 10, 768]
        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 768], &device);
        let output = mlp.forward(x);

        // Output should be same shape
        assert_eq!(output.dims(), [2, 10, 768]);
    }

    #[test]
    fn test_causal_self_attention_forward() {
        let device = Default::default();
        let attn = CausalSelfAttention::<TestBackend>::new(768, 12, 0.1, &device);

        // Input: [2, 10, 768]
        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 768], &device);
        let output = attn.forward(x);

        // Output should be same shape
        assert_eq!(output.dims(), [2, 10, 768]);
    }

    #[test]
    fn test_gpt2_block_forward() {
        let device = Default::default();
        let block = Gpt2Block::<TestBackend>::new(768, 12, 0.1, &device);

        // Input: [2, 10, 768]
        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 768], &device);
        let output = block.forward(x);

        // Output should be same shape
        assert_eq!(output.dims(), [2, 10, 768]);
    }
}
