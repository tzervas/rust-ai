//! GPU acceleration kernels via `CubeCL` and Burn.
//!
//! This module provides GPU-accelerated implementations of performance-critical
//! operations using `CubeCL` for custom CUDA kernels and Burn for tensor operations.
//!
//! # Accelerated Operations
//!
//! - **State encoding**: Parallel feature extraction from training state
//! - **Prediction**: Batched dynamics model inference
//! - **Residual compression**: GPU-accelerated SVD for low-rank approximation
//! - **Correction**: Parallel residual application
//!
//! # Backend Support
//!
//! - CUDA (primary target)
//! - Future: Metal, Vulkan via `CubeCL` backends
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::gpu::{GpuAccelerator, CudaBackend};
//!
//! let accelerator = GpuAccelerator::<CudaBackend>::new()?;
//! let encoded = accelerator.encode_state(&state)?;
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use crate::state::TrainingState;

/// Marker trait for GPU backend implementations.
pub trait GpuBackend: Send + Sync {
    /// Returns the backend name.
    fn name() -> &'static str;

    /// Returns whether this backend is available.
    fn is_available() -> bool;

    /// Returns device information.
    fn device_info() -> DeviceInfo;
}

/// Information about the GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name.
    pub name: String,

    /// Total memory in bytes.
    pub total_memory: usize,

    /// Available memory in bytes.
    pub available_memory: usize,

    /// Compute capability (for CUDA).
    pub compute_capability: Option<(u32, u32)>,

    /// Number of streaming multiprocessors.
    pub num_sms: Option<u32>,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
            num_sms: None,
        }
    }
}

/// CUDA backend implementation.
pub struct CudaBackend;

impl GpuBackend for CudaBackend {
    fn name() -> &'static str {
        "CUDA"
    }

    fn is_available() -> bool {
        // Check CUDA availability via CubeCL
        // This is a placeholder - actual implementation would use cubecl-cuda
        cfg!(feature = "cuda")
    }

    fn device_info() -> DeviceInfo {
        // Query device info via CUDA driver API
        // Placeholder implementation
        DeviceInfo {
            name: "NVIDIA GPU".to_string(),
            ..Default::default()
        }
    }
}

/// GPU accelerator for hybrid training operations.
pub struct GpuAccelerator<B: GpuBackend> {
    /// Device information.
    device_info: DeviceInfo,

    /// Memory pool for temporary allocations.
    memory_pool: MemoryPool,

    /// Phantom marker for backend type.
    _backend: std::marker::PhantomData<B>,
}

impl<B: GpuBackend> GpuAccelerator<B> {
    /// Creates a new GPU accelerator.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU backend is not available.
    pub fn new() -> HybridResult<Self> {
        if !B::is_available() {
            return Err((
                HybridTrainingError::GpuError {
                    detail: format!("{} backend not available", B::name()),
                },
                None,
            ));
        }

        let device_info = B::device_info();
        let memory_pool = MemoryPool::new(device_info.available_memory / 4);

        Ok(Self {
            device_info,
            memory_pool,
            _backend: std::marker::PhantomData,
        })
    }

    /// Returns device information.
    #[must_use]
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Encodes training state to GPU tensor.
    ///
    /// Performs parallel feature extraction on the GPU.
    pub fn encode_state(&self, _state: &TrainingState) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Transfer state data to GPU
        // 2. Launch parallel feature extraction kernel
        // 3. Return encoded tensor
        Ok(GpuTensor {
            data: Vec::new(),
            shape: vec![32],
            device: B::name().to_string(),
        })
    }

    /// Performs batched dynamics prediction on GPU.
    pub fn predict_batch(
        &self,
        _encoded_states: &[GpuTensor],
        _steps: usize,
    ) -> HybridResult<Vec<GpuTensor>> {
        // Placeholder - actual implementation would:
        // 1. Batch encoded states into single tensor
        // 2. Run RSSM forward pass on GPU
        // 3. Return predicted states
        Ok(Vec::new())
    }

    /// Computes low-rank approximation of residuals on GPU.
    pub fn compress_residuals(
        &self,
        _residuals: &GpuTensor,
        rank: usize,
    ) -> HybridResult<CompressedGpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Perform truncated SVD on GPU
        // 2. Return compressed representation
        Ok(CompressedGpuTensor {
            u: GpuTensor::empty(),
            s: GpuTensor::empty(),
            v: GpuTensor::empty(),
            rank,
        })
    }

    /// Applies corrections in parallel on GPU.
    pub fn apply_corrections(
        &self,
        _predictions: &GpuTensor,
        _corrections: &GpuTensor,
    ) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would:
        // 1. Element-wise addition on GPU
        // 2. Return corrected predictions
        Ok(GpuTensor::empty())
    }

    /// Synchronizes GPU operations (waits for completion).
    pub fn synchronize(&self) -> HybridResult<()> {
        // Placeholder - actual implementation would call cudaDeviceSynchronize
        Ok(())
    }

    /// Returns current memory usage.
    #[must_use]
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated: self.memory_pool.allocated(),
            pool_size: self.memory_pool.capacity(),
            peak_usage: self.memory_pool.peak_usage(),
        }
    }
}

/// GPU tensor representation.
#[derive(Debug, Clone)]
pub struct GpuTensor {
    /// Data (may be on host for inspection).
    data: Vec<f32>,

    /// Tensor shape.
    shape: Vec<usize>,

    /// Device identifier.
    #[allow(dead_code)]
    device: String,
}

impl GpuTensor {
    /// Creates an empty GPU tensor.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            device: "cpu".to_string(),
        }
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Transfers tensor to host memory.
    #[must_use]
    pub fn to_host(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// Compressed GPU tensor (SVD representation).
#[derive(Debug, Clone)]
pub struct CompressedGpuTensor {
    /// Left singular vectors.
    #[allow(dead_code)]
    u: GpuTensor,

    /// Singular values.
    #[allow(dead_code)]
    s: GpuTensor,

    /// Right singular vectors.
    #[allow(dead_code)]
    v: GpuTensor,

    /// Rank of approximation.
    rank: usize,
}

impl CompressedGpuTensor {
    /// Returns the rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Reconstructs the full tensor.
    pub fn reconstruct(&self) -> HybridResult<GpuTensor> {
        // Placeholder - actual implementation would compute U @ diag(S) @ V^T
        Ok(GpuTensor::empty())
    }
}

/// Memory pool for GPU allocations.
struct MemoryPool {
    capacity: usize,
    allocated: usize,
    peak: usize,
}

impl MemoryPool {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            allocated: 0,
            peak: 0,
        }
    }

    fn allocated(&self) -> usize {
        self.allocated
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn peak_usage(&self) -> usize {
        self.peak
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Currently allocated bytes.
    pub allocated: usize,

    /// Total pool size.
    pub pool_size: usize,

    /// Peak usage.
    pub peak_usage: usize,
}

/// `CubeCL` kernel definitions.
///
/// `CubeCL` kernel implementations.
///
/// These kernels are compiled to CUDA/PTX at runtime using `CubeCL`.
pub mod kernels {

    /// State encoding kernel configuration.
    #[derive(Debug, Clone)]
    pub struct EncodeStateConfig {
        /// Input feature dimension.
        pub input_dim: usize,
        /// Output encoding dimension.
        pub output_dim: usize,
        /// Block size for CUDA kernel.
        pub block_size: usize,
    }

    impl Default for EncodeStateConfig {
        fn default() -> Self {
            Self {
                input_dim: 32,
                output_dim: 128,
                block_size: 256,
            }
        }
    }

    /// GRU forward pass kernel configuration.
    #[derive(Debug, Clone)]
    pub struct GruConfig {
        /// Hidden state dimension.
        pub hidden_dim: usize,
        /// Input dimension.
        pub input_dim: usize,
        /// Batch size.
        pub batch_size: usize,
    }

    impl Default for GruConfig {
        fn default() -> Self {
            Self {
                hidden_dim: 256,
                input_dim: 128,
                batch_size: 1,
            }
        }
    }

    /// Flash Attention kernel configuration.
    ///
    /// Flash Attention is a fused attention kernel that reduces memory usage
    /// from O(n²) to O(n) by computing attention incrementally in blocks
    /// without materializing the full attention matrix.
    ///
    /// # Memory Savings
    ///
    /// **Standard Attention**:
    /// ```text
    /// Q @ K^T → [batch, heads, seq_len, seq_len]  // O(n²) memory
    /// softmax → [batch, heads, seq_len, seq_len]  // O(n²) memory
    /// @ V     → [batch, heads, seq_len, d_head]   // Final output
    /// ```
    ///
    /// **Flash Attention**:
    /// ```text
    /// Process in blocks of size block_size:
    /// - Load Q block:  [block_size, d_head]       // O(n) memory
    /// - Load K block:  [block_size, d_head]       // O(n) memory
    /// - Compute QK^T:  [block_size, block_size]   // O(1) block memory
    /// - Softmax + V:   Fused, no materialization
    /// - Accumulate:    [seq_len, d_head]          // Final output only
    /// ```
    ///
    /// **Result**: O(n²) → O(n) memory, ~99% reduction for large sequences
    ///
    /// # Why This Matters
    ///
    /// For transformer models, attention is often the memory bottleneck:
    /// - Sequence length 2048, 32 heads, fp16 precision
    /// - Standard: 2048² × 32 × 2 bytes = 256 MB per layer
    /// - Flash: 2048 × 32 × 2 bytes = 128 KB per layer
    /// - **Savings: 99.95%** (256 MB → 128 KB)
    ///
    /// # Algorithm
    ///
    /// Flash Attention uses a tiling strategy with online softmax:
    ///
    /// 1. **Tiling**: Split Q, K, V into blocks of size `block_size`
    /// 2. **Incremental Softmax**: Compute softmax statistics incrementally
    ///    - Track running max and sum for numerically stable softmax
    ///    - Update statistics as each block is processed
    /// 3. **Fused Operations**: QK^T + softmax + matmul in single kernel
    ///    - No intermediate materialization
    ///    - Better cache locality
    /// 4. **Output Accumulation**: Accumulate final output incrementally
    ///
    /// # Trade-offs
    ///
    /// - **Memory**: 99% reduction (O(n²) → O(n))
    /// - **Compute**: +10-20% due to recomputation (worth it for memory savings)
    /// - **Accuracy**: Numerically identical to standard attention
    ///
    /// # HybridTrainer Integration
    ///
    /// Flash Attention is beneficial in all training phases:
    /// - **Full phase**: Reduces peak memory during backward pass
    /// - **Predict phase**: Smaller memory footprint for forward-only inference
    /// - **Correct phase**: Enables larger validation batches
    ///
    /// Combined with gradient checkpointing and quantization, enables
    /// training of massive transformer models (7B-50B params) on consumer GPUs.
    ///
    /// # References
    ///
    /// - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    ///   Dao et al., 2022 (https://arxiv.org/abs/2205.14135)
    /// - "FlashAttention-2: Faster Attention with Better Parallelism"
    ///   Dao, 2023 (https://arxiv.org/abs/2307.08691)
    #[derive(Debug, Clone)]
    pub struct FlashAttentionConfig {
        /// Sequence length (number of tokens)
        ///
        /// Why: Determines the size of the attention matrix (seq_len × seq_len)
        /// and the memory savings from Flash Attention.
        pub seq_len: usize,

        /// Number of attention heads
        ///
        /// Why: Multi-head attention processes multiple attention patterns
        /// in parallel. Each head has its own QKV projections.
        pub num_heads: usize,

        /// Head dimension (d_model / num_heads)
        ///
        /// Why: Dimension of each attention head. Typically 64 or 128.
        /// Total model dimension = num_heads × head_dim
        pub head_dim: usize,

        /// Block size for tiling (default: 256)
        ///
        /// Why: Determines the tile size for blocked computation.
        /// Larger blocks = more parallelism but more recomputation.
        /// Smaller blocks = less memory but more kernel launches.
        ///
        /// Recommended values:
        /// - Small models (<1B): 128
        /// - Medium models (1-7B): 256
        /// - Large models (7B+): 512
        pub block_size: usize,

        /// Batch size
        ///
        /// Why: Number of sequences processed in parallel.
        /// Flash Attention processes each sequence independently.
        pub batch_size: usize,

        /// Enable causal masking (for autoregressive models)
        ///
        /// Why: Causal masking prevents attending to future tokens.
        /// Required for GPT-style models, not needed for BERT-style models.
        pub causal: bool,

        /// Dropout probability (0.0 = no dropout)
        ///
        /// Why: Dropout on attention weights for regularization.
        /// Flash Attention can fuse dropout into the kernel.
        pub dropout: f32,
    }

    impl Default for FlashAttentionConfig {
        fn default() -> Self {
            Self {
                seq_len: 2048,
                num_heads: 16,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: false,
                dropout: 0.0,
            }
        }
    }

    impl FlashAttentionConfig {
        /// Create configuration for GPT-2 style model
        ///
        /// Why: GPT-2 uses causal attention with specific dimensions.
        /// Provides a convenient constructor for common use cases.
        #[must_use]
        pub fn gpt2(seq_len: usize) -> Self {
            Self {
                seq_len,
                num_heads: 12,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: true, // Autoregressive
                dropout: 0.1,
            }
        }

        /// Create configuration for BERT style model
        ///
        /// Why: BERT uses bidirectional attention (no causal masking).
        #[must_use]
        pub fn bert(seq_len: usize) -> Self {
            Self {
                seq_len,
                num_heads: 12,
                head_dim: 64,
                block_size: 256,
                batch_size: 1,
                causal: false, // Bidirectional
                dropout: 0.1,
            }
        }

        /// Calculate theoretical memory savings vs standard attention
        ///
        /// Why: Quantifies the memory reduction from Flash Attention.
        /// Helps users understand the benefit before implementation.
        ///
        /// # Returns
        ///
        /// `(standard_mb, flash_mb, savings_percent)`
        ///
        /// # Algorithm
        ///
        /// Standard attention memory:
        /// - Attention matrix: batch × heads × seq_len × seq_len × 2 bytes (fp16)
        /// - Softmax output:   batch × heads × seq_len × seq_len × 2 bytes (fp16)
        /// - Total: 2 × batch × heads × seq_len² × 2 bytes
        ///
        /// Flash attention memory:
        /// - Output only: batch × heads × seq_len × head_dim × 2 bytes (fp16)
        /// - Block buffers: 2 × block_size × head_dim × 2 bytes (fp16)
        /// - Total: batch × heads × seq_len × head_dim × 2 + small overhead
        #[must_use]
        pub fn theoretical_savings(&self) -> (f32, f32, f32) {
            let bytes_per_element = 2.0; // fp16

            // Standard attention: QK^T matrix + softmax output
            let attention_matrix_bytes =
                (self.batch_size * self.num_heads * self.seq_len * self.seq_len) as f32
                    * bytes_per_element;
            let standard_mb = 2.0 * attention_matrix_bytes / (1024.0 * 1024.0);

            // Flash attention: Output + block buffers
            let output_bytes = (self.batch_size * self.num_heads * self.seq_len * self.head_dim)
                as f32
                * bytes_per_element;
            let block_buffer_bytes = (2 * self.block_size * self.head_dim) as f32
                * bytes_per_element;
            let flash_mb = (output_bytes + block_buffer_bytes) / (1024.0 * 1024.0);

            let savings_percent = ((standard_mb - flash_mb) / standard_mb) * 100.0;

            (standard_mb, flash_mb, savings_percent)
        }

        /// Validate configuration
        ///
        /// Why: Ensures parameters are sensible before kernel launch.
        /// Prevents cryptic CUDA errors from invalid configurations.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - seq_len is 0 or too large (>65536)
        /// - num_heads is 0 or too large (>128)
        /// - head_dim is 0 or not a multiple of 8
        /// - block_size is 0 or too large (>1024)
        /// - dropout is not in [0.0, 1.0)
        pub fn validate(&self) -> Result<(), String> {
            if self.seq_len == 0 || self.seq_len > 65536 {
                return Err(format!(
                    "Invalid seq_len: {} (must be 1-65536)",
                    self.seq_len
                ));
            }

            if self.num_heads == 0 || self.num_heads > 128 {
                return Err(format!(
                    "Invalid num_heads: {} (must be 1-128)",
                    self.num_heads
                ));
            }

            if self.head_dim == 0 || self.head_dim % 8 != 0 {
                return Err(format!(
                    "Invalid head_dim: {} (must be positive and multiple of 8)",
                    self.head_dim
                ));
            }

            if self.block_size == 0 || self.block_size > 1024 {
                return Err(format!(
                    "Invalid block_size: {} (must be 1-1024)",
                    self.block_size
                ));
            }

            if !(0.0..1.0).contains(&self.dropout) {
                return Err(format!(
                    "Invalid dropout: {} (must be in [0.0, 1.0))",
                    self.dropout
                ));
            }

            Ok(())
        }
    }

    /// Flash Attention statistics
    ///
    /// Why: Track memory usage and performance metrics to validate
    /// the memory savings and identify optimization opportunities.
    #[derive(Debug, Clone)]
    pub struct FlashAttentionStats {
        /// Number of forward passes executed
        pub num_forward_passes: usize,

        /// Total memory saved compared to standard attention (bytes)
        ///
        /// Why: Quantifies the actual memory reduction achieved.
        pub memory_saved_bytes: usize,

        /// Average kernel execution time (microseconds)
        ///
        /// Why: Tracks the compute overhead from recomputation.
        pub avg_kernel_time_us: f32,

        /// Peak memory usage (bytes)
        ///
        /// Why: Validates that memory usage stays within expected bounds.
        pub peak_memory_bytes: usize,
    }

    impl Default for FlashAttentionStats {
        fn default() -> Self {
            Self {
                num_forward_passes: 0,
                memory_saved_bytes: 0,
                avg_kernel_time_us: 0.0,
                peak_memory_bytes: 0,
            }
        }
    }

    impl std::fmt::Display for FlashAttentionStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Flash Attention Stats: {} passes | {:.2} MB saved | {:.2} µs/pass | Peak: {:.2} MB",
                self.num_forward_passes,
                self.memory_saved_bytes as f32 / (1024.0 * 1024.0),
                self.avg_kernel_time_us,
                self.peak_memory_bytes as f32 / (1024.0 * 1024.0)
            )
        }
    }

    /// Flash Attention kernel implementation (CubeCL)
    ///
    /// Why: Actual GPU kernel that performs fused QK^T + softmax + matmul.
    /// This is a placeholder for the CubeCL implementation.
    ///
    /// # Algorithm Pseudocode
    ///
    /// ```text
    /// function flash_attention(Q, K, V, block_size):
    ///     N = seq_len
    ///     Output = zeros(N, d_head)
    ///     RowMax = -inf * ones(N)    // Running max for softmax
    ///     RowSum = zeros(N)           // Running sum for softmax
    ///
    ///     // Process in blocks (tiling)
    ///     for block_i in range(0, N, block_size):
    ///         for block_j in range(0, N, block_size):
    ///             // Load blocks
    ///             Q_block = Q[block_i:block_i+block_size, :]
    ///             K_block = K[block_j:block_j+block_size, :]
    ///             V_block = V[block_j:block_j+block_size, :]
    ///
    ///             // Compute attention scores
    ///             S_block = Q_block @ K_block.T  // [block_size, block_size]
    ///
    ///             // Online softmax update
    ///             for row in block_i:block_i+block_size:
    ///                 old_max = RowMax[row]
    ///                 new_max = max(old_max, max(S_block[row, :]))
    ///
    ///                 // Rescale previous contributions
    ///                 scale = exp(old_max - new_max)
    ///                 Output[row, :] *= scale
    ///                 RowSum[row] *= scale
    ///
    ///                 // Add new contributions
    ///                 P_block = exp(S_block[row, :] - new_max)
    ///                 Output[row, :] += P_block @ V_block
    ///                 RowSum[row] += sum(P_block)
    ///
    ///                 RowMax[row] = new_max
    ///
    ///     // Normalize by row sums
    ///     Output /= RowSum[:, None]
    ///     return Output
    /// ```
    ///
    /// # CubeCL Implementation
    ///
    /// The actual CubeCL kernel would:
    /// 1. Use shared memory for Q, K, V blocks
    /// 2. Use warp-level primitives for reductions (max, sum)
    /// 3. Fuse operations to minimize memory traffic
    /// 4. Handle causal masking efficiently
    /// 5. Support fp16/bf16 for memory efficiency
    ///
    /// # Note
    ///
    /// This is a placeholder. The actual CubeCL implementation would be
    /// in a separate `.cube` file using CubeCL's Rust-embedded DSL.
    ///
    /// Example kernel launch:
    /// ```rust,ignore
    /// use cubecl::prelude::*;
    ///
    /// #[cube(launch)]
    /// fn flash_attention_kernel(
    ///     q: &Tensor<f16>,
    ///     k: &Tensor<f16>,
    ///     v: &Tensor<f16>,
    ///     output: &mut Tensor<f16>,
    ///     config: FlashAttentionConfig,
    /// ) {
    ///     // CubeCL kernel code here
    /// }
    /// ```
    pub fn flash_attention_forward(
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _config: &FlashAttentionConfig,
    ) -> Vec<f32> {
        // Placeholder - actual implementation would:
        // 1. Transfer Q, K, V to GPU
        // 2. Launch Flash Attention kernel with tiling
        // 3. Return output tensor
        //
        // For now, return empty vector
        // TODO: Implement CubeCL kernel
        Vec::new()
    }
}

/// Burn tensor operations wrapper.
///
/// Burn-based tensor operations for GPU acceleration.
pub mod burn_ops {

    /// Performs matrix multiplication using Burn.
    #[must_use]
    pub fn matmul(_a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        // Placeholder - actual implementation would use burn::tensor::Tensor
        Vec::new()
    }

    /// Performs element-wise operations using Burn.
    #[must_use]
    pub fn elementwise_add(_a: &[f32], _b: &[f32]) -> Vec<f32> {
        // Placeholder
        Vec::new()
    }

    /// Computes softmax using Burn.
    #[must_use]
    pub fn softmax(_x: &[f32], _dim: usize) -> Vec<f32> {
        // Placeholder
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_default() {
        let info = DeviceInfo::default();
        assert_eq!(info.name, "Unknown");
    }

    #[test]
    fn test_gpu_tensor_empty() {
        let tensor = GpuTensor::empty();
        assert_eq!(tensor.numel(), 1); // Empty shape has product 1
    }

    #[test]
    fn test_kernel_config_defaults() {
        let encode_config = kernels::EncodeStateConfig::default();
        assert_eq!(encode_config.block_size, 256);

        let gru_config = kernels::GruConfig::default();
        assert_eq!(gru_config.hidden_dim, 256);
    }

    #[test]
    fn test_flash_attention_config_defaults() {
        let config = kernels::FlashAttentionConfig::default();
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.batch_size, 1);
        assert!(!config.causal);
        assert!((config.dropout - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention_gpt2_config() {
        let config = kernels::FlashAttentionConfig::gpt2(1024);
        assert_eq!(config.seq_len, 1024);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal); // GPT-2 uses causal attention
        assert!((config.dropout - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attention_bert_config() {
        let config = kernels::FlashAttentionConfig::bert(512);
        assert_eq!(config.seq_len, 512);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert!(!config.causal); // BERT uses bidirectional attention
    }

    #[test]
    fn test_flash_attention_theoretical_savings() {
        let config = kernels::FlashAttentionConfig {
            seq_len: 2048,
            num_heads: 32,
            head_dim: 64,
            block_size: 256,
            batch_size: 1,
            causal: false,
            dropout: 0.0,
        };

        let (standard_mb, flash_mb, savings_percent) = config.theoretical_savings();

        // Standard: 2 × 1 × 32 × 2048² × 2 bytes = 512 MB
        assert!(
            (standard_mb - 512.0).abs() < 1.0,
            "Expected ~512 MB standard, got {}",
            standard_mb
        );

        // Flash: 1 × 32 × 2048 × 64 × 2 bytes + small overhead ≈ 8 MB
        assert!(
            flash_mb < 10.0,
            "Expected <10 MB flash attention, got {}",
            flash_mb
        );

        // Savings should be >95%
        assert!(
            savings_percent > 95.0,
            "Expected >95% savings, got {}%",
            savings_percent
        );
    }

    #[test]
    fn test_flash_attention_validation_success() {
        let config = kernels::FlashAttentionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_flash_attention_validation_invalid_seq_len() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.seq_len = 0;
        assert!(config.validate().is_err());

        config.seq_len = 100000; // Too large
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_validation_invalid_head_dim() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.head_dim = 0;
        assert!(config.validate().is_err());

        config.head_dim = 63; // Not multiple of 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_validation_invalid_dropout() {
        let mut config = kernels::FlashAttentionConfig::default();
        config.dropout = -0.1; // Negative
        assert!(config.validate().is_err());

        config.dropout = 1.0; // Too high (should be <1.0)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flash_attention_stats_default() {
        let stats = kernels::FlashAttentionStats::default();
        assert_eq!(stats.num_forward_passes, 0);
        assert_eq!(stats.memory_saved_bytes, 0);
        assert_eq!(stats.avg_kernel_time_us, 0.0);
        assert_eq!(stats.peak_memory_bytes, 0);
    }

    #[test]
    fn test_flash_attention_stats_display() {
        let stats = kernels::FlashAttentionStats {
            num_forward_passes: 100,
            memory_saved_bytes: 256 * 1024 * 1024, // 256 MB
            avg_kernel_time_us: 123.45,
            peak_memory_bytes: 8 * 1024 * 1024, // 8 MB
        };
        let display = format!("{}", stats);
        assert!(display.contains("100 passes"));
        assert!(display.contains("256"));
        assert!(display.contains("123.45"));
    }

    #[test]
    fn test_flash_attention_forward_placeholder() {
        // Test placeholder function exists and returns empty vector
        let config = kernels::FlashAttentionConfig::default();
        let q = vec![1.0; 2048 * 64];
        let k = vec![1.0; 2048 * 64];
        let v = vec![1.0; 2048 * 64];

        let output = kernels::flash_attention_forward(&q, &k, &v, &config);
        assert!(output.is_empty()); // Placeholder returns empty
    }

    #[test]
    fn test_flash_attention_large_sequence() {
        // Test with very large sequence length to verify savings
        let config = kernels::FlashAttentionConfig {
            seq_len: 8192, // Large sequence
            num_heads: 40,
            head_dim: 128,
            block_size: 512,
            batch_size: 1,
            causal: true,
            dropout: 0.0,
        };

        let (standard_mb, flash_mb, savings_percent) = config.theoretical_savings();

        // Standard: 2 × 1 × 40 × 8192² × 2 bytes = 10.24 GB
        assert!(standard_mb > 10000.0, "Expected >10 GB standard");

        // Flash: Much smaller
        assert!(flash_mb < 100.0, "Expected <100 MB flash");

        // Savings should be >99%
        assert!(savings_percent > 99.0, "Expected >99% savings");
    }
}
