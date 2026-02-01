//! Optimized training configurations for different use cases.
//!
//! These presets are designed based on best practices from modern LLM training.

use serde::{Deserialize, Serialize};

/// Training configuration preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPreset {
    /// Preset name
    pub name: String,
    /// Description
    pub description: String,
    /// Model configuration
    pub model: ModelPreset,
    /// Optimization configuration
    pub optimization: OptimizationPreset,
    /// Data configuration
    pub data: DataPreset,
    /// Hybrid training configuration
    pub hybrid: HybridPreset,
}

/// Model-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreset {
    /// Model size identifier
    pub size: String,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP intermediate size (typically 4x hidden)
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Checkpoint every N layers (if gradient_checkpointing enabled)
    pub checkpoint_every_n_layers: usize,
}

/// Optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPreset {
    /// Peak learning rate
    pub learning_rate: f32,
    /// Minimum learning rate (for decay)
    pub min_learning_rate: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Beta1 for AdamW
    pub beta1: f32,
    /// Beta2 for AdamW
    pub beta2: f32,
    /// Epsilon for AdamW
    pub epsilon: f32,
    /// Maximum gradient norm (for clipping)
    pub max_grad_norm: f32,
    /// Warmup steps (fraction of total)
    pub warmup_fraction: f32,
    /// Decay steps (fraction of total)
    pub decay_fraction: f32,
    /// Total training steps
    pub total_steps: u64,
}

/// Data loading configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreset {
    /// Micro batch size (per GPU)
    pub micro_batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Sequence length for training
    pub seq_length: usize,
    /// Number of data loader workers
    pub num_workers: usize,
    /// Shuffle data each epoch
    pub shuffle: bool,
}

impl DataPreset {
    /// Effective batch size = micro_batch_size * gradient_accumulation_steps
    pub fn effective_batch_size(&self) -> usize {
        self.micro_batch_size * self.gradient_accumulation_steps
    }

    /// Tokens per step
    pub fn tokens_per_step(&self) -> usize {
        self.effective_batch_size() * self.seq_length
    }
}

/// Hybrid predictive training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridPreset {
    /// Warmup steps before starting hybrid training
    pub warmup_steps: usize,
    /// Full backward steps per cycle
    pub full_steps_per_cycle: usize,
    /// Maximum predict steps per cycle
    pub max_predict_steps: usize,
    /// Confidence threshold for prediction
    pub confidence_threshold: f32,
    /// Divergence threshold for correction
    pub divergence_threshold: f32,
}

impl TrainingPreset {
    /// Create preset for research/scientific/coding assistant (100M model).
    ///
    /// Optimized for:
    /// - Long-form technical content understanding
    /// - Code comprehension and generation
    /// - Scientific/mathematical reasoning
    /// - Stability over speed
    pub fn research_assistant_100m() -> Self {
        Self {
            name: "research-assistant-100m".to_string(),
            description:
                "100M parameter model optimized for research, scientific, and coding assistance"
                    .to_string(),
            model: ModelPreset {
                size: "100m".to_string(),
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                intermediate_size: 3072, // 4x hidden
                vocab_size: 32000,
                max_seq_length: 2048, // Longer context for code/docs
                dropout: 0.05,        // Lower dropout for smaller model
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 3, // Every 3 layers
            },
            optimization: OptimizationPreset {
                learning_rate: 6e-5, // Lower for stability
                min_learning_rate: 6e-7,
                weight_decay: 0.1,
                beta1: 0.9,
                beta2: 0.95, // Lower beta2 for faster adaptation
                epsilon: 1e-8,
                max_grad_norm: 1.0,    // Gradient clipping
                warmup_fraction: 0.05, // 5% warmup
                decay_fraction: 0.20,  // 20% decay
                total_steps: 50_000,
            },
            data: DataPreset {
                micro_batch_size: 2,            // For 16GB GPU
                gradient_accumulation_steps: 8, // Effective batch = 16
                seq_length: 1024,               // Balance between context and memory
                num_workers: 4,
                shuffle: true,
            },
            hybrid: HybridPreset {
                warmup_steps: 500,        // More warmup for stability
                full_steps_per_cycle: 20, // More full steps before predicting
                max_predict_steps: 100,
                confidence_threshold: 0.90, // Higher threshold for accuracy
                divergence_threshold: 0.15,
            },
        }
    }

    /// Create preset for fast iteration/experimentation (100M model).
    ///
    /// Optimized for:
    /// - Quick feedback during development
    /// - Validating data pipeline
    /// - Architecture experiments
    pub fn fast_iteration_100m() -> Self {
        Self {
            name: "fast-iteration-100m".to_string(),
            description: "100M model for fast iteration and experimentation".to_string(),
            model: ModelPreset {
                size: "100m".to_string(),
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                intermediate_size: 3072,
                vocab_size: 32000,
                max_seq_length: 512, // Shorter for speed
                dropout: 0.1,
                gradient_checkpointing: false, // Faster without checkpointing
                checkpoint_every_n_layers: 4,
            },
            optimization: OptimizationPreset {
                learning_rate: 3e-4, // Higher for faster learning
                min_learning_rate: 3e-6,
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                max_grad_norm: 1.0,
                warmup_fraction: 0.01, // Quick warmup
                decay_fraction: 0.10,
                total_steps: 10_000,
            },
            data: DataPreset {
                micro_batch_size: 4,
                gradient_accumulation_steps: 2, // Effective batch = 8
                seq_length: 512,
                num_workers: 4,
                shuffle: true,
            },
            hybrid: HybridPreset {
                warmup_steps: 100,
                full_steps_per_cycle: 10,
                max_predict_steps: 150,
                confidence_threshold: 0.85,
                divergence_threshold: 0.20,
            },
        }
    }

    /// Create preset for production training (100M model).
    ///
    /// Optimized for:
    /// - Maximum quality
    /// - Long training runs
    /// - Best final performance
    pub fn production_100m() -> Self {
        Self {
            name: "production-100m".to_string(),
            description: "100M model production training for maximum quality".to_string(),
            model: ModelPreset {
                size: "100m".to_string(),
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                intermediate_size: 3072,
                vocab_size: 32000,
                max_seq_length: 2048,
                dropout: 0.0, // No dropout for final training
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 2,
            },
            optimization: OptimizationPreset {
                learning_rate: 3e-4,
                min_learning_rate: 1e-5,
                weight_decay: 0.1,
                beta1: 0.9,
                beta2: 0.95,
                epsilon: 1e-8,
                max_grad_norm: 1.0,
                warmup_fraction: 0.03, // 3% warmup
                decay_fraction: 0.30,  // Long decay
                total_steps: 100_000,
            },
            data: DataPreset {
                micro_batch_size: 2,
                gradient_accumulation_steps: 16, // Large effective batch = 32
                seq_length: 2048,
                num_workers: 8,
                shuffle: true,
            },
            hybrid: HybridPreset {
                warmup_steps: 1000,
                full_steps_per_cycle: 30,
                max_predict_steps: 200,
                confidence_threshold: 0.92,
                divergence_threshold: 0.10,
            },
        }
    }

    /// Create preset for specialist models (100M model).
    ///
    /// Optimized for task-specific capabilities with deep expertise in a narrow domain.
    ///
    /// ## Design Philosophy
    ///
    /// Specialists need different training characteristics than generalists:
    ///
    /// ### 1. Conservative Learning Schedule
    /// - **Longer warmup (5% vs 3%)**: Task-specific patterns can be more subtle.
    ///   Gradual learning prevents early overfitting to superficial patterns.
    /// - **Extended decay (30% vs 20%)**: More time to refine nuanced domain knowledge.
    /// - **Lower peak LR (6e-4 vs 3e-4)**: Specialists benefit from careful parameter
    ///   updates that preserve task-critical features.
    ///
    /// ### 2. Strong Regularization
    /// - **Higher weight decay (0.1)**: Prevents overfitting to limited domain data.
    /// - **Active dropout (0.1)**: Encourages robust representations that generalize
    ///   within the specialist domain.
    ///
    /// ### 3. Quality-First Hybrid Training
    /// - **More full computation (30 steps/cycle vs 20)**: Specialists require precise
    ///   gradient information to capture domain nuances.
    /// - **Conservative prediction (50 max steps vs 200)**: Prioritizes accuracy over
    ///   speed. Specialists cannot afford prediction drift.
    /// - **Higher confidence threshold (0.80 vs 0.92)**: Only predicts when highly certain,
    ///   reducing risk of divergence from optimal trajectory.
    ///
    /// ### 4. Early Stopping Support
    /// - **Patience (1000 steps)**: Monitors validation loss to prevent overfitting.
    /// - **Min delta (0.001)**: Requires meaningful improvement to continue.
    ///
    /// ## Use Cases
    /// - Medical/legal/scientific domain models
    /// - Code generation for specific frameworks
    /// - Mathematical theorem proving
    /// - Technical documentation generation
    pub fn specialist_100m() -> Self {
        Self {
            name: "specialist-100m".to_string(),
            description: "100M model optimized for deep task-specific expertise in narrow domains"
                .to_string(),
            model: ModelPreset {
                size: "100m".to_string(),
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                intermediate_size: 3072,
                vocab_size: 32000,
                max_seq_length: 2048, // Long context for complex domain content
                dropout: 0.1,         // Active regularization for generalization
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 3,
            },
            optimization: OptimizationPreset {
                learning_rate: 6e-4,     // Conservative for stability
                min_learning_rate: 1e-5, // Gentle final refinement
                weight_decay: 0.1,       // Strong regularization
                beta1: 0.9,
                beta2: 0.95, // Lower beta2 for task adaptation
                epsilon: 1e-8,
                max_grad_norm: 1.0,
                warmup_fraction: 0.05, // 5% warmup - careful initialization
                decay_fraction: 0.30,  // 30% decay - extended refinement
                total_steps: 50_000,
            },
            data: DataPreset {
                micro_batch_size: 2,
                gradient_accumulation_steps: 12, // Effective batch = 24
                seq_length: 1024,
                num_workers: 4,
                shuffle: true,
            },
            hybrid: HybridPreset {
                warmup_steps: 500,          // More warmup before prediction
                full_steps_per_cycle: 30,   // More full computation
                max_predict_steps: 50,      // Less prediction (quality over speed)
                confidence_threshold: 0.80, // Conservative prediction
                divergence_threshold: 0.12, // Tighter correction threshold
            },
        }
    }

    /// Create preset for generalist models (100M model).
    ///
    /// Optimized for broad knowledge across diverse tasks and domains.
    ///
    /// ## Design Philosophy
    ///
    /// Generalists prioritize breadth and adaptability:
    ///
    /// ### 1. Aggressive Learning Schedule
    /// - **Quick warmup (2%)**: Diverse data provides strong learning signal early.
    /// - **Shorter decay (20%)**: Broad patterns converge faster than specialist nuances.
    /// - **Higher peak LR (1e-3)**: Aggressive exploration of diverse patterns.
    ///
    /// ### 2. Minimal Regularization
    /// - **Lower weight decay (0.05)**: Preserves capacity for diverse knowledge.
    /// - **Minimal dropout (0.05)**: Maximizes information flow across domains.
    ///
    /// ### 3. Speed-Optimized Hybrid Training
    /// - **Less full computation (20 steps/cycle)**: Diverse data allows faster prediction.
    /// - **Aggressive prediction (200 max steps)**: Amortizes compute across many domains.
    /// - **Lower confidence threshold (0.92)**: Accepts more prediction risk for speed.
    ///
    /// ### 4. Large Effective Batch
    /// - **Batch size 32**: Stabilizes gradients across diverse data distribution.
    /// - **Longer sequences (2048)**: Handles varied content types.
    ///
    /// ## Comparison with Specialist
    ///
    /// | Aspect | Specialist | Generalist | Reasoning |
    /// |--------|-----------|------------|-----------|
    /// | **Learning Rate** | 6e-4 | 1e-3 | Specialists need careful updates |
    /// | **Warmup** | 5% | 2% | Task patterns need gentle intro |
    /// | **Decay Phase** | 30% | 20% | Domain refinement takes longer |
    /// | **Weight Decay** | 0.1 | 0.05 | Specialists prone to overfitting |
    /// | **Dropout** | 0.1 | 0.05 | Regularization vs capacity trade-off |
    /// | **Full Steps/Cycle** | 30 | 20 | Specialists need precise gradients |
    /// | **Max Predict Steps** | 50 | 200 | Quality vs speed priority |
    /// | **Confidence Threshold** | 0.80 | 0.92 | Conservative vs aggressive |
    /// | **Divergence Threshold** | 0.12 | 0.10 | Tighter correction for specialists |
    ///
    /// ## Use Cases
    /// - General-purpose assistants
    /// - Multi-task learning
    /// - Foundation models for transfer learning
    /// - Chatbots with broad knowledge
    pub fn generalist_100m() -> Self {
        Self {
            name: "generalist-100m".to_string(),
            description:
                "100M model optimized for broad knowledge across diverse tasks and domains"
                    .to_string(),
            model: ModelPreset {
                size: "100m".to_string(),
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                intermediate_size: 3072,
                vocab_size: 32000,
                max_seq_length: 2048, // Long context for varied content
                dropout: 0.05,        // Minimal regularization
                gradient_checkpointing: true,
                checkpoint_every_n_layers: 3,
            },
            optimization: OptimizationPreset {
                learning_rate: 1e-3, // Aggressive for broad learning
                min_learning_rate: 1e-5,
                weight_decay: 0.05, // Light regularization
                beta1: 0.9,
                beta2: 0.999, // Standard beta2 for diverse data
                epsilon: 1e-8,
                max_grad_norm: 1.0,
                warmup_fraction: 0.02, // 2% warmup - quick start
                decay_fraction: 0.20,  // 20% decay - standard
                total_steps: 100_000,  // More steps for diverse data
            },
            data: DataPreset {
                micro_batch_size: 2,
                gradient_accumulation_steps: 16, // Large effective batch = 32
                seq_length: 2048,                // Full sequences
                num_workers: 8,
                shuffle: true,
            },
            hybrid: HybridPreset {
                warmup_steps: 200,          // Quick warmup
                full_steps_per_cycle: 20,   // Standard ratio
                max_predict_steps: 200,     // Aggressive prediction
                confidence_threshold: 0.92, // High confidence
                divergence_threshold: 0.10, // Standard correction
            },
        }
    }

    /// Tokens per step for this configuration.
    pub fn tokens_per_step(&self) -> usize {
        self.data.tokens_per_step()
    }

    /// Estimated total tokens for training.
    pub fn total_tokens(&self) -> u64 {
        self.tokens_per_step() as u64 * self.optimization.total_steps
    }

    /// Print configuration summary.
    pub fn summary(&self) -> String {
        format!(
            r#"
=== Training Configuration: {} ===
{}

Model:
  Size: {} ({} params estimated)
  Hidden: {}, Layers: {}, Heads: {}
  Max Seq Length: {}
  Dropout: {:.2}
  Gradient Checkpointing: {}

Optimization:
  Learning Rate: {:.2e} → {:.2e}
  Weight Decay: {}
  Grad Clip: {}
  Warmup: {:.1}%, Decay: {:.1}%
  Total Steps: {}

Data:
  Batch Size: {} × {} = {} effective
  Seq Length: {}
  Tokens/Step: {}
  Total Tokens: {:.2}B

Hybrid Training:
  Warmup: {} steps
  Full/Cycle: {}, Max Predict: {}
  Confidence: {:.2}, Divergence: {:.2}
"#,
            self.name,
            self.description,
            self.model.size,
            estimate_params(&self.model),
            self.model.hidden_size,
            self.model.num_layers,
            self.model.num_heads,
            self.model.max_seq_length,
            self.model.dropout,
            if self.model.gradient_checkpointing {
                "Yes"
            } else {
                "No"
            },
            self.optimization.learning_rate,
            self.optimization.min_learning_rate,
            self.optimization.weight_decay,
            self.optimization.max_grad_norm,
            self.optimization.warmup_fraction * 100.0,
            self.optimization.decay_fraction * 100.0,
            self.optimization.total_steps,
            self.data.micro_batch_size,
            self.data.gradient_accumulation_steps,
            self.data.effective_batch_size(),
            self.data.seq_length,
            self.data.tokens_per_step(),
            self.total_tokens() as f64 / 1e9,
            self.hybrid.warmup_steps,
            self.hybrid.full_steps_per_cycle,
            self.hybrid.max_predict_steps,
            self.hybrid.confidence_threshold,
            self.hybrid.divergence_threshold,
        )
    }
}

/// Estimate parameter count from model config.
fn estimate_params(model: &ModelPreset) -> String {
    let embed = model.vocab_size * model.hidden_size;
    let per_layer = 4 * model.hidden_size * model.hidden_size  // attention
        + 2 * model.hidden_size * model.intermediate_size; // mlp
    let total = embed + model.num_layers * per_layer;

    if total >= 1_000_000_000 {
        format!("{:.1}B", total as f64 / 1e9)
    } else if total >= 1_000_000 {
        format!("{:.0}M", total as f64 / 1e6)
    } else {
        format!("{:.0}K", total as f64 / 1e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_preset() {
        let preset = TrainingPreset::research_assistant_100m();
        assert_eq!(preset.data.effective_batch_size(), 16);
        assert!(preset.optimization.learning_rate < 1e-4);
        println!("{}", preset.summary());
    }

    #[test]
    fn test_fast_preset() {
        let preset = TrainingPreset::fast_iteration_100m();
        assert_eq!(preset.data.effective_batch_size(), 8);
        println!("{}", preset.summary());
    }

    #[test]
    fn test_specialist_preset() {
        let preset = TrainingPreset::specialist_100m();

        // Verify specialist characteristics
        assert_eq!(preset.data.effective_batch_size(), 24);
        assert_eq!(preset.optimization.learning_rate, 6e-4);
        assert_eq!(preset.optimization.warmup_fraction, 0.05);
        assert_eq!(preset.optimization.decay_fraction, 0.30);
        assert_eq!(preset.model.dropout, 0.1);
        assert_eq!(preset.hybrid.full_steps_per_cycle, 30);
        assert_eq!(preset.hybrid.max_predict_steps, 50);
        assert_eq!(preset.hybrid.confidence_threshold, 0.80);

        println!("{}", preset.summary());
    }

    #[test]
    fn test_generalist_preset() {
        let preset = TrainingPreset::generalist_100m();

        // Verify generalist characteristics
        assert_eq!(preset.data.effective_batch_size(), 32);
        assert_eq!(preset.optimization.learning_rate, 1e-3);
        assert_eq!(preset.optimization.warmup_fraction, 0.02);
        assert_eq!(preset.optimization.decay_fraction, 0.20);
        assert_eq!(preset.model.dropout, 0.05);
        assert_eq!(preset.hybrid.full_steps_per_cycle, 20);
        assert_eq!(preset.hybrid.max_predict_steps, 200);
        assert_eq!(preset.hybrid.confidence_threshold, 0.92);

        println!("{}", preset.summary());
    }

    #[test]
    fn test_specialist_vs_generalist() {
        let specialist = TrainingPreset::specialist_100m();
        let generalist = TrainingPreset::generalist_100m();

        // Specialist should be more conservative
        assert!(
            specialist.optimization.learning_rate < generalist.optimization.learning_rate,
            "Specialist LR should be lower"
        );
        assert!(
            specialist.optimization.warmup_fraction > generalist.optimization.warmup_fraction,
            "Specialist warmup should be longer"
        );
        assert!(
            specialist.optimization.decay_fraction > generalist.optimization.decay_fraction,
            "Specialist decay should be longer"
        );
        assert!(
            specialist.model.dropout > generalist.model.dropout,
            "Specialist dropout should be higher"
        );
        assert!(
            specialist.optimization.weight_decay > generalist.optimization.weight_decay,
            "Specialist weight decay should be higher"
        );

        // Specialist should prioritize quality over speed in hybrid training
        assert!(
            specialist.hybrid.full_steps_per_cycle > generalist.hybrid.full_steps_per_cycle,
            "Specialist should do more full steps"
        );
        assert!(
            specialist.hybrid.max_predict_steps < generalist.hybrid.max_predict_steps,
            "Specialist should predict less"
        );
        assert!(
            specialist.hybrid.confidence_threshold < generalist.hybrid.confidence_threshold,
            "Specialist should have lower confidence threshold"
        );

        println!("\n=== SPECIALIST ===\n{}", specialist.summary());
        println!("\n=== GENERALIST ===\n{}", generalist.summary());
    }
}
