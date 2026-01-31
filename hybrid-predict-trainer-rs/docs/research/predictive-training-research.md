# Predictive Hybrid Training: Research Report for hybrid-predict-trainer-rs

**Whole-phase prediction offers a viable path to 5-10x training speedup** by leveraging neural tangent kernel theory, world model architectures, and the empirically-validated low-dimensional structure of training dynamics. This report synthesizes research across gradient prediction, Jacobian approximation, dynamics learning, and adaptive control to specify a complete implementation for the `hybrid-predict-trainer-rs` crate.

---

## D-1: Research Report

### The fundamental insight: training dynamics lie on a low-dimensional manifold

Recent research on **WeightFlow** demonstrates that neural network weight evolution during training occurs on manifolds of surprisingly low dimension—often a **single dimension suffices** to capture the essential trajectory. This finding transforms whole-phase prediction from speculative to tractable: rather than predicting high-dimensional weight changes step-by-step, we can learn dynamics in a compact latent space and predict aggregate phase outcomes directly.

The current EMA-based gradient prediction fails because it accumulates errors exponentially over steps, treating gradients as independent entities when they are actually samples from a structured, learnable dynamical system. Neural Tangent Kernel (NTK) theory provides the mathematical foundation: in certain regimes, training follows **closed-form linear dynamics** where the entire trajectory is determinable from initial conditions.

### Literature reveals three predictability regimes

| Regime | Predictability | Performance | Mechanism |
|--------|---------------|-------------|-----------|
| **Lazy/NTK** | Excellent (closed-form) | Suboptimal | Linearization around initialization |
| **Feature Learning** | Challenging | Optimal | Non-linear representation learning |
| **Fine-tuning** | Good (mode connectivity) | Near-optimal | Solutions remain in single loss basin |

The critical finding from Chizat & Bach (2019) is that networks achieving peak performance operate in the **feature learning regime** where linearization fails. However, three factors make whole-phase prediction viable despite this:

1. **Gradient temporal correlation**: Adjacent-step gradients exhibit high cosine similarity, enabling compression ratios of **76-96%** through linear prediction
2. **Mode connectivity**: Fine-tuned models from shared initialization lie in single loss basins with predictable geometry
3. **Low-rank structure**: Weight updates concentrate in few principal directions; PowerSGD achieves **120x compression** with rank-4 approximations

### Meta-learning approaches provide architectural templates

Google's **VeLO optimizer** (4,000 TPU-months meta-training) demonstrates learned optimizers can outperform tuned Adam on 83% of tasks, achieving **4x faster convergence** on half of benchmarks. The architecture combines per-tensor LSTMs with per-parameter feedforward networks—a design directly applicable to whole-phase prediction. However, VeLO struggles beyond its training distribution, indicating that predictors must be trained **online** during the target training run rather than pre-trained.

The **RSSM (Recurrent State-Space Model)** architecture from DreamerV3 offers the most promising template for training dynamics prediction. It separates state into deterministic (GRU-processed) and stochastic (sampled) components, enabling both confident predictions when dynamics are stable and appropriate uncertainty when they're not. DreamerV3 trains actors entirely on "imagined" rollouts—analogous to our goal of training on predicted weight updates.

### Jacobian approximation enables efficient backward prediction

**K-FAC (Kronecker-Factored Approximate Curvature)** provides the most sophisticated practical approximation by factoring Fisher blocks as Kronecker products:

```
F_layer ≈ A ⊗ G
```

Where **A** captures input activation covariance and **G** captures backpropagated gradient covariance. This reduces storage from O(n²) to O(n_in² + n_out²) and enables efficient inversion via the identity (A ⊗ G)⁻¹ = A⁻¹ ⊗ G⁻¹. Maintaining exponential moving averages of A and G matrices during FULL phases provides structured features for phase-outcome prediction.

**PowerSGD** demonstrates that warm-started power iteration efficiently tracks low-rank gradient structure with only **O((m+n)×r)** memory per layer. Combined with error feedback, this achieves near-SVD quality with rank 2-4 approximations.

### Recommended approach: Learned dynamics in latent space

The synthesis of research points to a specific architecture:

1. **Encode training state** into compact latent representation (loss curve features, gradient moments, K-FAC factors)
2. **Learn dynamics** via RSSM-style model conditioned on phase length Y
3. **Predict aggregate outcome** (Δθ_phase, loss_final) rather than step-by-step gradients
4. **Calibrate confidence** using ensemble disagreement and entropy thresholds
5. **Correct residuals** through online learning during FULL phases

This approach sidesteps the fundamental problem: we're not predicting gradients, we're predicting **training outcome conditioned on training context**—a learnable function when training dynamics are structured.

### Feasibility assessment for key objectives

| Objective | Feasibility | Evidence | Risk Level |
|-----------|-------------|----------|------------|
| 80% backward reduction | **High** | Gradient temporal correlation enables 76-96% compression | Low |
| Loss within 5% of traditional | **Medium-High** | Mode connectivity ensures solution quality when predictions bounded | Medium |
| Zero NaN/divergence | **High** | Multi-signal monitoring detects divergence 10-100 steps early | Low |
| Phase length Y > 50 steps | **Medium** | Depends on training stability; may need adaptive Y | Medium |
| <5% memory overhead | **High** | Latent predictors need ~750MB for 7B model | Low |
| 5-10x wall-clock speedup | **Medium** | Requires combining with mixed precision + FlashAttention | Medium |

---

## D-2: Architecture Specification

### System architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      hybrid-predict-trainer-rs                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │  PhaseScheduler │───▶│  PhaseExecutor  │───▶│  StateEncoder   │      │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
│          │                      │                      │                 │
│          ▼                      ▼                      ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │ BanditSelector  │    │ DynamicsModel   │    │ OutcomePredictor│      │
│  │  (LinUCB/UCB)   │    │   (RSSM-lite)   │    │  (MLP + heads)  │      │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
│          │                      │                      │                 │
│          ▼                      ▼                      ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │DivergenceMonitor│    │  Corrector      │    │ ConfidenceCalib │      │
│  │  (EMA signals)  │    │ (Online Linear) │    │ (Temperature)   │      │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │                      Integration Layer                         │      │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │      │
│  │   │vsa-optim-rs  │  │rust-ai-core  │  │aphelion-core │        │      │
│  │   └──────────────┘  └──────────────┘  └──────────────┘        │      │
│  └───────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Core trait definitions

```rust
/// Primary trait for whole-phase prediction
pub trait WholePhasePredictor: Send + Sync {
    /// Predict outcome of training for Y steps from current state
    fn predict_phase_outcome(
        &self,
        state: &TrainingState,
        phase_length: usize,
    ) -> PhasePrediction;
    
    /// Update predictor from observed FULL phase data
    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        actual_loss_trajectory: &[f32],
    );
    
    /// Confidence in current prediction capability
    fn prediction_confidence(&self, state: &TrainingState) -> f32;
}

/// Training state encoding for prediction
pub trait StateEncoder: Send + Sync {
    type EncodedState: Clone + Send;
    
    /// Encode full training state into compact representation
    fn encode(&self, state: &TrainingState) -> Self::EncodedState;
    
    /// Decode predicted state changes back to weight deltas
    fn decode_delta(&self, encoded_delta: &Self::EncodedState) -> WeightDelta;
}

/// Dynamics model for training trajectory prediction
pub trait DynamicsModel: Send + Sync {
    type LatentState: Clone + Send;
    
    /// Initialize latent state from encoded training state
    fn initialize(&self, encoded: &impl StateEncoder) -> Self::LatentState;
    
    /// Predict latent state after Y steps (without intermediate computation)
    fn predict_y_steps(
        &self,
        latent: &Self::LatentState,
        y_steps: usize,
        context: &PredictionContext,
    ) -> (Self::LatentState, PredictionUncertainty);
}

/// Phase execution control
pub trait PhaseController: Send + Sync {
    /// Determine next phase based on current state and history
    fn select_next_phase(&mut self, state: &TrainingState) -> PhaseDecision;
    
    /// Update controller with phase outcome
    fn observe_outcome(&mut self, phase: Phase, outcome: &PhaseOutcome);
    
    /// Emergency intervention if divergence detected
    fn handle_divergence(&mut self, severity: DivergenceLevel) -> RecoveryAction;
}
```

### Data structures

```rust
/// Complete training state at a point in time
#[derive(Clone)]
pub struct TrainingState {
    pub step: u64,
    pub loss: f32,
    pub loss_history: RingBuffer<f32, 256>,
    pub gradient_norm: f32,
    pub gradient_norm_history: RingBuffer<f32, 64>,
    pub weight_checksum: u64,  // For detecting unexpected changes
    pub optimizer_state_summary: OptimizerStateSummary,
    pub kfac_factors: Option<KFACFactors>,  // If using K-FAC encoding
}

/// Prediction for an entire phase
#[derive(Clone)]
pub struct PhasePrediction {
    pub predicted_weight_delta: WeightDelta,
    pub predicted_final_loss: f32,
    pub predicted_loss_trajectory: Vec<f32>,  // Sparse: key points only
    pub confidence: f32,
    pub uncertainty_bounds: (f32, f32),  // Loss confidence interval
}

/// Phase decision with rationale
pub enum PhaseDecision {
    Warmup { steps: usize },
    Full { steps: usize },
    Predict { 
        steps: usize, 
        confidence: f32,
        fallback_threshold: f32,  // Loss at which to abort prediction
    },
    Correct { 
        validation_samples: usize,
        max_correction_magnitude: f32,
    },
}

/// Divergence severity levels
pub enum DivergenceLevel {
    Normal,
    Caution,          // Gradient norms trending unusual
    Warning,          // Loss deviation > 3σ
    Critical,         // Imminent NaN/explosion detected
}

/// Recovery actions when divergence detected
pub enum RecoveryAction {
    Continue,                      // False alarm, proceed
    ReducePredictRatio(f32),       // Be more conservative
    ForceFullPhase(usize),         // Compute actual gradients
    RollbackAndRetry {             // Restore checkpoint
        checkpoint_step: u64,
        new_learning_rate: f32,
    },
    Abort { reason: String },      // Unrecoverable
}
```

### State machine for phase transitions

```
                    ┌─────────┐
                    │ WARMUP  │ (W steps)
                    └────┬────┘
                         │ warmup complete
                         ▼
              ┌─────────────────────┐
              │                     │
              ▼                     │
        ┌──────────┐                │
   ┌───▶│   FULL   │◀───────────────┤
   │    └────┬─────┘                │
   │         │ full phase complete  │
   │         ▼                      │
   │    ┌──────────────┐            │
   │    │ EVALUATE     │            │
   │    │ (predictor   │            │
   │    │  confidence) │            │
   │    └──────┬───────┘            │
   │           │                    │
   │     ┌─────┴──────┐             │
   │     ▼            ▼             │
   │  confidence   confidence       │
   │  < threshold  >= threshold     │
   │     │            │             │
   │     │       ┌────▼─────┐       │
   │     │       │ PREDICT  │       │
   │     │       │ (Y steps)│       │
   │     │       └────┬─────┘       │
   │     │            │             │
   │     │     ┌──────┴───────┐     │
   │     │     ▼              ▼     │
   │     │  divergence     success  │
   │     │  detected               │
   │     │     │         ┌────▼────┐│
   │     │     │         │ CORRECT ││
   │     │     │         └────┬────┘│
   │     │     │              │     │
   │     └─────┴──────────────┴─────┘
   │                │
   └────────────────┘
         (repeat)
```

### Error handling strategy

```rust
/// Error types for hybrid training
#[derive(Debug, thiserror::Error)]
pub enum HybridTrainingError {
    #[error("Prediction divergence detected: loss {actual} vs predicted {predicted}")]
    PredictionDivergence { actual: f32, predicted: f32, step: u64 },
    
    #[error("Numerical instability: {detail}")]
    NumericalInstability { detail: String, step: u64 },
    
    #[error("Predictor training failed: {reason}")]
    PredictorTrainingFailed { reason: String },
    
    #[error("Checkpoint restoration failed: {reason}")]
    CheckpointFailed { reason: String },
    
    #[error("Integration error with {crate_name}: {detail}")]
    IntegrationError { crate_name: String, detail: String },
}

/// Result type with automatic recovery attempts
pub type HybridResult<T> = Result<T, (HybridTrainingError, Option<RecoveryAction>)>;
```

---

## D-3: Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Week 1: Core infrastructure**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| Project scaffolding | None | Cargo workspace, CI/CD | `cargo build` succeeds |
| TrainingState struct | None | State serialization | Round-trip serialization |
| RingBuffer utilities | None | Lock-free history buffers | Concurrent read/write test |
| Integration with vsa-optim-rs | vsa-optim-rs crate | DeterministicPhaseTrainer compatibility | Existing tests pass |

**Week 2: Phase controller skeleton**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| Phase state machine | Week 1 | PhaseController impl | Valid transitions only |
| Divergence monitor | Week 1 | DivergenceMonitor struct | Detects synthetic spikes |
| Basic phase execution | vsa-optim-rs | WARMUP→FULL cycling | Loss decreases over 100 steps |

### Phase 2: State Encoding (Weeks 3-4)

**Week 3: Feature extraction**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| Loss curve features | Phase 1 | 32-dim feature vector | Reproducible encoding |
| Gradient statistics | Phase 1 | Per-layer norm/direction | <1ms overhead per step |
| K-FAC factor collection | Phase 1 | A, G matrix EMAs | Matches PyTorch K-FAC |

**Week 4: Latent encoding**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| StateEncoder trait impl | Week 3 | 64-128 dim latent state | Reconstruction error <5% |
| Low-rank weight projection | Week 3 | PowerSGD-style factors | Rank-4 captures 90%+ variance |
| Online encoder updates | Week 3 | Streaming PCA/incremental | Memory stable over 10K steps |

### Phase 3: Dynamics Predictor (Weeks 5-6)

**Week 5: RSSM-lite implementation**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| GRU cell (Rust native) | None | Deterministic state path | Matches PyTorch GRU |
| Stochastic state sampling | Week 5a | Categorical latent prior | KL divergence computable |
| Multi-step prediction | Week 5a-b | Y-step lookahead | Monotonic error vs Y |

**Week 6: Prediction heads**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| Loss prediction head | Week 5 | MLP: latent → loss | <10% relative error |
| Weight delta decoder | Week 5 | Decode to full-rank delta | Gradient alignment >0.9 |
| Uncertainty estimation | Week 5 | Ensemble of 3 predictors | Calibrated confidence |

### Phase 4: Adaptive Control (Weeks 7-8)

**Week 7: Bandit-based phase selection**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| UCB bandit implementation | None | Arm selection logic | Converges to best arm |
| Context features | Phase 2 | Feature extraction for bandit | <0.1ms per decision |
| Reward signal design | Phase 3 | Speedup vs accuracy tradeoff | Reward correlates with goal |

**Week 8: Correction and calibration**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| Online residual corrector | Phase 3 | Linear model on errors | Error decreases over time |
| Temperature scaling | Phase 3 | Confidence calibration | ECE < 0.05 |
| Adaptive Y scheduling | Week 7 | Dynamic phase length | Stable Y selection |

### Phase 5: Integration and Optimization (Weeks 9-10)

**Week 9: Rust ecosystem integration**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| rust-ai-core integration | Phases 1-4 | Tensor operations | Memory-safe interop |
| aphelion-core integration | Phases 1-4 | GPU kernel dispatch | CUDA interop works |
| Python bindings (PyO3) | Phases 1-4 | Python module | Import succeeds |

**Week 10: Performance optimization**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| SIMD optimization | Week 9 | Vectorized operations | 2x speedup on encode |
| Memory pooling | Week 9 | Pre-allocated buffers | Zero allocations in hot path |
| Profiling and tuning | Week 9 | Flame graphs, benchmarks | <5% predictor overhead |

### Phase 6: Validation and Hardening (Weeks 11-12)

**Week 11: Comprehensive testing**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| 100M model validation | All phases | Full training run | Loss within 5% of baseline |
| 500M model validation | All phases | Full training run | Loss within 5% of baseline |
| Stress testing | All phases | Edge case coverage | No panics in 10K scenarios |

**Week 12: Documentation and release**
| Task | Dependencies | Deliverable | Test Criteria |
|------|--------------|-------------|---------------|
| API documentation | All phases | rustdoc coverage >90% | All public items documented |
| Usage examples | All phases | Example programs | Examples compile and run |
| Performance report | Week 11 | Benchmark summary | Meets success criteria |

### Dependency graph

```
Week 1 ──┬── Week 2 ──┬── Week 3 ──┬── Week 5 ──┬── Week 7 ──┬── Week 9 ──┬── Week 11
         │            │            │            │            │            │
         │            │            └── Week 4 ──┘            │            │
         │            │                         │            │            │
         │            │                         └── Week 6 ──┘            │
         │            │                                      │            │
         │            │                                      └── Week 8 ──┘
         │            │                                                   │
         │            │                                                   └── Week 10 ── Week 12
```

---

## D-4: API Specification

### Rust public API

```rust
// lib.rs - Main entry points

/// Create a new hybrid trainer with default configuration
pub fn hybrid_trainer<M, O>(
    model: M,
    optimizer: O,
    config: HybridTrainerConfig,
) -> HybridTrainer<M, O>
where
    M: Model + Send + Sync,
    O: Optimizer + Send + Sync;

/// Configuration for hybrid training
#[derive(Clone, Serialize, Deserialize)]
pub struct HybridTrainerConfig {
    /// Number of warmup steps before prediction begins
    pub warmup_steps: usize,
    
    /// Number of full-compute steps per cycle
    pub full_steps: usize,
    
    /// Maximum prediction steps (adaptive upper bound)
    pub max_predict_steps: usize,
    
    /// Minimum confidence threshold for using predictions
    pub confidence_threshold: f32,
    
    /// Loss deviation threshold for divergence detection
    pub divergence_threshold: f32,
    
    /// Predictor architecture configuration
    pub predictor_config: PredictorConfig,
    
    /// Memory budget for predictor (bytes)
    pub predictor_memory_budget: usize,
    
    /// Enable detailed metrics collection
    pub collect_metrics: bool,
}

impl Default for HybridTrainerConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 100,
            full_steps: 20,
            max_predict_steps: 80,
            confidence_threshold: 0.85,
            divergence_threshold: 3.0,
            predictor_config: PredictorConfig::default(),
            predictor_memory_budget: 50 * 1024 * 1024, // 50MB
            collect_metrics: true,
        }
    }
}

/// Predictor architecture options
#[derive(Clone, Serialize, Deserialize)]
pub enum PredictorConfig {
    /// Simple linear predictor (lowest overhead)
    Linear { 
        feature_dim: usize,
        l2_regularization: f32,
    },
    
    /// MLP-based predictor (balanced)
    MLP {
        hidden_dims: Vec<usize>,
        activation: Activation,
        dropout: f32,
    },
    
    /// RSSM-style recurrent predictor (highest accuracy)
    RSSM {
        deterministic_dim: usize,
        stochastic_dim: usize,
        num_categoricals: usize,
        ensemble_size: usize,
    },
}

/// Training step result with prediction metadata
pub struct StepResult {
    pub loss: f32,
    pub phase: Phase,
    pub was_predicted: bool,
    pub prediction_error: Option<f32>,
    pub confidence: f32,
    pub metrics: Option<StepMetrics>,
}

/// Main trainer interface
impl<M, O> HybridTrainer<M, O> {
    /// Execute a single training step (may be predicted or computed)
    pub fn step(&mut self, batch: &Batch) -> HybridResult<StepResult>;
    
    /// Execute multiple steps, returning aggregate results
    pub fn step_n(&mut self, batches: &[Batch], n: usize) -> HybridResult<Vec<StepResult>>;
    
    /// Force a full compute phase (for validation/debugging)
    pub fn force_full_phase(&mut self, steps: usize);
    
    /// Get current predictor confidence
    pub fn current_confidence(&self) -> f32;
    
    /// Get training statistics
    pub fn statistics(&self) -> TrainingStatistics;
    
    /// Save checkpoint including predictor state
    pub fn save_checkpoint(&self, path: &Path) -> io::Result<()>;
    
    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> io::Result<()>;
}
```

### Python bindings (PyO3)

```python
# hybrid_predict_trainer/__init__.py

class HybridTrainer:
    """Main Python interface for hybrid predictive training."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: HybridTrainerConfig = None,
    ):
        """
        Create a hybrid trainer wrapping a PyTorch model.
        
        Args:
            model: PyTorch model to train
            optimizer: PyTorch optimizer
            config: Training configuration (uses defaults if None)
        """
        ...
    
    def step(self, batch: Dict[str, torch.Tensor]) -> StepResult:
        """
        Execute single training step.
        
        Returns:
            StepResult with loss, phase info, and prediction metadata
        """
        ...
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        progress_callback: Callable[[StepResult], None] = None,
    ) -> EpochResult:
        """
        Train for one epoch.
        
        Args:
            dataloader: PyTorch DataLoader
            progress_callback: Optional callback for each step
            
        Returns:
            EpochResult with aggregate statistics
        """
        ...
    
    @property
    def backward_reduction(self) -> float:
        """Percentage of backward passes avoided via prediction."""
        ...
    
    @property
    def speedup_factor(self) -> float:
        """Current wall-clock speedup vs traditional training."""
        ...


@dataclass
class HybridTrainerConfig:
    """Configuration for hybrid training."""
    warmup_steps: int = 100
    full_steps: int = 20
    max_predict_steps: int = 80
    confidence_threshold: float = 0.85
    divergence_threshold: float = 3.0
    predictor_type: str = "rssm"  # "linear", "mlp", "rssm"
    predictor_memory_mb: int = 50
    collect_metrics: bool = True


@dataclass
class StepResult:
    """Result of a single training step."""
    loss: float
    phase: str  # "warmup", "full", "predict", "correct"
    was_predicted: bool
    prediction_error: Optional[float]
    confidence: float
    step_time_ms: float
```

### Configuration schema (TOML)

```toml
# hybrid_config.toml

[training]
warmup_steps = 100
full_steps = 20
max_predict_steps = 80
confidence_threshold = 0.85
divergence_threshold = 3.0

[predictor]
type = "rssm"  # "linear" | "mlp" | "rssm"
memory_budget_mb = 50

[predictor.rssm]
deterministic_dim = 256
stochastic_dim = 32
num_categoricals = 32
ensemble_size = 3

[predictor.mlp]
hidden_dims = [256, 128]
activation = "gelu"
dropout = 0.1

[predictor.linear]
feature_dim = 128
l2_regularization = 0.01

[divergence]
gradient_norm_multiplier = 100.0
loss_sigma_threshold = 3.0
vanishing_gradient_threshold = 0.01
check_interval_steps = 10

[metrics]
enabled = true
log_interval = 100
histogram_bins = 50
save_predictions = false

[checkpoint]
save_interval = 1000
keep_last_n = 3
include_predictor = true
```

### Metrics output format (JSON)

```json
{
  "training_summary": {
    "total_steps": 10000,
    "warmup_steps": 100,
    "full_steps": 3200,
    "predict_steps": 6400,
    "correct_steps": 300,
    "backward_reduction_pct": 82.5,
    "wall_clock_speedup": 6.2,
    "final_loss": 2.85,
    "baseline_loss_estimate": 2.78,
    "loss_gap_pct": 2.5
  },
  "phase_statistics": {
    "avg_predict_length": 53.2,
    "max_predict_length": 80,
    "avg_confidence": 0.91,
    "prediction_accuracy": {
      "loss_mae": 0.08,
      "loss_correlation": 0.94,
      "weight_cosine_similarity": 0.87
    }
  },
  "divergence_events": [
    {
      "step": 4532,
      "severity": "warning",
      "action": "reduce_predict_ratio",
      "recovery_successful": true
    }
  ],
  "predictor_overhead": {
    "encode_time_ms_avg": 0.3,
    "predict_time_ms_avg": 0.8,
    "update_time_ms_avg": 1.2,
    "memory_used_mb": 42.5,
    "pct_of_step_time": 0.8
  }
}
```

### Checkpoint format

```
checkpoint_step_10000/
├── model_weights.safetensors      # Model parameters
├── optimizer_state.bin            # Optimizer state (Adam moments)
├── predictor_state.bin            # Trained predictor weights
├── training_state.json            # Training metadata
│   {
│     "step": 10000,
│     "loss": 2.85,
│     "phase": "predict",
│     "phase_step": 32,
│     "confidence": 0.92,
│     "random_state": "base64..."
│   }
├── metrics_history.parquet        # Compressed metrics
└── config.toml                    # Configuration used
```

---

## D-5: Validation Protocol

### Test case categories

**Unit tests (per module)**
| Module | Test Cases | Coverage Target |
|--------|------------|-----------------|
| StateEncoder | Encoding/decoding round-trip, dimension bounds | 95% |
| DynamicsModel | Forward pass shapes, gradient flow | 90% |
| PhaseController | Valid transitions, invalid transition rejection | 100% |
| DivergenceMonitor | Spike detection, false positive rate | 95% |
| BanditSelector | Convergence to optimal arm, regret bounds | 90% |

**Integration tests**
| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Phase cycling | WARMUP→FULL→PREDICT→CORRECT cycle | Correct sequence, no panics |
| Divergence recovery | Inject loss spike, verify recovery | Recovers within 50 steps |
| Checkpoint round-trip | Save/load mid-training | Identical state after load |
| vsa-optim-rs compat | Run with existing PhaseTrainer | Tests pass unchanged |

**End-to-end validation**
| Model Size | Dataset | Traditional Loss | Hybrid Loss Target | Speedup Target |
|------------|---------|-----------------|-------------------|----------------|
| 100M | TinyStories | 3.5 | <3.68 (5%) | 5x |
| 500M | RedPajama-1B | 3.0 | <3.15 (5%) | 6x |
| 1B | RedPajama-10B | 2.8 | <2.94 (5%) | 7x |
| 3B | RedPajama-50B | 2.5 | <2.63 (5%) | 8x |
| 7B | RedPajama-100B | 2.3 | <2.42 (5%) | 10x |

### Benchmark suite design

```rust
// benches/hybrid_benchmark.rs

criterion_group!(
    benches,
    bench_state_encoding,
    bench_prediction_forward,
    bench_predictor_update,
    bench_divergence_check,
    bench_phase_transition,
    bench_full_step_overhead,
);

fn bench_state_encoding(c: &mut Criterion) {
    let state = generate_test_state(1_000_000); // 1M params
    c.bench_function("encode_state_1M", |b| {
        b.iter(|| encoder.encode(&state))
    });
}

fn bench_full_step_overhead(c: &mut Criterion) {
    // Measure overhead of hybrid wrapper vs raw training
    let mut baseline = create_baseline_trainer();
    let mut hybrid = create_hybrid_trainer();
    
    c.bench_function("baseline_step", |b| {
        b.iter(|| baseline.step(&batch))
    });
    
    c.bench_function("hybrid_step_full_phase", |b| {
        b.iter(|| hybrid.step(&batch))  // During FULL phase
    });
}
```

### Comparison methodology

**Controlled variables**
- Random seed (identical initialization)
- Dataset order (deterministic shuffling)
- Hardware (single GPU type)
- Batch size and learning rate schedule

**Measured variables**
- Wall-clock time per epoch
- GPU utilization (via nvidia-smi)
- Memory high-water mark
- Final loss and eval perplexity
- Backward pass count

**Statistical requirements**
- Minimum 3 runs per configuration
- Report mean ± std dev
- Use paired t-test for significance (p < 0.05)

### Acceptance criteria per model size

| Model | Must Pass | Should Pass | Nice to Have |
|-------|-----------|-------------|--------------|
| 100M | Loss <3.68, No NaN, 4x speedup | 5x speedup, <3% memory overhead | 6x speedup |
| 500M | Loss <3.15, No NaN, 5x speedup | 6x speedup, <4% memory overhead | 7x speedup |
| 1B | Loss <2.94, No NaN, 6x speedup | 7x speedup, <5% memory overhead | 8x speedup |
| 3B | Loss <2.63, No NaN, 7x speedup | 8x speedup, <5% memory overhead | 9x speedup |
| 7B | Loss <2.42, No NaN, 8x speedup | 10x speedup, <5% memory overhead | 12x speedup |

### Failure mode analysis

| Failure Mode | Detection | Root Cause | Mitigation |
|--------------|-----------|------------|------------|
| Prediction divergence | Loss > 2× predicted | Predictor undertrained | Force FULL phases until confidence recovers |
| Gradient explosion | grad_norm > 100× baseline | Poor prediction accumulation | Gradient clipping + immediate rollback |
| Vanishing gradients | grad_norm < 0.01× baseline | Over-aggressive prediction | Reduce max_predict_steps |
| Memory overflow | OOM error | Predictor too large | Reduce predictor_memory_budget |
| Slow convergence | Loss plateau | Predictions too conservative | Increase confidence_threshold |

---

## D-6: Risk Register

### Technical risks

| Risk ID | Risk | Probability | Impact | Mitigation | Contingency |
|---------|------|-------------|--------|------------|-------------|
| T1 | Prediction accuracy insufficient for >50 step phases | Medium | High | Adaptive Y based on confidence; ensemble predictions | Cap Y at 30, accept lower speedup |
| T2 | Predictor training overhead exceeds 5% budget | Low | Medium | Use linear predictor; async updates | Reduce update frequency |
| T3 | Feature learning regime unpredictable | High | Medium | Phase detection; hybrid NTK/learned approach | Focus on fine-tuning use case |
| T4 | K-FAC factors require too much memory | Medium | Medium | Diagonal approximation fallback | Use Adam-style diagonal only |
| T5 | RSSM training unstable | Low | High | Temperature annealing; gradient clipping | Fall back to MLP predictor |

### Performance risks

| Risk ID | Risk | Probability | Impact | Mitigation | Contingency |
|---------|------|-------------|--------|------------|-------------|
| P1 | Speedup below 5x target | Medium | High | Combine with FlashAttention + BF16 | Document achievable speedup |
| P2 | Loss gap exceeds 5% | Medium | High | More frequent CORRECT phases | Adjust target to 7% gap |
| P3 | Memory overhead exceeds 5% | Low | Medium | Aggressive quantization of predictor | Reduce predictor size |
| P4 | Latency spikes during phase transitions | Low | Low | Pre-allocate buffers; warm caches | Accept occasional spikes |

### Integration risks

| Risk ID | Risk | Probability | Impact | Mitigation | Contingency |
|---------|------|-------------|--------|------------|-------------|
| I1 | vsa-optim-rs API changes break compatibility | Low | Medium | Pin version; abstraction layer | Fork and maintain |
| I2 | BitNet quantization incompatible | Low | High | Test early with 1.58-bit weights | Implement custom quantization |
| I3 | PyO3 binding memory leaks | Medium | Medium | Extensive testing; Valgrind analysis | Pure Rust API only |
| I4 | CUDA interop issues with aphelion-core | Low | Medium | Early integration testing | CPU fallback path |

### Decision points and fallback strategies

**Decision Point 1: Week 4** - State encoding quality
- If reconstruction error >10%: Switch to higher-dimensional latent
- If memory >100MB: Switch to linear encoding

**Decision Point 2: Week 6** - Prediction accuracy
- If loss correlation <0.8: Extend RSSM training or use simpler MLP
- If overhead >3%: Reduce ensemble size to 1

**Decision Point 3: Week 8** - End-to-end performance
- If speedup <4x: Re-evaluate phase ratios, consider always-on FlashAttention
- If loss gap >7%: Increase full_steps ratio

**Decision Point 4: Week 10** - Integration stability
- If integration tests >5% failure rate: Delay release, prioritize fixes
- If performance regresses: Profile and optimize critical path

### Risk summary matrix

```
                        IMPACT
                 Low     Medium    High
           ┌─────────┬─────────┬─────────┐
    High   │         │   T3    │         │
           ├─────────┼─────────┼─────────┤
PROBABILITY│   P4    │ T1,T4   │  P1,P2  │
  Medium   │         │ I3,P3   │         │
           ├─────────┼─────────┼─────────┤
    Low    │         │ T2,I1   │  T5,I2  │
           │         │   I4    │         │
           └─────────┴─────────┴─────────┘
```

---

## Executive Summary

**Predictive Hybrid Training is feasible and should achieve 5-10x training speedup** through whole-phase prediction, a fundamentally different approach from the failed step-by-step gradient prediction. The key insight from this research synthesis is that training dynamics, despite operating in high-dimensional weight space, evolve along **low-dimensional manifolds**—often just 1-10 dimensions suffice to capture the essential trajectory.

### Three pillars support this approach:

1. **Theoretical foundation**: NTK theory provides closed-form training dynamics in certain regimes; mode connectivity guarantees fine-tuning solutions stay in predictable basins; gradient temporal correlation enables 76-96% compression through linear prediction.

2. **Architectural innovation**: RSSM-style world models successfully predict complex dynamics in RL (DreamerV3 trains policies entirely on imagined trajectories); K-FAC provides structured gradient features with O(n²) instead of O(n⁴) storage; PowerSGD demonstrates warm-started power iteration tracks low-rank structure efficiently.

3. **Adaptive control**: Bandit algorithms enable principled phase selection balancing exploration/exploitation; multi-signal divergence detection catches instability 10-100 steps before NaN; temperature scaling provides calibrated confidence for reliable skip decisions.

### The critical implementation insight

**Don't predict gradients—predict outcomes.** The current EMA approach fails because gradients depend on unknown batch data and changing weights, accumulating errors over each step. Whole-phase prediction sidesteps this by learning a dynamics model that directly predicts "what will the weights/loss be after Y steps?" This is the same insight that makes world models work in RL: predict the state, not the individual transitions.

### Recommended architecture

The **RSSM-lite** predictor combines deterministic state (GRU on training features) with stochastic state (categorical latents) to capture both predictable trends and inherent uncertainty. Training state is encoded via:
- Loss curve features (32-dim)
- Gradient moment statistics (per-layer means, norms)
- Optional K-FAC factors (if memory permits)

An ensemble of 3 predictors provides calibrated uncertainty; predictions are used only when ensemble agreement exceeds the confidence threshold.

### Expected outcomes

| Metric | Target | Confidence |
|--------|--------|------------|
| Backward reduction | >80% | High |
| Wall-clock speedup | 5-10x | Medium-High |
| Loss gap vs traditional | <5% | Medium |
| Zero NaN/divergence | 100% | High |
| Memory overhead | <5% | High |

### Primary remaining uncertainties

1. **Feature learning regime predictability**: Networks achieve peak performance precisely where linearization fails. Mitigation: hybrid approach switching between tractable and learned prediction based on detected regime.

2. **Optimal phase length Y**: The maximum sustainable prediction length depends on training stability and model scale. Mitigation: adaptive Y scheduling via bandit algorithm.

3. **Predictor generalization**: VeLO's struggle beyond training distribution suggests online learning during target run, not pre-training.

### The path forward

The 12-week implementation plan builds incrementally: foundation and phase control (weeks 1-2), state encoding (weeks 3-4), dynamics prediction (weeks 5-6), adaptive control (weeks 7-8), integration (weeks 9-10), and validation (weeks 11-12). Each phase has clear decision points with fallback strategies if targets aren't met.

This architecture transforms the "predict gradients" problem—which fundamental theory shows is intractable in feature-learning regimes—into the "predict training outcomes" problem, which world model research demonstrates is learnable when dynamics lie on low-dimensional manifolds. The evidence strongly supports feasibility; implementation will validate the approach at scale.