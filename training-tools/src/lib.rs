//! Training Tools for rust-ai
//!
//! This crate provides:
//! - Real-time training monitoring with TUI
//! - HuggingFace Hub integration for model uploads
//! - Checkpoint management with compression and cleanup
//! - Progressive parameter expansion training (100M → 500M → 1B)
//! - GPU memory management and monitoring
//! - Training alerts and notification system
//! - Adaptive training control with dynamic hyperparameter adjustment
//!
//! # Binaries
//!
//! - `train` - Main training orchestrator with progressive expansion
//! - `train-monitor` - Real-time TUI monitor for active training
//! - `hf-upload` - Upload models and checkpoints to HuggingFace Hub

pub mod adaptive;
pub mod alerts;
pub mod attention_viz;
pub mod batch_tuner;
pub mod checkpoint_manager;
pub mod curve_analysis;
pub mod early_stopping;
pub mod embeddings;
pub mod gpu_stats;
pub mod inference_viz;
pub mod gradient_control;
pub mod landscape;
pub mod hf;
pub mod live_monitor;
pub mod lr_advisor;
pub mod lr_controller;
pub mod lr_scheduler;
pub mod memory;
pub mod monitor;
#[cfg(feature = "parquet")]
pub mod parquet_export;
pub mod progressive;
pub mod screenshot;
pub mod smoothing;
pub mod training_config;
pub mod training_state;
pub mod training_viz;
pub mod viz3d;
pub mod wordcloud;
pub mod rerun_viz;

pub use alerts::{Alert, AlertCondition, AlertManager, AlertSeverity, MetricsSnapshot};
pub use batch_tuner::{BatchTuner, BatchTunerStats};
pub use checkpoint_manager::CheckpointManager;
pub use curve_analysis::{CurveAnalysis, CurveAnalyzer, CurveTrend};
pub use early_stopping::{EarlyStopping, StoppingDecision, StoppingMode};
pub use gpu_stats::{query_gpu_stats, GpuStats, GpuStatsMonitor};
pub use gradient_control::{GradientAction, GradientController, GradientStats};
pub use hf::HuggingFaceUploader;
pub use lr_advisor::{analyze_lr, Issue, LRAdvice, TrainingPhase as LRTrainingPhase, Urgency};
pub use lr_controller::AdaptiveLRController;
pub use lr_scheduler::{LRScheduler, SchedulerError, WSDScheduler, WSDSchedulerBuilder};
pub use memory::{
    find_optimal_params, query_gpu_memory, GpuMemoryInfo, MemoryBudget, MemoryMonitor,
    OptimalTrainingParams,
};
pub use monitor::{TrainingMonitor, ViewMode};
pub use progressive::ProgressiveTrainer;
pub use smoothing::{
    ExponentialMovingAverage, OscillationDamper, SmoothingConfig, SmoothingDiagnostics,
    SmoothingPipeline, SpikeSuppressor,
};
pub use training_config::{
    DataPreset, HybridPreset, ModelPreset, OptimizationPreset, TrainingPreset,
};
pub use screenshot::{
    compare_buffers, create_sample_metrics_file, ComparisonResult, ScreenshotCapture,
    ScreenshotFormat,
};
pub use training_state::{
    calculate_loss_dynamics, capture_git_info, compute_config_hash, CheckpointEvent,
    GeneralizationHealth, GitInfo, LayerGradientStats, LossDynamicsTracker, PhaseTransition,
    StepMetrics, TrainingConfig, TrainingPhase, TrainingRun, TrainingStatus,
};
pub use embeddings::{
    EmbeddingError, EmbeddingProjector, PCA, PCABuilder, ProjectionMethod, ProjectionStats,
    TSNE, TSNEBuilder, UMAP, UMAPBuilder,
};
pub use wordcloud::{
    ClusterConfig, ConceptCluster, ForceDirectedLayout, KMeansClusterer, Layout3D, LayoutConfig,
    RenderConfig, RenderToken, SphericalLayout, Token3D, TokenCloud3D, WordCloudError,
    WordRelationGraph,
};
pub use attention_viz::{
    AttentionHead, AttentionPattern, AttentionViz, ColorMap, Connection, ConnectionType,
    FlowConfig, FlowConnection, FlowDiagram, FlowNode, FlowRenderer, FlowStatistics,
    GradientHealth, HeadAggregation, HeadAnalysis, HeadPatternType, HeatMap, HeatMapConfig,
    HeatMapRenderer, LayerType as AttentionLayerType, LayerViz, NetworkArchViz, NetworkRenderer, NetworkVizConfig,
    OutputFormat, Renderer, VizError, VizResult,
};
pub use viz3d::{
    ArchitectureDiagram, LayerBlock, LayerConnection, LayerInfo, LayerType as ArchLayerType,
    LayoutStyle, SkipConnection, SkipStyle, TritterConfig, isometric_project,
    Color, Colormap, ColormapPreset, CategoricalPalette,
    // Non-Bevy viz3d types
    AnimationPhase, GradientMagnitude, TrainingDataPoint, Viz3dError, Viz3dResult,
    Camera3D, Mesh3D, Vertex3D, Viz3DConfig, Viz3DEngine,
    DenseNetworkViz, DenseVizConfig, DenseLayer, WeightMatrix, NetworkStats,
};

// Bevy-based 3D viewer (requires viz3d feature)
#[cfg(feature = "viz3d")]
pub use viz3d::{
    CameraController, CameraMode, Connection as Viz3dConnection, ConnectionBundle,
    ConnectionRenderer, ConnectionStyle, LayerMesh, LayerMeshBuilder, LayerShape, LayerStyle,
    NetworkConfig, NeuralLayer, OrbitCamera, UiOverlay, UiState, Viz3dApp, Viz3dPlugin,
};

// Training visualization with efficient activation capture
pub use training_viz::{
    // Capture configuration
    CaptureConfig, CaptureConfigBuilder,
    // Step and snapshot types
    StepStats, LayerStats, ActivationSnapshot, LayerActivationSample, AttentionSnapshot,
    TriggerReason,
    // Capture and streaming
    TrainingVizCapture, VizStream, LayerAggregate, CaptureExport,
    // Dashboard presets
    DashboardPreset, PanelConfig, PanelKind,
    // Animation
    AnimationFrame, AnimPhase, AnimationController,
    // Analysis and assessment
    TrainingSummary, HealthStatus, LossAnalysis, GradientAnalysis, LayerAnalysis,
    GradientStats as VizGradientStats, LayerIssue, IssueSeverity,
    AttentionAnalysis, Recommendation, RecommendationPriority, RecommendationCategory,
    DetectedAnomaly, AnomalyType, PerformanceMetrics, TrainingAnalyzer,
};
pub use landscape::{
    DirectionMethod, GradientField, GradientSample, LandscapeConfig, LossLandscape,
    LossSurface, MeshExporter, TrajectoryPoint, TrajectoryTracker,
};

// Rerun visualization (optional feature)
#[cfg(feature = "rerun")]
pub use rerun_viz::{
    AttentionLogger, EmbeddingLogger, LandscapeLogger, LayerInfo as RerunLayerInfo,
    LayerType as RerunLayerType, MetricsLogger, RerunError, RerunLogger, RerunResult,
    TrainingPhaseLog,
};

// Re-export error type even when feature is disabled (for graceful handling)
#[cfg(not(feature = "rerun"))]
pub use rerun_viz::{RerunError, RerunResult};

// Inference visualization
pub use inference_viz::{
    ActivationRecorder, ActivationStats, AttentionGraph, AttentionPattern as InferenceAttentionPattern,
    FlowNode3D, FlowNodeType, FlowRenderer3D, InferenceViz, InferenceVizError, InterpretabilitySummary,
    LayerActivation, LayerHeatmap, LayerRecord, LayerType as InferenceLayerType, ModelConfig as InferenceModelConfig,
    OutputAnalysis, PredictionEntry, RenderConfig as InferenceRenderConfig, TokenFlow,
};

// Adaptive training controller
pub use adaptive::{
    AdaptiveConfig, AdaptiveTrainingController, AdaptedParam, Adaptation,
    AdaptationStrategy, AdaptationHistory, TrainingHealthReport,
    Health, GradientHealth as AdaptiveGradientHealth,
    AdaptiveProgressiveTrainer, AdaptiveProgressiveConfig,
    AdaptiveLoopHelper, AdaptiveUpdate, StepResult as AdaptiveStepResult,
};
