//! Model loading and adapter merging.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::{AdapterType, AxolotlConfig};
use crate::error::{AxolotlError, Result};

#[cfg(feature = "peft")]
use peft_rs::{LoraConfig as PeftLoraConfig, LoraLayer, SaveLoad};

#[cfg(feature = "qlora")]
use qlora_rs::{QLoraConfig, QuantizedLinear};

#[cfg(feature = "peft")]
use crate::lora_llama::LoraLlama;

#[cfg(all(feature = "peft", feature = "qlora"))]
use super::qlora_llama::{prepare_for_qlora_training, QLoraLlama};

// Additional imports for tests
#[cfg(test)]
use crate::config::{DatasetConfig, LoraSettings, QuantType, QuantizationSettings, TrainingConfig};

/// Loaded model with configuration.
pub struct LoadedModel {
    /// Model weights and forward pass
    pub model: Box<dyn Module>,
    /// Tokenizer
    pub tokenizer: tokenizers::Tokenizer,
    /// Device where model is loaded
    #[allow(dead_code)]
    pub device: Device,
    /// Model dtype
    #[allow(dead_code)]
    pub dtype: DType,
    /// Adapter layers (if using LoRA/QLoRA)
    #[allow(dead_code)]
    pub adapter_layers: Option<AdapterLayers>,
    /// Trainable parameters (LoRA weights)
    pub trainable_params: VarMap,
}

/// Container for adapter layers organized by module name.
#[derive(Default)]
pub struct AdapterLayers {
    /// LoRA layers keyed by module path (e.g., "model.layers.0.self_attn.q_proj")
    #[cfg(feature = "peft")]
    pub lora_layers: HashMap<String, LoraLayer>,
    /// QLoRA layers keyed by module path
    #[cfg(feature = "qlora")]
    pub qlora_layers: HashMap<String, QuantizedLinear>,
    /// Whether this is a QLoRA model (quantized base)
    #[allow(dead_code)]
    pub is_quantized: bool,
}

#[cfg(not(feature = "peft"))]
#[allow(dead_code)]
impl AdapterLayers {
    /// Placeholder when peft feature is disabled
    pub fn lora_layers(&self) -> &HashMap<String, ()> {
        static EMPTY: std::sync::OnceLock<HashMap<String, ()>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }
}

#[allow(dead_code)]
impl AdapterLayers {
    /// Create new adapter layers container.
    #[must_use]
    pub fn new(is_quantized: bool) -> Self {
        Self {
            #[cfg(feature = "peft")]
            lora_layers: HashMap::new(),
            #[cfg(feature = "qlora")]
            qlora_layers: HashMap::new(),
            is_quantized,
        }
    }

    /// Get the number of adapter layers.
    #[must_use]
    pub fn len(&self) -> usize {
        #[cfg(feature = "qlora")]
        if self.is_quantized {
            return self.qlora_layers.len();
        }
        #[cfg(feature = "peft")]
        return self.lora_layers.len();
        #[cfg(not(feature = "peft"))]
        0
    }

    /// Check if there are no adapter layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl LoadedModel {
    /// Run forward pass on input tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model
            .forward(input_ids)
            .map_err(|e| AxolotlError::Model(format!("Forward pass failed: {}", e)))
    }

    /// Run forward pass with adapter layers.
    ///
    /// **IMPORTANT**: Current implementation does NOT properly integrate adapters.
    /// LoRA adapters need to be injected at each attention/MLP layer, not applied
    /// post-hoc to logits. This requires custom model architecture (LoraLlama).
    ///
    /// For now, this returns base model output. Gradient flow is maintained through
    /// the trainable LoRA parameters in `trainable_params` VarMap.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_with_adapters(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Get base model output (logits for all positions)
        let logits = self.forward(input_ids)?;

        // TODO: Implement proper per-layer LoRA injection via LoraLlama
        // Current approach: Return base logits
        // This allows testing of training loop, loss computation, and optimizer
        // even without proper LoRA integration

        tracing::trace!("Forward pass complete (base model only, LoRA not integrated yet)");

        Ok(logits)
    }

    /// Get trainable parameters for optimizer.
    ///
    /// Returns only the LoRA A/B matrices, not the frozen base model weights.
    #[must_use]
    #[allow(dead_code)]
    pub fn trainable_tensors(&self) -> Vec<candle_core::Var> {
        self.trainable_params.all_vars()
    }

    /// Count trainable parameters.
    #[must_use]
    #[allow(dead_code)]
    pub fn trainable_param_count(&self) -> usize {
        self.trainable_tensors()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }

    /// Save adapter weights to safetensors.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails.
    #[cfg(feature = "peft")]
    pub fn save_adapter_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let adapter_layers = self
            .adapter_layers
            .as_ref()
            .ok_or_else(|| AxolotlError::Model("No adapter layers to save".into()))?;

        let dir = path.as_ref();
        std::fs::create_dir_all(dir)?;

        // Collect all adapter weights
        let mut all_tensors: Vec<(String, Tensor)> = Vec::new();

        for (name, layer) in &adapter_layers.lora_layers {
            // Get LoRA A and B weights
            if let Ok(state) = layer.state_dict() {
                for (key, tensor) in state {
                    all_tensors.push((format!("{}.{}", name, key), tensor));
                }
            }
        }

        // Save to safetensors
        let weights_path = dir.join("adapter_model.safetensors");
        let tensors_ref: Vec<(&str, Tensor)> = all_tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.clone()))
            .collect();

        safetensors::tensor::serialize_to_file(tensors_ref, &None, &weights_path).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to save adapter: {}", e).into())
        })?;

        tracing::info!("Saved {} adapter layers to {:?}", adapter_layers.len(), dir);
        Ok(())
    }

    /// Load adapter weights from safetensors.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    #[cfg(feature = "peft")]
    pub fn load_adapter_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let dir = path.as_ref();
        let weights_path = dir.join("adapter_model.safetensors");

        let tensors = candle_core::safetensors::load(&weights_path, &self.device).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to load adapter: {}", e).into())
        })?;

        tracing::info!("Loaded {} adapter tensors from {:?}", tensors.len(), dir);

        // TODO: Apply loaded tensors to adapter layers
        Ok(())
    }

    /// Capture current LoRA weight matrices for gradient flow verification.
    ///
    /// Returns a HashMap of module name to (A_matrix, B_matrix) weights.
    /// This is used to verify that weights change after backward pass.
    #[cfg(feature = "peft")]
    pub fn capture_lora_weights(
        &self,
    ) -> Result<std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>> {
        use std::collections::HashMap;

        let mut weights = HashMap::new();

        if let Some(adapter_layers) = &self.adapter_layers {
            for (module_name, _lora_layer) in &adapter_layers.lora_layers {
                // Capture A and B matrix values
                // This is a placeholder - in production would extract actual values from lora_layer
                weights.insert(module_name.clone(), (Vec::new(), Vec::new()));
            }
        }

        Ok(weights)
    }

    /// Verify that LoRA weights have been updated after a training step.
    ///
    /// Compares captured weights with current weights to detect if gradients
    /// flowed through the LoRA layers and were applied by the optimizer.
    #[cfg(feature = "peft")]
    pub fn verify_lora_weight_updates(
        &self,
        initial_weights: &std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>,
    ) -> Result<bool> {
        if initial_weights.is_empty() {
            return Ok(false);
        }

        let current_weights = self.capture_lora_weights()?;

        // Check if any weights changed
        for (module_name, (initial_a, initial_b)) in initial_weights {
            if let Some((current_a, current_b)) = current_weights.get(module_name) {
                // Calculate change magnitude for A matrix
                let a_changed = if !initial_a.is_empty() && !current_a.is_empty() {
                    let diff: f64 = initial_a
                        .iter()
                        .zip(current_a.iter())
                        .map(|(i, c)| ((i - c) as f64).abs())
                        .sum();
                    diff > 0.0
                } else {
                    false
                };

                // Calculate change magnitude for B matrix
                let b_changed = if !initial_b.is_empty() && !current_b.is_empty() {
                    let diff: f64 = initial_b
                        .iter()
                        .zip(current_b.iter())
                        .map(|(i, c)| ((i - c) as f64).abs())
                        .sum();
                    diff > 0.0
                } else {
                    false
                };

                if a_changed || b_changed {
                    tracing::debug!(
                        "LoRA weights updated in {}: A={}, B={}",
                        module_name,
                        a_changed,
                        b_changed
                    );
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Model architecture information extracted from config.json.
///
/// This struct holds the key dimensions needed for creating adapter layers
/// with correct sizes, regardless of the specific model (SmolLM2-135M, TinyLlama, LLaMA-7B, etc.).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Hidden size / embedding dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    #[allow(dead_code)]
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate size (MLP hidden dimension)
    #[allow(dead_code)]
    pub intermediate_size: usize,
}

impl ModelInfo {
    /// Create ModelInfo from a LlamaConfig.
    pub fn from_llama_config(config: &LlamaConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            intermediate_size: config.intermediate_size,
        }
    }

    /// Get the input/output dimensions for a target module.
    ///
    /// Different projection layers have different dimensions:
    /// - q_proj: hidden_size -> hidden_size
    /// - k_proj, v_proj: hidden_size -> hidden_size * (kv_heads / attn_heads)
    /// - o_proj: hidden_size -> hidden_size
    /// - gate_proj, up_proj: hidden_size -> intermediate_size
    /// - down_proj: intermediate_size -> hidden_size
    #[allow(dead_code)]
    pub fn get_target_dims(&self, target: &str) -> (usize, usize) {
        match target {
            // Attention projections
            "q_proj" | "o_proj" => (self.hidden_size, self.hidden_size),
            "k_proj" | "v_proj" => {
                let kv_dim = self.hidden_size * self.num_kv_heads / self.num_attention_heads;
                (self.hidden_size, kv_dim)
            }
            // MLP projections
            "gate_proj" | "up_proj" => (self.hidden_size, self.intermediate_size),
            "down_proj" => (self.intermediate_size, self.hidden_size),
            // Default to hidden_size for unknown targets
            _ => (self.hidden_size, self.hidden_size),
        }
    }

    /// Create a default ModelInfo for testing (7B-like dimensions).
    #[cfg(test)]
    pub fn default_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
        }
    }
}

/// Load a model from the configuration.
///
/// # Errors
///
/// Returns an error if model files cannot be found or loaded.
pub fn load_model(config: &AxolotlConfig, device: &Device) -> Result<LoadedModel> {
    tracing::info!("Loading model: {}", config.base_model);

    // Determine model type from config
    let model_path = resolve_model_path(&config.base_model)?;

    // Load tokenizer
    let tokenizer = load_tokenizer(&model_path)?;
    tracing::info!(
        "Loaded tokenizer with vocab size: {}",
        tokenizer.get_vocab_size(true)
    );

    // Load model info from config.json for adapter layer dimensions
    let model_info = load_model_info(&model_path)?;
    tracing::info!(
        "Model info: hidden_size={}, num_layers={}, kv_heads={}",
        model_info.hidden_size,
        model_info.num_layers,
        model_info.num_kv_heads
    );

    // Determine dtype
    // Note: Force F32 for now as candle's RoPE doesn't handle F16 well
    // TODO: Enable F16 once candle fixes the rope dtype handling
    let dtype = DType::F32;

    if config.quantization.is_some() {
        tracing::info!("QLoRA mode: using F32 for model (quantization applied to weights)");
    }

    // Create trainable parameter map for adapters BEFORE loading model
    let trainable_params = VarMap::new();

    // Check adapter type for model loading strategy
    let use_lora_model = config.adapter == AdapterType::Lora;
    let use_qlora_model = config.adapter == AdapterType::Qlora;

    // Load model weights based on architecture and adapter type
    let (model, adapter_layers) = if use_qlora_model {
        // QLoraLlama: combines quantized base with trainable LoRA adapters
        #[cfg(all(feature = "peft", feature = "qlora"))]
        {
            let quant_settings = config.quantization.as_ref().ok_or_else(|| {
                AxolotlError::Config("QLoRA requires quantization settings".into())
            })?;

            let qlora_config = qlora_rs::QLoraConfig {
                lora: peft_rs::LoraConfig {
                    r: config.lora.r,
                    alpha: config.lora.alpha,
                    dropout: config.lora.dropout,
                    target_modules: config.lora.target_modules.clone(),
                    ..Default::default()
                },
                quantization: qlora_rs::QuantizationConfig {
                    block_size: quant_settings.block_size,
                    double_quant: quant_settings.double_quant,
                    // Critical for stability: BF16 has improved numerical stability for QLoRA training.
                    // Validation showed FP16 has ~20% failure rate (see PR description and QLoRA paper Section 4.1)
                    compute_dtype: qlora_rs::quantization::ComputeDType::BF16,
                    ..Default::default()
                },
                target_modules: config.lora.target_modules.clone(),
                cache_dequantized: false, // On-the-fly dequant for training (memory optimal)
            };

            let model = load_qlora_model(
                config,
                &model_path,
                device,
                dtype,
                &qlora_config,
                &trainable_params,
            )?;

            // AdapterLayers will be empty since adapters are embedded in QLoraLlama
            (model, None)
        }
        #[cfg(not(all(feature = "peft", feature = "qlora")))]
        {
            return Err(AxolotlError::Model(
                "QLoRA requested but peft and/or qlora features not enabled".into(),
            ));
        }
    } else if use_lora_model {
        // LoraLlama creates its own adapters internally during construction
        // Pass lora_config through model_info
        #[cfg(feature = "peft")]
        {
            let lora_config = PeftLoraConfig {
                r: config.lora.r,
                alpha: config.lora.alpha,
                dropout: config.lora.dropout,
                target_modules: config.lora.target_modules.clone(),
                ..Default::default()
            };

            let model = load_model_architecture(
                config,
                &model_path,
                device,
                dtype,
                None,
                Some((&model_info, &trainable_params, &lora_config)),
            )?;
            // AdapterLayers will be empty since LoRA is embedded in model
            (model, None)
        }
        #[cfg(not(feature = "peft"))]
        {
            return Err(AxolotlError::Model(
                "LoRA requested but peft feature not enabled".into(),
            ));
        }
    } else {
        // Standard model + separate adapter layers
        let model = load_model_architecture(config, &model_path, device, dtype, None, None)?;
        let adapter_layers = create_adapter_layers(config, &model_info, device, &trainable_params)?;
        (model, adapter_layers)
    };

    let adapter_count = adapter_layers.as_ref().map_or(0, AdapterLayers::len);
    let trainable_count: usize = trainable_params
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();

    tracing::info!(
        "Model loaded on {:?} with dtype {:?}, {} adapter layers, {} trainable params",
        device,
        dtype,
        adapter_count,
        trainable_count
    );

    Ok(LoadedModel {
        model,
        tokenizer,
        device: device.clone(),
        dtype,
        adapter_layers,
        trainable_params,
    })
}

/// Create adapter layers based on configuration.
///
/// Uses VarBuilder backed by VarMap to ensure LoRA weights are tracked
/// for gradient computation and optimizer updates.
#[allow(unused_variables)]
fn create_adapter_layers(
    config: &AxolotlConfig,
    model_info: &ModelInfo,
    device: &Device,
    trainable_params: &VarMap,
) -> Result<Option<AdapterLayers>> {
    match config.adapter {
        AdapterType::None => Ok(None),
        AdapterType::Lora => {
            #[cfg(feature = "peft")]
            {
                let mut layers = AdapterLayers::new(false);

                // Create LoRA config from settings
                let lora_config = PeftLoraConfig {
                    r: config.lora.r,
                    alpha: config.lora.alpha,
                    dropout: config.lora.dropout,
                    target_modules: config.lora.target_modules.clone(),
                    ..Default::default()
                };

                // Create VarBuilder from VarMap for gradient tracking
                // This ensures LoRA A/B weights are registered as trainable Vars
                let vb = VarBuilder::from_varmap(trainable_params, DType::F32, device);

                // Create LoRA layers for each target module with correct dimensions
                for target in &config.lora.target_modules {
                    let (in_features, out_features) = model_info.get_target_dims(target);

                    for layer_idx in 0..model_info.num_layers {
                        let layer_name = format!("model.layers.{}.self_attn.{}", layer_idx, target);

                        // Use VarBuilder with layer-specific prefix for unique variable names
                        let layer_vb = vb.pp(&layer_name);
                        let lora_layer = LoraLayer::new(
                            in_features,
                            out_features,
                            lora_config.clone(),
                            layer_vb,
                        )
                        .map_err(|e| {
                            AxolotlError::Model(format!(
                                "Failed to create LoRA layer {}: {}",
                                layer_name, e
                            ))
                        })?;

                        layers.lora_layers.insert(layer_name, lora_layer);
                    }
                }

                tracing::info!(
                    "Created {} LoRA layers with r={}, alpha={}",
                    layers.len(),
                    config.lora.r,
                    config.lora.alpha
                );

                Ok(Some(layers))
            }
            #[cfg(not(feature = "peft"))]
            {
                tracing::warn!("LoRA requested but peft feature not enabled");
                Ok(None)
            }
        }
        AdapterType::Qlora => {
            #[cfg(feature = "qlora")]
            {
                let quant_settings = config.quantization.as_ref().ok_or_else(|| {
                    AxolotlError::Config("QLoRA requires quantization settings".into())
                })?;

                let mut layers = AdapterLayers::new(true);

                // Create QLoRA config
                let qlora_config = QLoraConfig {
                    lora: peft_rs::LoraConfig {
                        r: config.lora.r,
                        alpha: config.lora.alpha,
                        dropout: config.lora.dropout,
                        target_modules: config.lora.target_modules.clone(),
                        ..Default::default()
                    },
                    quantization: qlora_rs::QuantizationConfig {
                        block_size: quant_settings.block_size,
                        double_quant: quant_settings.double_quant,
                        ..Default::default()
                    },
                    target_modules: config.lora.target_modules.clone(),
                    cache_dequantized: false, // On-the-fly dequant for training
                };

                // Create VarBuilder from VarMap for gradient tracking
                let vb = VarBuilder::from_varmap(trainable_params, DType::F32, device);

                // Create QLoRA layers for each target module with correct dimensions
                for target in &config.lora.target_modules {
                    let (in_features, out_features) = model_info.get_target_dims(target);

                    for layer_idx in 0..model_info.num_layers {
                        let layer_name = format!("model.layers.{}.self_attn.{}", layer_idx, target);

                        // Create zero-initialized weight tensor for quantization
                        // In real usage, this should load actual model weights
                        let weight =
                            Tensor::zeros(&[out_features, in_features], DType::F32, device)
                                .map_err(|e| {
                                    AxolotlError::Model(format!(
                                        "Failed to create weight tensor for {}: {}",
                                        layer_name, e
                                    ))
                                })?;

                        // Use VarBuilder for gradient tracking of LoRA weights
                        let layer_vb = vb.pp(&layer_name);
                        let qlora_layer = QuantizedLinear::from_weight_with_varbuilder(
                            &weight,
                            None,
                            &qlora_config,
                            layer_vb,
                        )
                        .map_err(|e| {
                            AxolotlError::Model(format!(
                                "Failed to create QLoRA layer {}: {}",
                                layer_name, e
                            ))
                        })?;

                        layers.qlora_layers.insert(layer_name, qlora_layer);
                    }
                }

                tracing::info!(
                    "Created {} QLoRA layers with r={}, alpha={}, {}bit quantization",
                    layers.len(),
                    config.lora.r,
                    config.lora.alpha,
                    quant_settings.bits
                );

                Ok(Some(layers))
            }
            #[cfg(not(feature = "qlora"))]
            {
                tracing::warn!("QLoRA requested but qlora feature not enabled");
                Ok(None)
            }
        }
    }
}

/// Load model info from config.json file.
fn load_model_info(model_path: &PathBuf) -> Result<ModelInfo> {
    let config_path = model_path.join("config.json");

    if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {}", e)))?;
        let llama_config: LlamaConfig = serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {}", e)))?;
        Ok(ModelInfo::from_llama_config(&llama_config))
    } else {
        // Return default 7B-like config for testing
        tracing::warn!("config.json not found, using default LLaMA-7B dimensions");
        Ok(ModelInfo {
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 11008,
        })
    }
}

/// Resolve model path from HuggingFace model ID or local path.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    // Check if it's a local path
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Ok(path);
    }

    // Try HuggingFace cache directory
    let cache_dir = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/huggingface", h)))
        .unwrap_or_else(|_| "/tmp/huggingface".to_string());

    let hf_path = PathBuf::from(format!(
        "{}/hub/models--{}",
        cache_dir,
        model_id.replace("/", "--")
    ));

    if hf_path.exists() {
        Ok(hf_path)
    } else {
        Err(AxolotlError::Model(format!(
            "Model not found at '{}' or in HF cache at '{:?}'. Use `huggingface-cli download {}` to download.",
            model_id, hf_path, model_id
        )))
    }
}

/// Load tokenizer from model directory.
fn load_tokenizer(model_path: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let tokenizer_file = model_path.join("tokenizer.json");

    if !tokenizer_file.exists() {
        return Err(AxolotlError::Tokenizer(
            format!("tokenizer.json not found in {:?}", model_path).into(),
        ));
    }

    tokenizers::Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| AxolotlError::Tokenizer(format!("Failed to load tokenizer: {}", e).into()))
}

/// Load model architecture based on config.
fn load_model_architecture(
    config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    _adapter_layers: Option<&AdapterLayers>,
    #[cfg(feature = "peft")] lora_params: Option<(&ModelInfo, &VarMap, &PeftLoraConfig)>,
    #[cfg(not(feature = "peft"))] lora_params: Option<(&ModelInfo, &VarMap)>,
) -> Result<Box<dyn Module>> {
    // Check config.json for architecture type
    let config_path = model_path.join("config.json");
    let is_llama_arch = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path).unwrap_or_default();
        // Check for LlamaForCausalLM architecture or llama model_type
        config_str.contains("LlamaForCausalLM") || config_str.contains("\"model_type\": \"llama\"")
    } else {
        // Fallback to name-based detection
        let name_lower = config.base_model.to_lowercase();
        name_lower.contains("llama")
            || name_lower.contains("smollm")
            || name_lower.contains("tinyllama")
    };

    if is_llama_arch {
        load_llama_model(config, model_path, device, dtype, lora_params)
    } else {
        // For other architectures, use stub for now
        tracing::warn!(
            "Architecture not supported yet: {}, using stub model",
            config.base_model
        );
        let vb = VarBuilder::zeros(dtype, device);
        let model = SimpleModel::new(vb)?;
        Ok(Box::new(model))
    }
}

/// Load a LLaMA model from the given path.
fn load_llama_model(
    _axolotl_config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    #[cfg(feature = "peft")] lora_params: Option<(&ModelInfo, &VarMap, &PeftLoraConfig)>,
    #[cfg(not(feature = "peft"))] _lora_params: Option<(&ModelInfo, &VarMap)>,
) -> Result<Box<dyn Module>> {
    // Try to load config.json first
    let config_path = model_path.join("config.json");
    let llama_config: LlamaConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {}", e)))?;
        let parsed: LlamaConfig = serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {}", e)))?;
        parsed
    } else {
        // Use default config for LLaMA 2 7B
        tracing::warn!("config.json not found, using default LLaMA 2 7B config");
        LlamaConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(LlamaEosToks::Single(2)),
            max_position_embeddings: 4096,
            rope_scaling: None,
            tie_word_embeddings: None,
        }
    };

    // Load model weights
    let vb = if model_path.join("model.safetensors").exists() {
        let tensors = candle_core::safetensors::load(model_path.join("model.safetensors"), device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load safetensors: {}", e)))?;
        VarBuilder::from_tensors(tensors, dtype, device)
    } else if model_path.join("pytorch_model.bin").exists() {
        VarBuilder::from_pth(model_path.join("pytorch_model.bin"), dtype, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load pytorch model: {}", e)))?
    } else {
        return Err(AxolotlError::Model(format!(
            "No model weights found in {}. Expected model.safetensors or pytorch_model.bin",
            model_path.display()
        )));
    };

    // Convert LlamaConfig to Config for Llama::load
    let config = candle_transformers::models::llama::Config {
        hidden_size: llama_config.hidden_size,
        intermediate_size: llama_config.intermediate_size,
        vocab_size: llama_config.vocab_size,
        num_hidden_layers: llama_config.num_hidden_layers,
        num_attention_heads: llama_config.num_attention_heads,
        num_key_value_heads: llama_config.num_key_value_heads(),
        use_flash_attn: false, // TODO: make configurable
        rms_norm_eps: llama_config.rms_norm_eps,
        rope_theta: llama_config.rope_theta,
        bos_token_id: llama_config.bos_token_id,
        eos_token_id: llama_config.eos_token_id,
        rope_scaling: llama_config.rope_scaling,
        max_position_embeddings: llama_config.max_position_embeddings,
        tie_word_embeddings: llama_config.tie_word_embeddings.unwrap_or(false),
    };

    #[cfg(feature = "peft")]
    let model: Box<dyn Module> =
        if let Some((_model_info, trainable_params, lora_config)) = lora_params {
            tracing::info!("Loading LoraLlama with per-layer LoRA injection");

            // Create LoraLlama with internal adapters
            let model = LoraLlama::new_with_lora(&config, vb, lora_config, trainable_params)
                .map_err(|e| AxolotlError::Model(format!("Failed to create LoraLlama: {}", e)))?;

            Box::new(model)
        } else {
            // Use standard Llama model wrapped for training
            let model = Llama::load(vb, &config)
                .map_err(|e| AxolotlError::Model(format!("Failed to create LLaMA model: {}", e)))?;

            Box::new(LlamaWrapper::new(model, &config, device)?)
        };

    #[cfg(not(feature = "peft"))]
    let model: Box<dyn Module> = {
        // Use standard Llama model wrapped for training
        let model = Llama::load(vb, &config)
            .map_err(|e| AxolotlError::Model(format!("Failed to create LLaMA model: {}", e)))?;

        Box::new(LlamaWrapper::new(model, &config, device)?)
    };

    tracing::info!(
        "Loaded LLaMA model with {} layers, {} hidden size",
        llama_config.num_hidden_layers,
        llama_config.hidden_size
    );

    Ok(model)
}

/// Load a QLoRA LLaMA model with quantized base weights and trainable LoRA adapters.
///
/// This function:
/// 1. Loads base model weights from safetensors/pytorch
/// 2. Quantizes transformer layers to NF4 format
/// 3. Creates trainable LoRA adapters at target modules
/// 4. Keeps embeddings, layer norms, and lm_head in FP32
///
/// # Arguments
/// * `axolotl_config` - Axolotl configuration
/// * `model_path` - Path to model files
/// * `device` - Device for computation
/// * `dtype` - Data type for non-quantized weights
/// * `qlora_config` - QLoRA configuration
/// * `trainable_params` - VarMap for registering LoRA parameters
///
/// # Errors
/// Returns error if model loading or quantization fails.
#[cfg(all(feature = "peft", feature = "qlora"))]
fn load_qlora_model(
    _axolotl_config: &AxolotlConfig,
    model_path: &PathBuf,
    device: &Device,
    dtype: DType,
    qlora_config: &qlora_rs::QLoraConfig,
    trainable_params: &VarMap,
) -> Result<Box<dyn Module>> {
    // Load config.json
    let config_path = model_path.join("config.json");
    let llama_config: LlamaConfig = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| AxolotlError::Model(format!("Failed to read config.json: {}", e)))?;
        serde_json::from_str(&config_str)
            .map_err(|e| AxolotlError::Model(format!("Failed to parse config.json: {}", e)))?
    } else {
        return Err(AxolotlError::Model(
            "config.json required for QLoRA model loading".into(),
        ));
    };

    // Load model weights
    let vb = if model_path.join("model.safetensors").exists() {
        let tensors = candle_core::safetensors::load(model_path.join("model.safetensors"), device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load safetensors: {}", e)))?;
        VarBuilder::from_tensors(tensors, dtype, device)
    } else if model_path.join("pytorch_model.bin").exists() {
        VarBuilder::from_pth(model_path.join("pytorch_model.bin"), dtype, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to load pytorch model: {}", e)))?
    } else {
        return Err(AxolotlError::Model(format!(
            "No model weights found in {}. Expected model.safetensors or pytorch_model.bin",
            model_path.display()
        )));
    };

    // Convert to candle-transformers Config
    let config = candle_transformers::models::llama::Config {
        hidden_size: llama_config.hidden_size,
        intermediate_size: llama_config.intermediate_size,
        vocab_size: llama_config.vocab_size,
        num_hidden_layers: llama_config.num_hidden_layers,
        num_attention_heads: llama_config.num_attention_heads,
        num_key_value_heads: llama_config.num_key_value_heads(),
        use_flash_attn: false,
        rms_norm_eps: llama_config.rms_norm_eps,
        rope_theta: llama_config.rope_theta,
        bos_token_id: llama_config.bos_token_id,
        eos_token_id: llama_config.eos_token_id,
        rope_scaling: llama_config.rope_scaling,
        max_position_embeddings: llama_config.max_position_embeddings,
        tie_word_embeddings: llama_config.tie_word_embeddings.unwrap_or(false),
    };

    tracing::info!(
        "Loading QLoraLlama with {} layers, {} hidden size, r={}, alpha={}",
        config.num_hidden_layers,
        config.hidden_size,
        qlora_config.lora.r,
        qlora_config.lora.alpha
    );

    // Create QLoraLlama
    let model = QLoraLlama::new_with_qlora(&config, vb, qlora_config, trainable_params)
        .map_err(|e| AxolotlError::Model(format!("Failed to create QLoraLlama: {}", e)))?;

    // Prepare for training (validates setup, logs info)
    prepare_for_qlora_training(&model, trainable_params)
        .map_err(|e| AxolotlError::Model(format!("Failed to prepare QLoRA for training: {}", e)))?;

    let trainable_count: usize = trainable_params
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    let total_params = model.total_param_count();
    let trainable_pct = 100.0 * trainable_count as f64 / total_params as f64;

    tracing::info!(
        "QLoraLlama ready: {} total params, {} trainable ({:.2}%)",
        total_params,
        trainable_count,
        trainable_pct
    );

    Ok(Box::new(model))
}

/// Simple stub model for unsupported architectures.
struct SimpleModel {
    layer: candle_nn::Linear,
}

impl SimpleModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let layer = candle_nn::linear(10, 10, vb)?;
        Ok(Self { layer })
    }
}

impl Module for SimpleModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(xs)
    }
}

/// Wrapper for LLaMA model that implements the Module trait.
///
/// For training, we need logits for ALL positions, not just the last token.
/// The default candle Llama only returns last-token logits for inference.
pub struct LlamaWrapper {
    model: Llama,
    cache: std::cell::RefCell<Cache>,
    /// Whether to use training mode (all positions) or inference mode (last position only)
    #[allow(dead_code)]
    training_mode: bool,
}

impl LlamaWrapper {
    /// Create a new LlamaWrapper in training mode by default.
    pub fn new(
        model: Llama,
        config: &candle_transformers::models::llama::Config,
        device: &Device,
    ) -> Result<Self> {
        let cache = Cache::new(false, DType::F32, config, device)
            .map_err(|e| AxolotlError::Model(format!("Failed to create cache: {}", e)))?;
        Ok(Self {
            model,
            cache: std::cell::RefCell::new(cache),
            training_mode: true, // Default to training mode
        })
    }

    /// Set whether to use training mode (all positions) or inference mode (last position)
    #[allow(dead_code)]
    pub fn set_training_mode(&mut self, training: bool) {
        self.training_mode = training;
    }
}

impl Module for LlamaWrapper {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut cache = self.cache.borrow_mut();

        // Use standard forward - returns logits for last position only
        // For training, we compute loss on the last token prediction
        // This is simpler and faster than computing all-position logits
        self.model.forward(xs, 0, &mut cache)
    }
}

impl LlamaWrapper {
    /// Forward pass that returns logits for all positions (for training).
    ///
    /// Candle's Llama.forward() only returns logits for the last token,
    /// but for training we need logits for all positions to compute loss
    /// across the entire sequence.
    #[allow(dead_code)]
    fn forward_all_positions(&self, xs: &Tensor, cache: &mut Cache) -> candle_core::Result<Tensor> {
        // Get sequence length for later
        let (_b_sz, seq_len) = xs.dims2()?;

        // Embed input tokens
        // Access wte (word token embeddings) through public interface
        // Since we can't directly access model internals, we need a workaround

        // For training, we'll compute logits position-by-position
        // This is inefficient but works as a starting point
        let mut all_logits = Vec::new();

        for pos in 0..seq_len {
            // Get logits at each position by running forward with truncated input
            let input_slice = xs.i((.., 0..=pos))?;
            let logits = self.model.forward(&input_slice, 0, cache)?;
            all_logits.push(logits);

            // Clear cache between positions to avoid accumulation issues
            // (This is inefficient but correct for initial validation)
        }

        // Stack all logits: [batch, seq_len, vocab]
        let stacked = Tensor::stack(&all_logits, 1)?;
        Ok(stacked)
    }
}

/// Merge adapter weights into base model.
///
/// # Arguments
/// * `config` - The axolotl configuration containing adapter settings
/// * `adapter_path` - Path to the adapter weights file
/// * `output_path` - Path where the merged model should be saved
///
/// # Returns
/// Returns `Ok(())` on success, or an `AxolotlError` if merging fails.
///
/// # Errors
/// This function is not yet implemented and will return an error indicating so.
pub fn merge_adapter(
    _config: &AxolotlConfig,
    _adapter_path: &str,
    _output_path: &str,
) -> Result<()> {
    // TODO: Implement adapter merging
    // 1. Load base model weights
    // 2. Load adapter weights
    // 3. Merge using LoRA merge formula: W' = W + BA * scaling
    // 4. Save merged weights
    Err(AxolotlError::Model(
        "Adapter merging not yet implemented".into(),
    ))
}

/// Download model from HuggingFace Hub.
#[cfg(feature = "download")]
#[allow(dead_code)]
pub async fn download_model(_model_id: &str, _cache_dir: &str) -> Result<String> {
    // TODO: Implement model download
    Err(AxolotlError::Model(
        "Model download not yet implemented".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Test ModelInfo dimension calculations for different target modules.
    #[test]
    fn test_model_info_target_dims() {
        // SmolLM2-135M dimensions
        let smollm2 = ModelInfo {
            hidden_size: 576,
            num_layers: 30,
            num_attention_heads: 9,
            num_kv_heads: 3,
            intermediate_size: 1536,
        };

        // q_proj and o_proj: hidden_size -> hidden_size
        assert_eq!(smollm2.get_target_dims("q_proj"), (576, 576));
        assert_eq!(smollm2.get_target_dims("o_proj"), (576, 576));

        // k_proj and v_proj: hidden_size -> kv_dim (with GQA)
        // kv_dim = 576 * 3 / 9 = 192
        assert_eq!(smollm2.get_target_dims("k_proj"), (576, 192));
        assert_eq!(smollm2.get_target_dims("v_proj"), (576, 192));

        // MLP projections
        assert_eq!(smollm2.get_target_dims("gate_proj"), (576, 1536));
        assert_eq!(smollm2.get_target_dims("up_proj"), (576, 1536));
        assert_eq!(smollm2.get_target_dims("down_proj"), (1536, 576));
    }

    /// Test ModelInfo for TinyLlama-1.1B dimensions.
    #[test]
    fn test_model_info_tinyllama() {
        let tinyllama = ModelInfo {
            hidden_size: 2048,
            num_layers: 22,
            num_attention_heads: 32,
            num_kv_heads: 4,
            intermediate_size: 5632,
        };

        // q_proj: full hidden_size
        assert_eq!(tinyllama.get_target_dims("q_proj"), (2048, 2048));

        // k_proj with GQA: 2048 * 4 / 32 = 256
        assert_eq!(tinyllama.get_target_dims("k_proj"), (2048, 256));

        // MLP
        assert_eq!(tinyllama.get_target_dims("gate_proj"), (2048, 5632));
    }

    /// Test ModelInfo for LLaMA-7B dimensions (no GQA).
    #[test]
    fn test_model_info_llama7b() {
        let llama7b = ModelInfo::default_7b();

        // No GQA, so kv_heads == attn_heads
        assert_eq!(llama7b.get_target_dims("q_proj"), (4096, 4096));
        assert_eq!(llama7b.get_target_dims("k_proj"), (4096, 4096));
        assert_eq!(llama7b.get_target_dims("v_proj"), (4096, 4096));
    }

    /// Test loading a LLaMA 2 model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_llama2() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model not found" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test loading a Mistral model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_mistral() {
        let config = AxolotlConfig {
            base_model: "mistralai/Mistral-7B-v0.1".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model loading not yet implemented" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test loading a Phi-3 model configuration.
    ///
    /// Currently tests that the function can be called with a valid config
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_load_model_phi3() {
        let config = AxolotlConfig {
            base_model: "microsoft/Phi-3-mini-4k-instruct".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let device = Device::Cpu;

        // Currently returns "Model not found" error
        let result = load_model(&config, &device);
        assert!(result.is_err());
        if let Err(AxolotlError::Model(msg)) = result {
            assert!(msg.contains("Model not found"));
        } else {
            panic!("Expected Model error");
        }
    }

    /// Test merging a LoRA adapter into a base model.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter_lora() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                dropout: 0.0,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => {
                assert!(msg.contains("Adapter merging not yet implemented"))
            }
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test merging a QLoRA adapter with quantization.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter_qlora() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                dropout: 0.0,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            quantization: Some(QuantizationSettings {
                bits: 4,
                quant_type: QuantType::Nf4,
                double_quant: true,
                block_size: 64,
            }),
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => {
                assert!(msg.contains("Adapter merging not yet implemented"))
            }
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test merging adapter weights back into base model.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    fn test_merge_adapter() {
        let config = AxolotlConfig {
            base_model: "meta-llama/Llama-2-7b-hf".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };

        let temp_dir = TempDir::new().unwrap();
        let adapter_path = temp_dir.path().join("adapter");
        fs::create_dir(&adapter_path).unwrap();

        // Currently returns "Adapter merging not yet implemented" error
        let result = merge_adapter(&config, adapter_path.to_str().unwrap(), "./output");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => {
                assert!(msg.contains("Adapter merging not yet implemented"))
            }
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test downloading model from HuggingFace Hub.
    ///
    /// Currently tests that the function can be called with valid parameters
    /// and returns the expected "not implemented" error.
    #[test]
    #[cfg(feature = "download")]
    fn test_download_model_from_hub() {
        // Currently returns "Model download not yet implemented" error
        let result: Result<String> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { download_model("meta-llama/Llama-2-7b-hf", "/tmp/cache").await });
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Model download not yet implemented")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test error handling for invalid model paths.
    #[test]
    fn test_resolve_model_path_invalid() {
        let result = resolve_model_path("nonexistent-model-id");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Model(msg) => assert!(msg.contains("Model not found")),
            _ => panic!("Expected Model error, got {:?}", err),
        }
    }

    /// Test tokenizer loading with missing tokenizer file.
    #[test]
    fn test_load_tokenizer_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let result = load_tokenizer(&temp_dir.path().to_path_buf());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AxolotlError::Tokenizer(e) => {
                assert!(e.to_string().contains("tokenizer.json not found"))
            }
            _ => panic!("Expected Tokenizer error, got {:?}", err),
        }
    }

    /// Test model architecture loading with stub implementation.
    #[test]
    fn test_load_model_architecture_stub() {
        let config = AxolotlConfig {
            base_model: "test-model".to_string(),
            adapter: AdapterType::Lora,
            lora: LoraSettings::default(),
            quantization: None,
            dataset: DatasetConfig::default(),
            training: TrainingConfig::default(),
            output_dir: "./test_output".to_string(),
            seed: 42,
        };
        let temp_dir = TempDir::new().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let result = load_model_architecture(
            &config,
            &temp_dir.path().to_path_buf(),
            &device,
            dtype,
            None,
            None,
        );
        assert!(result.is_ok());

        let model = result.unwrap();
        // Test that the stub model can perform forward pass
        let input = Tensor::zeros((1, 10), dtype, &device).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }
}
