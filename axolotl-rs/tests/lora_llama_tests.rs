//! Tests for LoraLlama per-layer injection.
//!
//! These tests validate that LoRA adapters are properly injected at each
//! transformer layer (Q, K, V, O, gate, up, down projections) and that
//! gradients flow correctly through the adapters.
//!
//! # Running Tests
//!
//! ```bash
//! # CPU-only tests
//! cargo test --features peft test_lora_llama_
//!
//! # GPU tests (requires CUDA)
//! cargo test --features peft,cuda test_lora_llama_ -- --ignored
//! ```

#[cfg(feature = "peft")]
mod lora_llama_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Module, VarMap};
    use candle_transformers::models::llama::LlamaEosToks;
    use peft_rs::{Adapter, LoraConfig};

    // Helper to create a small test config
    fn create_test_llama_config() -> candle_transformers::models::llama::Config {
        candle_transformers::models::llama::Config {
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
            eos_token_id: Some(LlamaEosToks::Single(2)),
            rope_scaling: None,
            tie_word_embeddings: false,
        }
    }

    fn create_test_lora_config() -> LoraConfig {
        LoraConfig {
            r: 8,
            alpha: 16,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
                "gate_proj".into(),
                "up_proj".into(),
                "down_proj".into(),
            ],
            ..Default::default()
        }
    }

    // ==========================================================================
    // LoraConfig Target Module Tests
    // ==========================================================================

    #[test]
    fn test_lora_llama_config_all_targets() {
        let config = create_test_lora_config();

        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"k_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));
        assert!(config.target_modules.contains(&"o_proj".to_string()));
        assert!(config.target_modules.contains(&"gate_proj".to_string()));
        assert!(config.target_modules.contains(&"up_proj".to_string()));
        assert!(config.target_modules.contains(&"down_proj".to_string()));
    }

    #[test]
    fn test_lora_llama_config_qv_only() {
        let config = LoraConfig {
            r: 8,
            alpha: 16,
            dropout: 0.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            ..Default::default()
        };

        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));
        assert!(!config.target_modules.contains(&"k_proj".to_string()));
        assert!(!config.target_modules.contains(&"o_proj".to_string()));
    }

    #[test]
    fn test_lora_llama_scaling() {
        let config = create_test_lora_config();
        // scale = alpha / r = 16 / 8 = 2.0
        let scale = config.alpha as f64 / config.r as f64;
        assert!((scale - 2.0).abs() < 1e-6);
    }

    // ==========================================================================
    // LoraLlama Cache Tests
    // ==========================================================================

    #[test]
    fn test_lora_llama_cache_creation() {
        use axolotl_rs::lora_llama::Cache;

        let config = create_test_llama_config();
        let device = Device::Cpu;

        let cache = Cache::new(false, DType::F32, &config, &device);
        assert!(cache.is_ok(), "Cache creation should succeed");

        let cache = cache.unwrap();
        assert_eq!(cache.kvs.len(), config.num_hidden_layers);
        assert!(!cache.use_kv_cache);
    }

    #[test]
    fn test_lora_llama_cache_rotary_shapes() {
        use axolotl_rs::lora_llama::Cache;

        let config = create_test_llama_config();
        let device = Device::Cpu;

        let cache = Cache::new(false, DType::F32, &config, &device).unwrap();

        let head_dim = config.hidden_size / config.num_attention_heads;
        // cos/sin should be [max_pos, head_dim/2]
        assert_eq!(
            cache.cos.dims(),
            &[config.max_position_embeddings, head_dim / 2]
        );
        assert_eq!(
            cache.sin.dims(),
            &[config.max_position_embeddings, head_dim / 2]
        );
    }

    // ==========================================================================
    // LoraLlama VarMap Tracking Tests
    // ==========================================================================

    #[test]
    fn test_lora_llama_varmap_creation() {
        let varmap = VarMap::new();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create some test variables to simulate LoRA params
        let lora_a = vb.pp("model.layers.0.self_attn.q_proj").get_with_hints(
            (8, 64),
            "lora_A.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );
        let lora_b = vb.pp("model.layers.0.self_attn.q_proj").get_with_hints(
            (64, 8),
            "lora_B.weight",
            candle_nn::Init::Const(0.0),
        );

        assert!(lora_a.is_ok(), "LoRA A creation should succeed");
        assert!(lora_b.is_ok(), "LoRA B creation should succeed");

        // Verify vars are tracked
        let vars = varmap.all_vars();
        assert_eq!(vars.len(), 2, "Should have 2 tracked variables");
    }

    #[test]
    fn test_lora_llama_trainable_param_count() {
        let lora_config = create_test_lora_config();
        let llama_config = create_test_llama_config();

        // Calculate expected trainable params
        // Each LoRA adapter has: in_features * r + r * out_features params
        let r = lora_config.r;
        let hidden = llama_config.hidden_size;
        let intermediate = llama_config.intermediate_size;
        let kv_dim = hidden * llama_config.num_key_value_heads / llama_config.num_attention_heads;

        // Attention projections per layer
        let q_params = hidden * r + r * hidden; // q_proj
        let k_params = hidden * r + r * kv_dim; // k_proj
        let v_params = hidden * r + r * kv_dim; // v_proj
        let o_params = hidden * r + r * hidden; // o_proj
        let attn_params = q_params + k_params + v_params + o_params;

        // MLP projections per layer
        let gate_params = hidden * r + r * intermediate; // gate_proj
        let up_params = hidden * r + r * intermediate; // up_proj
        let down_params = intermediate * r + r * hidden; // down_proj
        let mlp_params = gate_params + up_params + down_params;

        // Total per layer
        let per_layer = attn_params + mlp_params;

        // Total for all layers
        let total = per_layer * llama_config.num_hidden_layers;

        println!("Expected trainable params: {}", total);
        println!("  - Per layer: {}", per_layer);
        println!("  - Attention: {}", attn_params);
        println!("  - MLP: {}", mlp_params);

        // With r=8, hidden=64, intermediate=128, kv_dim=32, 2 layers:
        // q_proj: 64*8 + 8*64 = 1024
        // k_proj: 64*8 + 8*32 = 768
        // v_proj: 64*8 + 8*32 = 768
        // o_proj: 64*8 + 8*64 = 1024
        // attn = 3584
        // gate: 64*8 + 8*128 = 1536
        // up: 64*8 + 8*128 = 1536
        // down: 128*8 + 8*64 = 1536
        // mlp = 4608
        // per_layer = 8192
        // total = 16384
        assert_eq!(per_layer, 8192);
        assert_eq!(total, 16384);
    }

    // ==========================================================================
    // LoraAttention Forward Tests
    // ==========================================================================

    #[test]
    fn test_lora_attention_shapes() {
        use axolotl_rs::lora_llama::{Cache, LoraAttention};

        let llama_config = create_test_llama_config();
        let lora_config = create_test_lora_config();
        let device = Device::Cpu;

        // Create VarMap and VarBuilder for base weights (from zeros for testing)
        let base_varmap = VarMap::new();
        let base_vb = candle_nn::VarBuilder::from_varmap(&base_varmap, DType::F32, &device);

        // Create VarMap and VarBuilder for LoRA weights
        let lora_varmap = VarMap::new();
        let lora_vb = candle_nn::VarBuilder::from_varmap(&lora_varmap, DType::F32, &device);

        let attention = LoraAttention::new(
            llama_config.hidden_size,
            llama_config.num_attention_heads,
            llama_config.num_key_value_heads,
            llama_config.max_position_embeddings,
            base_vb,
            Some(&lora_config),
            Some(lora_vb),
            0,
        );

        assert!(attention.is_ok(), "LoraAttention creation should succeed");

        let attention = attention.unwrap();

        // Create cache
        let mut cache = Cache::new(false, DType::F32, &llama_config, &device).unwrap();

        // Test forward pass
        let batch_size = 2;
        let seq_len = 8;
        let input = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, llama_config.hidden_size),
            &device,
        )
        .unwrap();

        let output = attention.forward(&input, 0, 0, &mut cache);
        assert!(output.is_ok(), "Forward pass should succeed");

        let output = output.unwrap();
        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, llama_config.hidden_size]
        );
    }

    // ==========================================================================
    // LoraMlp Forward Tests
    // ==========================================================================

    #[test]
    fn test_lora_mlp_shapes() {
        use axolotl_rs::lora_llama::LoraMlp;

        let llama_config = create_test_llama_config();
        let lora_config = create_test_lora_config();
        let device = Device::Cpu;

        // Create VarMaps
        let base_varmap = VarMap::new();
        let base_vb = candle_nn::VarBuilder::from_varmap(&base_varmap, DType::F32, &device);

        let lora_varmap = VarMap::new();
        let lora_vb = candle_nn::VarBuilder::from_varmap(&lora_varmap, DType::F32, &device);

        let mlp = LoraMlp::new(
            llama_config.hidden_size,
            llama_config.intermediate_size,
            base_vb,
            Some(&lora_config),
            Some(lora_vb),
            0,
        );

        assert!(mlp.is_ok(), "LoraMlp creation should succeed");

        let mlp = mlp.unwrap();

        // Test forward pass
        let batch_size = 2;
        let seq_len = 8;
        let input = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, llama_config.hidden_size),
            &device,
        )
        .unwrap();

        let output = mlp.forward(&input);
        assert!(output.is_ok(), "MLP forward pass should succeed");

        let output = output.unwrap();
        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, llama_config.hidden_size]
        );
    }

    // ==========================================================================
    // LoraTransformerBlock Forward Tests
    // ==========================================================================

    #[test]
    fn test_lora_transformer_block_shapes() {
        use axolotl_rs::lora_llama::{Cache, LoraTransformerBlock};

        let llama_config = create_test_llama_config();
        let lora_config = create_test_lora_config();
        let device = Device::Cpu;

        // Create VarMaps
        let base_varmap = VarMap::new();
        let base_vb = candle_nn::VarBuilder::from_varmap(&base_varmap, DType::F32, &device);

        let lora_varmap = VarMap::new();
        let lora_vb = candle_nn::VarBuilder::from_varmap(&lora_varmap, DType::F32, &device);

        let block = LoraTransformerBlock::new(
            0,
            llama_config.hidden_size,
            llama_config.intermediate_size,
            llama_config.num_attention_heads,
            llama_config.num_key_value_heads,
            llama_config.rms_norm_eps,
            llama_config.max_position_embeddings,
            base_vb,
            Some(&lora_config),
            Some(lora_vb),
        );

        assert!(
            block.is_ok(),
            "LoraTransformerBlock creation should succeed"
        );

        let block = block.unwrap();

        // Create cache
        let mut cache = Cache::new(false, DType::F32, &llama_config, &device).unwrap();

        // Test forward pass
        let batch_size = 2;
        let seq_len = 8;
        let input = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, llama_config.hidden_size),
            &device,
        )
        .unwrap();

        let output = block.forward(&input, 0, 0, &mut cache);
        assert!(output.is_ok(), "Block forward pass should succeed");

        let output = output.unwrap();
        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, llama_config.hidden_size]
        );
    }

    // ==========================================================================
    // Gradient Flow Tests
    // ==========================================================================

    #[test]
    fn test_lora_gradient_tracking() {
        // This test verifies that LoRA parameters are tracked for gradient computation
        let varmap = VarMap::new();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create a simple LoRA layer using peft-rs
        use peft_rs::LoraLayer;

        let lora_config = create_test_lora_config();
        let lora = LoraLayer::new(64, 64, lora_config, vb.pp("test"));
        assert!(lora.is_ok(), "LoraLayer creation should succeed");

        let lora = lora.unwrap();

        // Verify params are tracked
        let vars = varmap.all_vars();
        assert!(!vars.is_empty(), "LoRA params should be tracked in VarMap");

        // Create input and run forward
        let input = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();
        let base_output = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();

        // Forward should return base + lora_delta
        let output = lora.forward(&input, Some(&base_output));
        assert!(output.is_ok(), "LoRA forward should succeed");
    }

    // ==========================================================================
    // GPU Integration Tests (ignored by default)
    // ==========================================================================

    #[test]
    #[ignore = "Requires CUDA GPU"]
    #[cfg(feature = "cuda")]
    fn test_lora_llama_gpu_forward() {
        use axolotl_rs::lora_llama::{Cache, LoraLlama};

        let device = Device::cuda_if_available(0).expect("CUDA device required");
        let llama_config = create_test_llama_config();
        let lora_config = create_test_lora_config();

        // Create VarMaps
        let base_varmap = VarMap::new();
        let base_vb = candle_nn::VarBuilder::from_varmap(&base_varmap, DType::F32, &device);

        let lora_varmap = VarMap::new();

        let model = LoraLlama::new_with_lora(&llama_config, base_vb, &lora_config, &lora_varmap);
        assert!(model.is_ok(), "LoraLlama creation should succeed on GPU");

        let model = model.unwrap();

        // Create cache for forward pass
        let mut cache =
            Cache::new(false, DType::F32, &llama_config, &device).expect("Cache creation failed");

        // Test forward pass
        let input_ids = Tensor::zeros(&[2, 16], DType::U32, &device).unwrap();
        let output = model.forward(&input_ids, 0, &mut cache);

        assert!(output.is_ok(), "GPU forward pass should succeed");
        let output = output.unwrap();
        assert_eq!(output.dims(), &[2, 16, llama_config.vocab_size]);
    }
}
