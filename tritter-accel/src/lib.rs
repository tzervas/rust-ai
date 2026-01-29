//! Rust acceleration for AI training and inference.
//!
//! `tritter-accel` provides high-performance operations for both ternary
//! (BitNet-style) and conventional neural network workloads. It serves as
//! an acceleration layer that can be used from either Rust or Python.
//!
//! # Architecture
//!
//! The crate is organized into two main APIs:
//!
//! - **Rust API** (`core` module): Pure Rust interfaces for direct integration
//! - **Python API** (PyO3 bindings): NumPy-compatible functions for Python users
//!
//! # Rust Usage
//!
//! ```rust,ignore
//! use tritter_accel::core::{
//!     ternary::{PackedTernary, matmul},
//!     quantization::{quantize_absmean, QuantizeConfig},
//!     training::{GradientCompressor, TrainingConfig},
//!     inference::{InferenceEngine, InferenceConfig},
//! };
//! use candle_core::{Device, Tensor};
//!
//! // Quantize weights to ternary
//! let device = Device::Cpu;
//! let weights = Tensor::randn(0f32, 1f32, (512, 512), &device)?;
//! let result = quantize_absmean(&weights, &QuantizeConfig::default())?;
//!
//! // Create packed representation for efficient matmul
//! let packed = result.to_packed()?;
//!
//! // Run ternary matmul
//! let input = Tensor::randn(0f32, 1f32, (1, 512), &device)?;
//! let output = matmul(&input, &packed, None)?;
//!
//! // Compress gradients for distributed training
//! let compressor = GradientCompressor::new(TrainingConfig::default());
//! let gradients: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
//! let compressed = compressor.compress(&gradients, Some(0.1))?;
//! ```
//!
//! # Python Usage
//!
//! Build with maturin:
//! ```bash
//! cd rust-ai/tritter-accel
//! maturin develop --release
//! ```
//!
//! Then in Python:
//! ```python
//! from tritter_accel import (
//!     pack_ternary_weights,
//!     unpack_ternary_weights,
//!     ternary_matmul,
//!     quantize_weights_absmean,
//!     compress_gradients_vsa,
//! )
//!
//! # Pack weights for efficient storage
//! packed = pack_ternary_weights(ternary_weights, scales)
//!
//! # Efficient matmul with packed weights
//! output = ternary_matmul(input, packed)
//!
//! # Compress gradients for distributed training
//! compressed = compress_gradients_vsa(gradients, compression_ratio=0.1)
//! ```
//!
//! # Features
//!
//! - `cuda`: Enable GPU acceleration via CubeCL (requires CUDA toolkit)
//!
//! # Modules
//!
//! - [`core`]: Pure Rust API for direct integration
//! - [`bitnet`]: Re-exports from `bitnet-quantize`
//! - [`ternary`]: Re-exports from `trit-vsa`
//! - [`vsa`]: Re-exports from `vsa-optim-rs`

#![allow(clippy::type_complexity)]
#![cfg_attr(feature = "python", allow(clippy::useless_conversion))] // PyO3 macro generates false positives

// =============================================================================
// CORE RUST API
// =============================================================================

pub mod core;

// Re-export core types at crate root for convenience
pub use core::{
    inference::{InferenceConfig, InferenceEngine, TernaryLayer},
    quantization::{quantize_absmean, quantize_absmax, QuantizationResult, QuantizeConfig},
    ternary::{matmul as ternary_matmul_rust, PackedTernary, TernaryMatmulConfig},
    training::{GradientCompressor, TrainingConfig},
    vsa::{VsaConfig, VsaOps},
};

// =============================================================================
// RE-EXPORTS FROM SISTER CRATES
// =============================================================================

/// Re-exports from `bitnet-quantize` for direct access.
pub mod bitnet;

/// Re-exports from `trit-vsa` for direct access.
pub mod ternary;

/// Re-exports from `vsa-optim-rs` for direct access.
pub mod vsa;

// =============================================================================
// PYTHON BINDINGS (PyO3) - Only compiled with "python" feature
// =============================================================================

#[cfg(feature = "python")]
mod python_bindings {
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::IntoPyObject;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // Delegate to sister crates
    use bitnet_quantize::{quantize_weights as bitnet_quantize_weights, BitLinear, BitNetConfig};
    use candle_nn::Module;
    use trit_vsa::{PackedTritVec, Trit};
    use vsa_optim_rs::{DeterministicPhaseConfig, DeterministicPhaseTrainer};

    #[cfg(feature = "cuda")]
    mod gpu;

// =============================================================================
// PYTHON WRAPPER TYPES FOR STATEFUL OBJECTS
// =============================================================================

/// Opaque handle for Python to reference a `DeterministicPhaseTrainer`.
///
/// Uses `Arc<Mutex<...>>` for thread-safe sharing between Python calls.
#[pyclass(name = "DeterministicTrainer")]
#[derive(Clone)]
struct PyDeterministicTrainer {
    inner: Arc<Mutex<DeterministicPhaseTrainer>>,
}

/// Opaque handle for Python to reference a `BitLinear` layer.
#[pyclass(name = "BitLinearLayer")]
#[derive(Clone)]
struct PyBitLinearLayer {
    inner: Arc<BitLinear>,
}

// =============================================================================
// DETERMINISTIC PHASE TRAINER BINDINGS
// =============================================================================

/// Create a deterministic phase trainer.
///
/// # Arguments
/// * `param_shapes` - List of (name, shape) tuples for model parameters.
///   Shape can be 1D or 2D (e.g., `[("layer.weight", [768, 768])]`).
/// * `warmup_steps` - Number of warmup steps before prediction begins (default: 10).
/// * `full_steps` - Full gradient steps per cycle after warmup (default: 5).
/// * `predict_steps` - Prediction steps per cycle (default: 20).
/// * `correct_every` - Correction frequency during prediction (default: 5).
///
/// # Returns
/// A `DeterministicTrainer` handle to use with `trainer_step` and `trainer_get_phase`.
///
/// # Example
/// ```python
/// trainer = create_trainer(
///     param_shapes=[("layer.weight", [768, 768]), ("layer.bias", [768])],
///     warmup_steps=10,
///     predict_steps=20,
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (param_shapes, warmup_steps=10, full_steps=5, predict_steps=20, correct_every=5))]
fn create_trainer(
    param_shapes: Vec<(String, Vec<usize>)>,
    warmup_steps: usize,
    full_steps: usize,
    predict_steps: usize,
    correct_every: usize,
) -> PyResult<PyDeterministicTrainer> {
    let config = DeterministicPhaseConfig::default()
        .with_warmup_steps(warmup_steps)
        .with_full_steps(full_steps)
        .with_predict_steps(predict_steps)
        .with_correct_every(correct_every);

    let device = candle_core::Device::Cpu;

    let trainer = DeterministicPhaseTrainer::new(&param_shapes, config, &device)
        .map_err(|e| PyValueError::new_err(format!("Failed to create trainer: {e}")))?;

    Ok(PyDeterministicTrainer {
        inner: Arc::new(Mutex::new(trainer)),
    })
}

/// Process one training step with the deterministic phase trainer.
///
/// # Arguments
/// * `trainer` - The trainer handle from `create_trainer`.
/// * `gradients` - Dictionary mapping parameter names to gradient arrays.
///   Required during WARMUP, FULL, and CORRECT phases. Can be `None` during PREDICT.
/// * `loss` - Loss value for this step.
///
/// # Returns
/// Dictionary with step information:
/// - `phase`: Current phase name ("WARMUP", "FULL", "PREDICT", "CORRECT")
/// - `needs_backward`: Whether backward pass is needed next step
/// - `total_step`: Total steps taken
/// - `speedup`: Effective speedup ratio
/// - `predicted_gradients`: If in PREDICT phase, the predicted gradients (dict of arrays)
///
/// # Example
/// ```python
/// result = trainer_step(trainer, gradients={"layer.weight": grad_array}, loss=0.5)
/// if result["needs_backward"]:
///     # Compute gradients via backpropagation
///     pass
/// else:
///     # Use predicted_gradients from result
///     predicted = result["predicted_gradients"]
/// ```
#[pyfunction]
#[pyo3(signature = (trainer, gradients=None, loss=0.0))]
fn trainer_step<'py>(
    py: Python<'py>,
    trainer: &PyDeterministicTrainer,
    gradients: Option<HashMap<String, PyReadonlyArray2<'py, f32>>>,
    loss: f32,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut inner = trainer
        .inner
        .lock()
        .map_err(|e| PyValueError::new_err(format!("Lock poisoned: {e}")))?;

    // Begin step to get phase info
    let step_info = inner
        .begin_step()
        .map_err(|e| PyValueError::new_err(format!("begin_step failed: {e}")))?;

    let mut result: HashMap<String, Py<PyAny>> = HashMap::new();
    result.insert("phase".to_string(), step_info.phase.to_string().into_pyobject(py).unwrap().into_any().unbind());
    result.insert(
        "needs_backward".to_string(),
        step_info.needs_backward.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
    );
    result.insert(
        "total_step".to_string(),
        step_info.total_step.into_pyobject(py).unwrap().into_any().unbind(),
    );
    result.insert("cycle".to_string(), step_info.cycle.into_pyobject(py).unwrap().into_any().unbind());

    // Handle gradients based on phase
    if step_info.needs_backward {
        // WARMUP, FULL, or CORRECT: record provided gradients
        if let Some(grad_dict) = gradients {
            let device = candle_core::Device::Cpu;
            let mut tensor_grads: HashMap<String, candle_core::Tensor> = HashMap::new();

            for (name, arr) in grad_dict {
                let arr = arr.as_array();
                let shape = arr.shape();
                let data: Vec<f32> = arr.iter().copied().collect();

                let tensor =
                    candle_core::Tensor::from_vec(data, shape.to_vec(), &device).map_err(|e| {
                        PyValueError::new_err(format!("Failed to create tensor for {name}: {e}"))
                    })?;

                tensor_grads.insert(name, tensor);
            }

            inner
                .record_full_gradients(&tensor_grads)
                .map_err(|e| PyValueError::new_err(format!("record_full_gradients failed: {e}")))?;
        }
        result.insert("predicted_gradients".to_string(), py.None());
    } else {
        // PREDICT: get predicted gradients
        let predicted = inner
            .get_predicted_gradients()
            .map_err(|e| PyValueError::new_err(format!("get_predicted_gradients failed: {e}")))?;

        let mut pred_dict: HashMap<String, Py<PyAny>> = HashMap::new();
        for (name, tensor) in predicted {
            let dims = tensor.dims();
            let flat: Vec<f32> = tensor
                .flatten_all()
                .map_err(|e| PyValueError::new_err(format!("flatten failed: {e}")))?
                .to_vec1()
                .map_err(|e| PyValueError::new_err(format!("to_vec1 failed: {e}")))?;

            if dims.len() == 1 {
                let arr = flat.to_pyarray(py);
                pred_dict.insert(name, arr.into_pyobject(py).unwrap().into_any().unbind());
            } else {
                let arr = flat
                    .to_pyarray(py)
                    .reshape(dims.to_vec())
                    .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
                pred_dict.insert(name, arr.into_pyobject(py).unwrap().into_any().unbind());
            }
        }
        result.insert("predicted_gradients".to_string(), pred_dict.into_pyobject(py).unwrap().into_any().unbind());
    }

    // End step
    inner
        .end_step(loss)
        .map_err(|e| PyValueError::new_err(format!("end_step failed: {e}")))?;

    // Add stats
    let stats = inner.get_stats();
    result.insert("speedup".to_string(), stats.speedup.into_pyobject(py).unwrap().into_any().unbind());
    result.insert(
        "mean_prediction_error".to_string(),
        stats.mean_prediction_error.into_pyobject(py).unwrap().into_any().unbind(),
    );

    Ok(result)
}

/// Get the current phase name from the trainer.
///
/// # Arguments
/// * `trainer` - The trainer handle from `create_trainer`.
///
/// # Returns
/// Phase name as string: "WARMUP", "FULL", "PREDICT", or "CORRECT".
#[pyfunction]
fn trainer_get_phase(trainer: &PyDeterministicTrainer) -> PyResult<String> {
    let inner = trainer
        .inner
        .lock()
        .map_err(|e| PyValueError::new_err(format!("Lock poisoned: {e}")))?;

    Ok(inner.current_phase().to_string())
}

/// Get training statistics from the trainer.
///
/// # Arguments
/// * `trainer` - The trainer handle from `create_trainer`.
///
/// # Returns
/// Dictionary with training statistics:
/// - `total_steps`: Total steps taken
/// - `warmup_steps`: Warmup steps taken
/// - `full_steps`: Full gradient steps taken
/// - `predict_steps`: Prediction steps taken
/// - `correct_steps`: Correction steps taken
/// - `cycles`: Training cycles completed
/// - `speedup`: Effective speedup ratio
/// - `mean_prediction_error`: Mean prediction error
/// - `current_loss`: Most recent loss
#[pyfunction]
fn trainer_get_stats(py: Python<'_>, trainer: &PyDeterministicTrainer) -> PyResult<HashMap<String, Py<PyAny>>> {
    let inner = trainer
        .inner
        .lock()
        .map_err(|e| PyValueError::new_err(format!("Lock poisoned: {e}")))?;

    let stats = inner.get_stats();
    let mut result: HashMap<String, Py<PyAny>> = HashMap::new();

    result.insert("total_steps".to_string(), stats.total_steps.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("warmup_steps".to_string(), stats.warmup_steps.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("full_steps".to_string(), stats.full_steps.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("predict_steps".to_string(), stats.predict_steps.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("correct_steps".to_string(), stats.correct_steps.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("cycles".to_string(), stats.cycles.into_pyobject(py).unwrap().into_any().unbind());
    result.insert("speedup".to_string(), stats.speedup.into_pyobject(py).unwrap().into_any().unbind());
    result.insert(
        "mean_prediction_error".to_string(),
        stats.mean_prediction_error.into_pyobject(py).unwrap().into_any().unbind(),
    );
    result.insert("current_loss".to_string(), stats.current_loss.into_pyobject(py).unwrap().into_any().unbind());

    Ok(result)
}

/// Reset the trainer state.
///
/// # Arguments
/// * `trainer` - The trainer handle from `create_trainer`.
#[pyfunction]
fn trainer_reset(trainer: &PyDeterministicTrainer) -> PyResult<()> {
    let mut inner = trainer
        .inner
        .lock()
        .map_err(|e| PyValueError::new_err(format!("Lock poisoned: {e}")))?;

    inner
        .reset()
        .map_err(|e| PyValueError::new_err(format!("reset failed: {e}")))?;

    Ok(())
}

// =============================================================================
// BITLINEAR LAYER BINDINGS
// =============================================================================

/// Create a BitLinear layer from weights.
///
/// BitLinear uses ternary weights {-1, 0, +1} with per-group scales,
/// providing significant compression while maintaining accuracy.
///
/// # Arguments
/// * `weight` - 2D weight array [out_features, in_features].
/// * `bias` - Optional 1D bias array [out_features].
/// * `group_size` - Group size for weight quantization (default: 64).
///
/// # Returns
/// A `BitLinearLayer` handle to use with `bitlinear_forward`.
///
/// # Example
/// ```python
/// layer = create_bitlinear(weight_array, bias=bias_array, group_size=64)
/// output = bitlinear_forward(layer, input_array)
/// print(f"Compression: {bitlinear_compression_ratio(layer):.2f}x")
/// ```
#[pyfunction]
#[pyo3(signature = (weight, bias=None, group_size=64))]
fn create_bitlinear<'py>(
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    group_size: usize,
) -> PyResult<PyBitLinearLayer> {
    let weight_arr = weight.as_array();
    let (out_features, in_features) = (weight_arr.nrows(), weight_arr.ncols());

    let device = candle_core::Device::Cpu;
    let weight_data: Vec<f32> = weight_arr.iter().copied().collect();
    let weight_tensor =
        candle_core::Tensor::from_vec(weight_data, (out_features, in_features), &device)
            .map_err(|e| PyValueError::new_err(format!("Failed to create weight tensor: {e}")))?;

    let bias_tensor = if let Some(b) = bias {
        let bias_arr = b.as_array();
        let bias_data: Vec<f32> = bias_arr.iter().copied().collect();
        Some(
            candle_core::Tensor::from_vec(bias_data, (out_features,), &device)
                .map_err(|e| PyValueError::new_err(format!("Failed to create bias tensor: {e}")))?,
        )
    } else {
        None
    };

    let config = BitNetConfig::default().with_group_size(group_size);

    let layer = BitLinear::from_weight(&weight_tensor, bias_tensor.as_ref(), &config)
        .map_err(|e| PyValueError::new_err(format!("Failed to create BitLinear: {e}")))?;

    Ok(PyBitLinearLayer {
        inner: Arc::new(layer),
    })
}

/// Forward pass through a BitLinear layer.
///
/// # Arguments
/// * `layer` - The BitLinear layer handle from `create_bitlinear`.
/// * `input` - 2D input array [batch_size, in_features] or
///   3D input array [batch_size, seq_len, in_features].
///
/// # Returns
/// Output array with shape [batch_size, out_features] or
/// [batch_size, seq_len, out_features].
#[pyfunction]
fn bitlinear_forward<'py>(
    py: Python<'py>,
    layer: &PyBitLinearLayer,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input_arr = input.as_array();
    let (batch_size, in_features) = (input_arr.nrows(), input_arr.ncols());

    let device = candle_core::Device::Cpu;
    let input_data: Vec<f32> = input_arr.iter().copied().collect();
    let input_tensor =
        candle_core::Tensor::from_vec(input_data, (batch_size, in_features), &device)
            .map_err(|e| PyValueError::new_err(format!("Failed to create input tensor: {e}")))?;

    let output_tensor = layer
        .inner
        .forward(&input_tensor)
        .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {e}")))?;

    let output_dims = output_tensor.dims();
    let output_data: Vec<f32> = output_tensor
        .flatten_all()
        .map_err(|e| PyValueError::new_err(format!("flatten failed: {e}")))?
        .to_vec1()
        .map_err(|e| PyValueError::new_err(format!("to_vec1 failed: {e}")))?;

    Ok(output_data
        .to_pyarray(py)
        .reshape([output_dims[0], output_dims[1]])
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?
        .to_owned())
}

/// Get the compression ratio of a BitLinear layer.
///
/// # Arguments
/// * `layer` - The BitLinear layer handle from `create_bitlinear`.
///
/// # Returns
/// Compression ratio (e.g., 8.0 means 8x compression vs float32).
#[pyfunction]
fn bitlinear_compression_ratio(layer: &PyBitLinearLayer) -> f32 {
    layer.inner.compression_ratio()
}

/// Get the sparsity of a BitLinear layer.
///
/// # Arguments
/// * `layer` - The BitLinear layer handle from `create_bitlinear`.
///
/// # Returns
/// Sparsity ratio (fraction of weights that are zero).
#[pyfunction]
fn bitlinear_sparsity(layer: &PyBitLinearLayer) -> f32 {
    layer.inner.sparsity()
}

/// Get the input features dimension of a BitLinear layer.
///
/// # Arguments
/// * `layer` - The BitLinear layer handle from `create_bitlinear`.
///
/// # Returns
/// Number of input features.
#[pyfunction]
fn bitlinear_in_features(layer: &PyBitLinearLayer) -> usize {
    layer.inner.in_features()
}

/// Get the output features dimension of a BitLinear layer.
///
/// # Arguments
/// * `layer` - The BitLinear layer handle from `create_bitlinear`.
///
/// # Returns
/// Number of output features.
#[pyfunction]
fn bitlinear_out_features(layer: &PyBitLinearLayer) -> usize {
    layer.inner.out_features()
}

// =============================================================================
// ORIGINAL TERNARY BINDINGS
// =============================================================================

/// Pack ternary weights into efficient 2-bit representation.
///
/// # Arguments
/// * `weights` - 2D array of ternary values {-1, 0, +1}
/// * `scales` - Per-row scale factors
///
/// # Returns
/// Tuple of (packed_bytes, scales) for storage/transmission.
///
/// Note: Internally uses trit-vsa's bitsliced storage, but returns
/// a compatible 2-bit packed format for interoperability.
#[pyfunction]
fn pack_ternary_weights<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
    scales: PyReadonlyArray1<'py, f32>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<f32>>)> {
    let weights = weights.as_array();
    let scales = scales.as_array();

    let (rows, cols) = (weights.nrows(), weights.ncols());

    // Convert to trit-vsa PackedTritVec per row
    let mut packed_vecs: Vec<PackedTritVec> = Vec::with_capacity(rows);

    for row in weights.rows() {
        let mut packed = PackedTritVec::new(cols);
        for (col_idx, &val) in row.iter().enumerate() {
            let trit = match val as i8 {
                v if v > 0 => Trit::P,  // +1
                v if v < 0 => Trit::N,  // -1
                _ => Trit::Z,           // 0
            };
            packed.set(col_idx, trit);
        }
        packed_vecs.push(packed);
    }

    // Convert bitsliced representation to 2-bit packed format for Python compatibility
    let packed_size = cols.div_ceil(4);
    let mut packed = vec![0u8; rows * packed_size];

    for (row_idx, pvec) in packed_vecs.iter().enumerate() {
        for col_idx in 0..cols {
            let trit = pvec.get(col_idx);
            let trit_bits = match trit {
                Trit::P => 0b01, // +1
                Trit::N => 0b10, // -1
                Trit::Z => 0b00, // 0
            };

            let byte_idx = row_idx * packed_size + col_idx / 4;
            let bit_offset = (col_idx % 4) * 2;
            packed[byte_idx] |= trit_bits << bit_offset;
        }
    }

    let packed_array = packed.to_pyarray(py);
    let scales_array = scales.to_vec().to_pyarray(py);

    Ok((packed_array, scales_array))
}

/// Unpack ternary weights from 2-bit representation.
///
/// # Arguments
/// * `packed` - Packed byte array
/// * `scales` - Per-row scale factors
/// * `shape` - Original (rows, cols) shape
///
/// # Returns
/// 2D array of dequantized weights.
///
/// Note: Uses trit-vsa for intermediate storage.
#[pyfunction]
fn unpack_ternary_weights<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u8>,
    scales: PyReadonlyArray1<'py, f32>,
    shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed = packed.as_array();
    let scales = scales.as_array();
    let (rows, cols) = shape;

    let packed_size = cols.div_ceil(4);

    // Convert 2-bit packed format to trit-vsa PackedTritVec
    let mut packed_vecs: Vec<PackedTritVec> = Vec::with_capacity(rows);

    for row_idx in 0..rows {
        let mut pvec = PackedTritVec::new(cols);
        for col_idx in 0..cols {
            let byte_idx = row_idx * packed_size + col_idx / 4;
            let bit_offset = (col_idx % 4) * 2;
            let trit_bits = (packed[byte_idx] >> bit_offset) & 0b11;

            let trit = match trit_bits {
                0b01 => Trit::P,  // +1
                0b10 => Trit::N,  // -1
                _ => Trit::Z,     // 0
            };
            pvec.set(col_idx, trit);
        }
        packed_vecs.push(pvec);
    }

    // Dequantize using scales
    let mut weights = vec![0.0f32; rows * cols];

    for (row_idx, pvec) in packed_vecs.iter().enumerate() {
        let scale = scales[row_idx];
        for col_idx in 0..cols {
            let value = f32::from(pvec.get(col_idx).value()) * scale;
            weights[row_idx * cols + col_idx] = value;
        }
    }

    Ok(weights
        .to_pyarray(py)
        .reshape([rows, cols])
        .expect("reshape failed")
        .to_owned())
}

/// Efficient matrix multiplication with packed ternary weights.
///
/// Computes: output = input @ weights.T
///
/// # Arguments
/// * `input` - 2D input tensor (batch, in_features)
/// * `packed_weights` - Packed ternary weights
/// * `scales` - Per-output-channel scale factors
/// * `weight_shape` - (out_features, in_features)
///
/// # Returns
/// Output tensor (batch, out_features)
///
/// Note: Uses trit-vsa PackedTritVec for efficient dot products.
#[pyfunction]
fn ternary_matmul<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    packed_weights: PyReadonlyArray1<'py, u8>,
    scales: PyReadonlyArray1<'py, f32>,
    weight_shape: (usize, usize),
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input = input.as_array();
    let packed = packed_weights.as_array();
    let scales = scales.as_array();

    let (batch_size, in_features) = (input.nrows(), input.ncols());
    let (out_features, _) = weight_shape;

    if in_features != weight_shape.1 {
        return Err(PyValueError::new_err(format!(
            "Input features {} doesn't match weight features {}",
            in_features, weight_shape.1
        )));
    }

    let packed_cols = in_features.div_ceil(4);

    // Convert packed weights to trit-vsa PackedTritVec for each output row
    let mut weight_vecs: Vec<PackedTritVec> = Vec::with_capacity(out_features);

    for o in 0..out_features {
        let mut pvec = PackedTritVec::new(in_features);
        for i in 0..in_features {
            let byte_idx = o * packed_cols + i / 4;
            let bit_offset = (i % 4) * 2;
            let trit_bits = (packed[byte_idx] >> bit_offset) & 0b11;

            let trit = match trit_bits {
                0b01 => Trit::P,
                0b10 => Trit::N,
                _ => Trit::Z,
            };
            pvec.set(i, trit);
        }
        weight_vecs.push(pvec);
    }

    let mut output = vec![0.0f32; batch_size * out_features];

    // Compute matmul: for each batch, compute dot product with each weight row
    for b in 0..batch_size {
        for (o, weight_vec) in weight_vecs.iter().enumerate() {
            let scale = scales[o];
            let mut sum = 0.0f32;

            // Use the trit values to select add/subtract/skip
            for i in 0..in_features {
                let trit = weight_vec.get(i);
                let x = input[[b, i]];
                sum += match trit {
                    Trit::P => x,   // +1: add
                    Trit::N => -x,  // -1: subtract
                    Trit::Z => 0.0, // 0: skip
                };
            }

            output[b * out_features + o] = sum * scale;
        }
    }

    Ok(output
        .to_pyarray(py)
        .reshape([batch_size, out_features])
        .expect("reshape failed")
        .to_owned())
}

/// Quantize weights to ternary using AbsMean scaling.
///
/// # Arguments
/// * `weights` - 2D weight tensor
///
/// # Returns
/// Tuple of (ternary_weights, scales)
///
/// Note: Delegates to bitnet-quantize for the core algorithm.
#[pyfunction]
fn quantize_weights_absmean<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
    let weights_arr = weights.as_array();
    let (rows, cols) = (weights_arr.nrows(), weights_arr.ncols());

    // Use bitnet-quantize with group_size = cols (per-row quantization)
    let config = BitNetConfig::default().with_group_size(cols);

    // Convert numpy array to candle tensor
    let device = candle_core::Device::Cpu;
    let weight_data: Vec<f32> = weights_arr.iter().copied().collect();
    let weight_tensor = candle_core::Tensor::from_vec(weight_data, (rows, cols), &device)
        .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {e}")))?;

    // Quantize using bitnet-quantize
    let ternary = bitnet_quantize_weights(&weight_tensor, &config)
        .map_err(|e| PyValueError::new_err(format!("Quantization failed: {e}")))?;

    // Extract ternary values and scales
    let mut ternary_output = vec![0.0f32; rows * cols];
    let scales: Vec<f32> = ternary.scales.clone();

    for (row_idx, packed) in ternary.data.iter().enumerate() {
        for col_idx in 0..cols {
            let trit = packed.get(col_idx);
            ternary_output[row_idx * cols + col_idx] = f32::from(trit.value());
        }
    }

    Ok((
        ternary_output
            .to_pyarray(py)
            .reshape([rows, cols])
            .expect("reshape failed")
            .to_owned(),
        scales.to_pyarray(py),
    ))
}

/// Compress gradients using VSA random projection.
///
/// # Arguments
/// * `gradients` - Flattened gradient tensor
/// * `compression_ratio` - Target compression (0.0 to 1.0)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Tuple of (compressed_gradients, projection_seed)
///
/// Note: Uses a simplified random projection for Python compatibility.
/// For full VSA with bind/bundle/unbind, use vsa-optim-rs directly.
#[pyfunction]
#[allow(clippy::cast_precision_loss)]
fn compress_gradients_vsa<'py>(
    py: Python<'py>,
    gradients: PyReadonlyArray1<'py, f32>,
    compression_ratio: f32,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f32>>, u64)> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let gradients = gradients.as_array();
    let original_dim = gradients.len();
    let compressed_dim = ((original_dim as f32 * compression_ratio).ceil() as usize).max(256);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut compressed = vec![0.0f32; compressed_dim];

    // Random projection (Johnson-Lindenstrauss style)
    let scale = 1.0 / (original_dim as f32).sqrt();
    for &g in gradients.iter() {
        for c in compressed.iter_mut() {
            // Sparse random projection: ~68% zeros, 16% +1, 16% -1
            let r: f32 = rng.gen();
            if r < 0.16 {
                *c += g * scale;
            } else if r < 0.32 {
                *c -= g * scale;
            }
        }
    }

    Ok((compressed.to_pyarray(py), seed))
}

/// Decompress gradients from VSA projection.
///
/// # Arguments
/// * `compressed` - Compressed gradient tensor
/// * `original_dim` - Original gradient dimension
/// * `seed` - Random seed (must match compression)
///
/// # Returns
/// Reconstructed gradient tensor (approximate)
///
/// Note: Uses simplified inverse projection for Python compatibility.
#[pyfunction]
#[allow(clippy::cast_precision_loss)]
fn decompress_gradients_vsa<'py>(
    py: Python<'py>,
    compressed: PyReadonlyArray1<'py, f32>,
    original_dim: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let compressed = compressed.as_array();

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut gradients = vec![0.0f32; original_dim];

    // Inverse projection (transpose of forward projection)
    let scale = 1.0 / (original_dim as f32).sqrt();
    for g in gradients.iter_mut() {
        for &c in compressed.iter() {
            let r: f32 = rng.gen();
            if r < 0.16 {
                *g += c * scale;
            } else if r < 0.32 {
                *g -= c * scale;
            }
        }
    }

    Ok(gradients.to_pyarray(py))
}

/// Get version information.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Check if CUDA is available (for Python).
#[pyfunction]
fn cuda_available_py() -> bool {
    #[cfg(feature = "cuda")]
    {
        candle_core::Device::cuda_if_available(0)
            .map(|d| matches!(d, candle_core::Device::Cuda(_)))
            .unwrap_or(false)
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Python module definition.
#[pymodule]
fn tritter_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register class types
    m.add_class::<PyDeterministicTrainer>()?;
    m.add_class::<PyBitLinearLayer>()?;

    // Deterministic phase trainer functions
    m.add_function(wrap_pyfunction!(create_trainer, m)?)?;
    m.add_function(wrap_pyfunction!(trainer_step, m)?)?;
    m.add_function(wrap_pyfunction!(trainer_get_phase, m)?)?;
    m.add_function(wrap_pyfunction!(trainer_get_stats, m)?)?;
    m.add_function(wrap_pyfunction!(trainer_reset, m)?)?;

    // BitLinear layer functions
    m.add_function(wrap_pyfunction!(create_bitlinear, m)?)?;
    m.add_function(wrap_pyfunction!(bitlinear_forward, m)?)?;
    m.add_function(wrap_pyfunction!(bitlinear_compression_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(bitlinear_sparsity, m)?)?;
    m.add_function(wrap_pyfunction!(bitlinear_in_features, m)?)?;
    m.add_function(wrap_pyfunction!(bitlinear_out_features, m)?)?;

    // Core ternary functions
    m.add_function(wrap_pyfunction!(pack_ternary_weights, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_ternary_weights, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_weights_absmean, m)?)?;

    // Gradient compression
    m.add_function(wrap_pyfunction!(compress_gradients_vsa, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_gradients_vsa, m)?)?;

    // Utilities
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available_py, m)?)?;

    // GPU-accelerated VSA operations (when cuda feature enabled)
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(gpu::cuda_available, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::get_device, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::set_device, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::vsa_bind, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::vsa_unbind, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::vsa_bundle, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::vsa_similarity, m)?)?;
        m.add_function(wrap_pyfunction!(gpu::vsa_random, m)?)?;
    }

    Ok(())
}
} // End of python_bindings module

// Re-export the Python module entry point when the python feature is enabled
#[cfg(feature = "python")]
pub use python_bindings::*;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use trit_vsa::{PackedTritVec, Trit};
    use bitnet_quantize::{quantize_weights as bitnet_quantize_weights, BitLinear, BitNetConfig};
    use candle_nn::Module;

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Test that pack/unpack preserves ternary values using trit-vsa
        let weights = vec![1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.0, -1.0];
        let scales = vec![1.0, 1.0];

        // Create PackedTritVec for each row
        let mut packed_vecs = Vec::new();
        for row in 0..2 {
            let mut pvec = PackedTritVec::new(4);
            for col in 0..4 {
                let val = weights[row * 4 + col];
                let trit = match val as i8 {
                    v if v > 0 => Trit::P,
                    v if v < 0 => Trit::N,
                    _ => Trit::Z,
                };
                pvec.set(col, trit);
            }
            packed_vecs.push(pvec);
        }

        // Unpack and dequantize
        let mut unpacked = vec![0.0f32; 8];
        for (row, pvec) in packed_vecs.iter().enumerate() {
            let scale = scales[row];
            for col in 0..4 {
                unpacked[row * 4 + col] = f32::from(pvec.get(col).value()) * scale;
            }
        }

        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_quantize_absmean_with_bitnet() {
        // Test that quantization works with bitnet-quantize
        let weights = vec![0.5, -0.3, 0.1, 0.8, -0.2, 0.6, -0.7, 0.4];

        // Use bitnet-quantize directly
        let device = candle_core::Device::Cpu;
        let config = BitNetConfig::default().with_group_size(4);
        let tensor = candle_core::Tensor::from_vec(weights.clone(), (2, 4), &device).unwrap();

        let ternary = bitnet_quantize_weights(&tensor, &config).unwrap();

        // Check structure
        assert_eq!(ternary.shape, (2, 4));
        assert_eq!(ternary.data.len(), 2);
    }

    #[test]
    fn test_trit_vsa_dot_product() {
        // Test trit-vsa dot product functionality
        let mut a = PackedTritVec::new(4);
        let mut b = PackedTritVec::new(4);

        // a = [+1, -1, 0, +1]
        a.set(0, Trit::P);
        a.set(1, Trit::N);
        a.set(2, Trit::Z);
        a.set(3, Trit::P);

        // b = [+1, +1, -1, 0]
        b.set(0, Trit::P);
        b.set(1, Trit::P);
        b.set(2, Trit::N);
        b.set(3, Trit::Z);

        // dot = 1*1 + (-1)*1 + 0*(-1) + 1*0 = 1 - 1 + 0 + 0 = 0
        assert_eq!(a.dot(&b), 0);
    }

    #[test]
    fn test_rust_api_quantization() {
        // Test the Rust API directly
        use crate::core::quantization::{quantize_absmean, QuantizeConfig};

        let device = candle_core::Device::Cpu;
        let weights = candle_core::Tensor::from_vec(
            vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.6, -0.7, 0.4],
            (2, 4),
            &device,
        )
        .unwrap();

        let config = QuantizeConfig::default();
        let result = quantize_absmean(&weights, &config).unwrap();

        assert_eq!(result.shape, (2, 4));
        assert_eq!(result.values.len(), 8);

        // All values should be ternary
        for v in &result.values {
            assert!([-1, 0, 1].contains(v));
        }
    }

    #[test]
    fn test_rust_api_gradient_compression() {
        // Test the Rust API for gradient compression
        use crate::core::training::{GradientCompressor, TrainingConfig};

        let config = TrainingConfig::default();
        let compressor = GradientCompressor::new(config);

        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 500.0).collect();
        let compressed = compressor.compress(&gradients, Some(0.1)).unwrap();
        let recovered = compressor.decompress(&compressed).unwrap();

        assert_eq!(recovered.len(), gradients.len());
    }

    #[test]
    fn test_deterministic_phase_trainer_creation() {
        // Test that DeterministicPhaseTrainer can be created from vsa-optim-rs
        use vsa_optim_rs::{DeterministicPhaseConfig, DeterministicPhaseTrainer};

        let shapes = vec![
            ("layer.weight".to_string(), vec![16, 32]),
            ("layer.bias".to_string(), vec![16]),
        ];

        let config = DeterministicPhaseConfig::default()
            .with_warmup_steps(5)
            .with_full_steps(3)
            .with_predict_steps(10);

        let trainer =
            DeterministicPhaseTrainer::new(&shapes, config, &candle_core::Device::Cpu).unwrap();

        // Initial phase should be WARMUP
        assert_eq!(
            trainer.current_phase(),
            vsa_optim_rs::DeterministicPhase::Warmup
        );
    }

    #[test]
    fn test_deterministic_trainer_step_cycle() {
        use std::collections::HashMap;
        use vsa_optim_rs::{DeterministicPhaseConfig, DeterministicPhaseTrainer};

        let shapes = vec![
            ("layer.weight".to_string(), vec![8, 16]),
            ("layer.bias".to_string(), vec![8]),
        ];

        let config = DeterministicPhaseConfig::default()
            .with_warmup_steps(3)
            .with_full_steps(2)
            .with_predict_steps(5);

        let mut trainer =
            DeterministicPhaseTrainer::new(&shapes, config, &candle_core::Device::Cpu).unwrap();

        // Run through several steps
        for i in 0..10 {
            let info = trainer.begin_step().unwrap();

            if info.needs_backward {
                // Create mock gradients
                let mut grads = HashMap::new();
                grads.insert(
                    "layer.weight".to_string(),
                    candle_core::Tensor::ones((8, 16), candle_core::DType::F32, &candle_core::Device::Cpu)
                        .unwrap()
                        .affine((i as f64 + 1.0) * 0.1, 0.0)
                        .unwrap(),
                );
                grads.insert(
                    "layer.bias".to_string(),
                    candle_core::Tensor::ones(8, candle_core::DType::F32, &candle_core::Device::Cpu)
                        .unwrap()
                        .affine((i as f64 + 1.0) * 0.1, 0.0)
                        .unwrap(),
                );
                trainer.record_full_gradients(&grads).unwrap();
            } else {
                let _predicted = trainer.get_predicted_gradients();
            }

            trainer.end_step(1.0 / (i + 1) as f32).unwrap();
        }

        let stats = trainer.get_stats();
        assert_eq!(stats.total_steps, 10);
    }

    #[test]
    fn test_bitlinear_layer_creation() {
        let device = candle_core::Device::Cpu;
        let config = BitNetConfig::default().with_group_size(16);

        let weight =
            candle_core::Tensor::randn(0.0f32, 1.0, (32, 64), &device).unwrap();

        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        assert_eq!(layer.in_features(), 64);
        assert_eq!(layer.out_features(), 32);
        assert!(layer.compression_ratio() > 1.0);
    }

    #[test]
    fn test_bitlinear_forward_pass() {
        let device = candle_core::Device::Cpu;
        let config = BitNetConfig::default().with_group_size(16);

        let weight =
            candle_core::Tensor::randn(0.0f32, 1.0, (32, 64), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        let input = candle_core::Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 32]);
    }

    #[test]
    fn test_bitlinear_with_bias() {
        let device = candle_core::Device::Cpu;
        let config = BitNetConfig::default().with_group_size(16);

        let weight =
            candle_core::Tensor::randn(0.0f32, 1.0, (32, 64), &device).unwrap();
        let bias = candle_core::Tensor::randn(0.0f32, 1.0, (32,), &device).unwrap();

        let layer = BitLinear::from_weight(&weight, Some(&bias), &config).unwrap();

        assert!(layer.bias().is_some());

        let input = candle_core::Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 32]);
    }
}
