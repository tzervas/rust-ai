//! GPU-accelerated VSA operations for Python.
//!
//! This module provides Python bindings for GPU-accelerated ternary VSA operations.
//! It wraps the GPU operations from `trit-vsa` using the `GpuDispatchable` trait.
//!
//! # Device Selection
//!
//! Operations support automatic device selection:
//! - `"cuda"` or `"gpu"`: Force GPU execution (fails if unavailable)
//! - `"cpu"`: Force CPU execution
//! - `"auto"`: Use GPU if available, fall back to CPU
//!
//! # Example
//!
//! ```python
//! from tritter_accel import cuda_available, vsa_bind, vsa_similarity
//!
//! if cuda_available():
//!     a = np.random.randint(-1, 2, size=10000, dtype=np.int8)
//!     b = np.random.randint(-1, 2, size=10000, dtype=np.int8)
//!     bound = vsa_bind(a, b, device="cuda")
//!     similarity = vsa_similarity(a, bound, metric="cosine")
//! ```

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use candle_core::Device;
use trit_vsa::gpu::{
    GpuBind, GpuBundle, GpuCosineSimilarity, GpuDispatchable, GpuDotSimilarity, GpuHammingDistance,
    GpuRandom, GpuUnbind, RandomInput,
};
use trit_vsa::{PackedTritVec, Trit};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse device string and return appropriate candle Device.
fn parse_device(device: &str) -> PyResult<Device> {
    match device.to_lowercase().as_str() {
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda_if_available(0)
                    .map_err(|e| PyValueError::new_err(format!("CUDA not available: {e}")))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(PyValueError::new_err(
                    "CUDA support not compiled. Rebuild with --features cuda",
                ))
            }
        }
        "cpu" => Ok(Device::Cpu),
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::cuda_if_available(0).unwrap_or(Device::Cpu))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Device::Cpu)
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown device '{device}'. Use 'cuda', 'cpu', or 'auto'"
        ))),
    }
}

/// Convert i8 numpy array to PackedTritVec.
fn array_to_packed(arr: &[i8]) -> PyResult<PackedTritVec> {
    let mut packed = PackedTritVec::new(arr.len());
    for (i, &val) in arr.iter().enumerate() {
        let trit = match val {
            1 => Trit::P,
            0 => Trit::Z,
            -1 => Trit::N,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid trit value {} at index {}. Must be -1, 0, or 1",
                    val, i
                )));
            }
        };
        packed.set(i, trit);
    }
    Ok(packed)
}

/// Convert PackedTritVec to i8 Vec.
fn packed_to_array(packed: &PackedTritVec) -> Vec<i8> {
    let mut result = Vec::with_capacity(packed.len());
    for i in 0..packed.len() {
        result.push(packed.get(i).value());
    }
    result
}

// =============================================================================
// DEVICE MANAGEMENT
// =============================================================================

/// Check if CUDA is available.
///
/// Returns True if the library was compiled with CUDA support and a CUDA
/// device is available on the system.
///
/// # Example
///
/// ```python
/// from tritter_accel import cuda_available
///
/// if cuda_available():
///     print("GPU acceleration available!")
/// else:
///     print("Running on CPU")
/// ```
#[pyfunction]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::cuda_if_available(0)
            .map(|d| matches!(d, Device::Cuda(_)))
            .unwrap_or(false)
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the current default device.
///
/// Returns:
///     str: "cuda" if CUDA is available and compiled, "cpu" otherwise.
///
/// # Example
///
/// ```python
/// from tritter_accel import get_device
///
/// device = get_device()
/// print(f"Default device: {device}")
/// ```
#[pyfunction]
pub fn get_device() -> &'static str {
    if cuda_available() {
        "cuda"
    } else {
        "cpu"
    }
}

/// Set the default CUDA device ordinal (placeholder for future multi-GPU support).
///
/// Currently this is a no-op but exists for API compatibility.
///
/// # Arguments
///
/// * `device` - Device string ("cuda", "cpu", or "cuda:N" for specific GPU)
///
/// # Example
///
/// ```python
/// from tritter_accel import set_device
///
/// set_device("cuda:0")  # Select first GPU
/// ```
#[pyfunction]
pub fn set_device(_device: &str) -> PyResult<()> {
    // Currently a no-op - multi-GPU support planned for future
    Ok(())
}

// =============================================================================
// VSA OPERATIONS
// =============================================================================

/// Bind two ternary vectors (association operation).
///
/// Bind is the composition operation in VSA, analogous to XOR in binary systems.
/// It creates associations between vectors while preserving dissimilarity.
///
/// # Properties
///
/// - Commutative: bind(a, b) == bind(b, a)
/// - Associative: bind(bind(a, b), c) == bind(a, bind(b, c))
/// - Can be undone with unbind: unbind(bind(a, b), b) ~= a
///
/// # Arguments
///
/// * `a` - First ternary vector (values in {-1, 0, +1})
/// * `b` - Second ternary vector (same length as a)
/// * `device` - Device to use: "cuda", "cpu", or "auto" (default: "auto")
///
/// # Returns
///
/// Bound ternary vector as numpy array of int8.
///
/// # Example
///
/// ```python
/// import numpy as np
/// from tritter_accel import vsa_bind
///
/// a = np.array([1, 0, -1, 1, -1], dtype=np.int8)
/// b = np.array([1, -1, 0, -1, 1], dtype=np.int8)
/// bound = vsa_bind(a, b, device="auto")
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, device = "auto"))]
pub fn vsa_bind<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, i8>,
    b: PyReadonlyArray1<'py, i8>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    if a_arr.len() != b_arr.len() {
        return Err(PyValueError::new_err(format!(
            "Vector dimensions must match: {} vs {}",
            a_arr.len(),
            b_arr.len()
        )));
    }

    let a_packed = array_to_packed(a_arr.as_slice().unwrap())?;
    let b_packed = array_to_packed(b_arr.as_slice().unwrap())?;

    let dev = parse_device(device)?;
    let result = GpuBind
        .dispatch(&(a_packed, b_packed), &dev)
        .map_err(|e| PyValueError::new_err(format!("Bind operation failed: {e}")))?;

    let result_arr = packed_to_array(&result);
    Ok(result_arr.to_pyarray_bound(py))
}

/// Unbind two ternary vectors (inverse association).
///
/// Unbind is the inverse of bind, used to recover associated vectors.
/// If bound = bind(a, b), then unbind(bound, b) recovers a.
///
/// # Arguments
///
/// * `bound` - Bound vector to unbind
/// * `key` - Key vector used in original bind
/// * `device` - Device to use: "cuda", "cpu", or "auto" (default: "auto")
///
/// # Returns
///
/// Unbound ternary vector as numpy array of int8.
///
/// # Example
///
/// ```python
/// import numpy as np
/// from tritter_accel import vsa_bind, vsa_unbind
///
/// a = np.array([1, 0, -1, 1, -1], dtype=np.int8)
/// b = np.array([1, -1, 0, -1, 1], dtype=np.int8)
/// bound = vsa_bind(a, b)
/// recovered = vsa_unbind(bound, b)  # recovers a
/// ```
#[pyfunction]
#[pyo3(signature = (bound, key, device = "auto"))]
pub fn vsa_unbind<'py>(
    py: Python<'py>,
    bound: PyReadonlyArray1<'py, i8>,
    key: PyReadonlyArray1<'py, i8>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let bound_arr = bound.as_array();
    let key_arr = key.as_array();

    if bound_arr.len() != key_arr.len() {
        return Err(PyValueError::new_err(format!(
            "Vector dimensions must match: {} vs {}",
            bound_arr.len(),
            key_arr.len()
        )));
    }

    let bound_packed = array_to_packed(bound_arr.as_slice().unwrap())?;
    let key_packed = array_to_packed(key_arr.as_slice().unwrap())?;

    let dev = parse_device(device)?;
    let result = GpuUnbind
        .dispatch(&(bound_packed, key_packed), &dev)
        .map_err(|e| PyValueError::new_err(format!("Unbind operation failed: {e}")))?;

    let result_arr = packed_to_array(&result);
    Ok(result_arr.to_pyarray_bound(py))
}

/// Bundle multiple ternary vectors (superposition via majority voting).
///
/// Bundle combines multiple vectors into one that is similar to all inputs.
/// This is the "addition" operation in hyperdimensional computing.
///
/// For each dimension, counts votes for each trit value and selects majority.
/// Ties are resolved to zero.
///
/// # Arguments
///
/// * `vectors` - List of ternary vectors (all same length)
/// * `device` - Device to use: "cuda", "cpu", or "auto" (default: "auto")
///
/// # Returns
///
/// Bundled ternary vector as numpy array of int8.
///
/// # Example
///
/// ```python
/// import numpy as np
/// from tritter_accel import vsa_bundle
///
/// vectors = [
///     np.array([1, 1, -1, 0], dtype=np.int8),
///     np.array([1, -1, -1, 1], dtype=np.int8),
///     np.array([1, 0, 1, -1], dtype=np.int8),
/// ]
/// bundled = vsa_bundle(vectors, device="cuda")
/// # bundled[0] = 1 (majority)
/// # bundled[2] = -1 (majority)
/// ```
#[pyfunction]
#[pyo3(signature = (vectors, device = "auto"))]
pub fn vsa_bundle<'py>(
    py: Python<'py>,
    vectors: Vec<PyReadonlyArray1<'py, i8>>,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    if vectors.is_empty() {
        return Err(PyValueError::new_err("Cannot bundle empty vector list"));
    }

    let first_len = vectors[0].as_array().len();
    let mut packed_vecs = Vec::with_capacity(vectors.len());

    for (i, v) in vectors.iter().enumerate() {
        let arr = v.as_array();
        if arr.len() != first_len {
            return Err(PyValueError::new_err(format!(
                "All vectors must have same length. Vector 0 has {}, vector {} has {}",
                first_len,
                i,
                arr.len()
            )));
        }
        packed_vecs.push(array_to_packed(arr.as_slice().unwrap())?);
    }

    let dev = parse_device(device)?;
    let result = GpuBundle
        .dispatch(&packed_vecs, &dev)
        .map_err(|e| PyValueError::new_err(format!("Bundle operation failed: {e}")))?;

    let result_arr = packed_to_array(&result);
    Ok(result_arr.to_pyarray_bound(py))
}

/// Compute similarity between two ternary vectors.
///
/// Supports multiple similarity metrics:
/// - "cosine": Cosine similarity in [-1, 1]
/// - "dot": Dot product in [-n, n]
/// - "hamming": Hamming distance in [0, n]
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `metric` - Similarity metric: "cosine", "dot", or "hamming" (default: "cosine")
/// * `device` - Device to use: "cuda", "cpu", or "auto" (default: "auto")
///
/// # Returns
///
/// Similarity value as float64.
///
/// # Example
///
/// ```python
/// import numpy as np
/// from tritter_accel import vsa_similarity
///
/// a = np.array([1, 0, -1, 1, -1], dtype=np.int8)
/// b = np.array([1, -1, 0, -1, 1], dtype=np.int8)
///
/// cos_sim = vsa_similarity(a, b, metric="cosine")
/// dot_prod = vsa_similarity(a, b, metric="dot")
/// hamming = vsa_similarity(a, b, metric="hamming")
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, metric = "cosine", device = "auto"))]
pub fn vsa_similarity<'py>(
    _py: Python<'py>,
    a: PyReadonlyArray1<'py, i8>,
    b: PyReadonlyArray1<'py, i8>,
    metric: &str,
    device: &str,
) -> PyResult<f64> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    if a_arr.len() != b_arr.len() {
        return Err(PyValueError::new_err(format!(
            "Vector dimensions must match: {} vs {}",
            a_arr.len(),
            b_arr.len()
        )));
    }

    let a_packed = array_to_packed(a_arr.as_slice().unwrap())?;
    let b_packed = array_to_packed(b_arr.as_slice().unwrap())?;
    let dev = parse_device(device)?;

    match metric.to_lowercase().as_str() {
        "cosine" => {
            let result = GpuCosineSimilarity
                .dispatch(&(a_packed, b_packed), &dev)
                .map_err(|e| PyValueError::new_err(format!("Cosine similarity failed: {e}")))?;
            Ok(f64::from(result))
        }
        "dot" => {
            let result = GpuDotSimilarity
                .dispatch(&(a_packed, b_packed), &dev)
                .map_err(|e| PyValueError::new_err(format!("Dot similarity failed: {e}")))?;
            Ok(f64::from(result))
        }
        "hamming" => {
            let result = GpuHammingDistance
                .dispatch(&(a_packed, b_packed), &dev)
                .map_err(|e| PyValueError::new_err(format!("Hamming distance failed: {e}")))?;
            Ok(result as f64)
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown metric '{metric}'. Use 'cosine', 'dot', or 'hamming'"
        ))),
    }
}

/// Generate a random ternary vector.
///
/// Creates a vector with uniformly distributed values in {-1, 0, +1}.
/// The generation is deterministic given the same seed.
///
/// # Arguments
///
/// * `dim` - Vector dimension
/// * `seed` - Random seed for reproducibility
/// * `device` - Device to use: "cuda", "cpu", or "auto" (default: "auto")
///
/// # Returns
///
/// Random ternary vector as numpy array of int8.
///
/// # Example
///
/// ```python
/// from tritter_accel import vsa_random
///
/// # Create a random 10000-dimensional vector
/// vec = vsa_random(10000, seed=42, device="cuda")
/// ```
#[pyfunction]
#[pyo3(signature = (dim, seed, device = "auto"))]
pub fn vsa_random<'py>(
    py: Python<'py>,
    dim: usize,
    seed: u64,
    device: &str,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    if dim == 0 {
        return Err(PyValueError::new_err("Dimension must be > 0"));
    }

    // GpuRandom uses u32 seed
    #[allow(clippy::cast_possible_truncation)]
    let seed_u32 = seed as u32;

    let dev = parse_device(device)?;
    let input = RandomInput::new(dim, seed_u32);

    let result = GpuRandom
        .dispatch(&input, &dev)
        .map_err(|e| PyValueError::new_err(format!("Random generation failed: {e}")))?;

    let result_arr = packed_to_array(&result);
    Ok(result_arr.to_pyarray_bound(py))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_to_packed_roundtrip() {
        let original: Vec<i8> = vec![1, 0, -1, 1, -1, 0, 0, 1];
        let packed = array_to_packed(&original).unwrap();
        let recovered = packed_to_array(&packed);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_array_to_packed_invalid_value() {
        let invalid: Vec<i8> = vec![1, 0, 2, -1]; // 2 is invalid
        let result = array_to_packed(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        // auto should always succeed
        let device = parse_device("auto").unwrap();
        // Result depends on whether CUDA is available
        assert!(matches!(device, Device::Cpu | Device::Cuda(_)));
    }

    #[test]
    fn test_parse_device_invalid() {
        let result = parse_device("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_available_returns_bool() {
        // Just verify it returns without panicking
        let _ = cuda_available();
    }

    #[test]
    fn test_get_device_returns_string() {
        let device = get_device();
        assert!(device == "cpu" || device == "cuda");
    }
}
