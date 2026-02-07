//! Feed-forward network with Squared ReLU and BitNet quantization.
//!
//! Implements SwiGLU-style MLP with:
//! - Squared ReLU activation: x * ReLU(x) (required for ternary stability)
//! - Gate/Up/Down projections
//! - BitNet ternary quantization via bitnet-quantize crate

use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::bitnet::TritterLinear;
use crate::config::TritterConfig;

/// Squared ReLU activation: f(x) = x * ReLU(x)
/// Critical for BitNet ternary weight stability
pub fn squared_relu(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    x.mul(&relu)
}

/// SwiGLU-style MLP with Squared ReLU
/// FFN(x) = down(squared_relu(gate(x)) * up(x))
pub struct TritterMLP {
    gate_proj: TritterLinear,
    up_proj: TritterLinear,
    down_proj: TritterLinear,
    use_bitnet: bool,
}

impl TritterMLP {
    /// Create MLP module
    ///
    /// If `config.use_bitnet` is true, the projection layers will use
    /// BitNet ternary quantization for ~16x memory reduction.
    pub fn new(config: &TritterConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let use_bitnet = config.use_bitnet;

        let gate_proj =
            TritterLinear::new(hidden, intermediate, use_bitnet, vb.pp("gate_proj"), device)?;
        let up_proj =
            TritterLinear::new(hidden, intermediate, use_bitnet, vb.pp("up_proj"), device)?;
        let down_proj =
            TritterLinear::new(intermediate, hidden, use_bitnet, vb.pp("down_proj"), device)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            use_bitnet,
        })
    }

    /// Check if this MLP layer is using BitNet quantization.
    #[must_use]
    pub const fn is_bitnet(&self) -> bool {
        self.use_bitnet
    }

    /// Get the average compression ratio across all projections.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        (self.gate_proj.compression_ratio()
            + self.up_proj.compression_ratio()
            + self.down_proj.compression_ratio())
            / 3.0
    }

    /// Forward pass
    /// Input: (batch, seq_len, hidden_size)
    /// Output: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Gate with squared ReLU for BitNet stability
        let gate = squared_relu(&self.gate_proj.forward(x)?)?;

        // Up projection
        let up = self.up_proj.forward(x)?;

        // Element-wise multiply gate and up
        let hidden = gate.mul(&up)?;

        // Down projection back to hidden size
        self.down_proj.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_squared_relu() {
        let device = Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &device).unwrap();
        let out = squared_relu(&x).unwrap();
        let vals: Vec<f32> = out.to_vec1().unwrap();

        // Squared ReLU: negative inputs -> 0, positive inputs -> x^2
        assert_eq!(vals[0], 0.0); // -2 * 0 = 0
        assert_eq!(vals[1], 0.0); // -1 * 0 = 0
        assert_eq!(vals[2], 0.0); // 0 * 0 = 0
        assert_eq!(vals[3], 1.0); // 1 * 1 = 1
        assert_eq!(vals[4], 4.0); // 2 * 2 = 4
    }

    #[test]
    fn test_mlp_shape() {
        let config = crate::config::TritterConfig::test();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let mlp = TritterMLP::new(&config, vb, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, config.hidden_size), &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[2, 8, config.hidden_size]);
    }

    #[test]
    fn test_mlp_bitnet() {
        let mut config = crate::config::TritterConfig::test();
        config.use_bitnet = true;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let mlp = TritterMLP::new(&config, vb, &device).unwrap();
        assert!(mlp.is_bitnet());
        assert!(mlp.compression_ratio() > 1.0);

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, config.hidden_size), &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[2, 8, config.hidden_size]);
    }
}
