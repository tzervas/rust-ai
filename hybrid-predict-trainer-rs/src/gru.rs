//! GRU (Gated Recurrent Unit) implementation with forward pass and training.
//!
//! This module provides a lightweight GRU implementation for the RSSM-lite
//! dynamics model. The GRU captures temporal dependencies in training dynamics,
//! enabling accurate multi-step prediction.
//!
//! # Why GRU over LSTM?
//!
//! GRUs are chosen for their:
//! - **Efficiency**: Fewer parameters than LSTM (2 gates vs 3)
//! - **Comparable performance**: For our use case, GRU matches LSTM accuracy
//! - **Simpler gradients**: Easier to train without vanishing gradients
//!
//! The gating mechanism allows the model to selectively remember or forget
//! past training dynamics, which is crucial for adapting to phase transitions.

use rand::Rng;

/// GRU cell with forward pass and training capabilities.
///
/// Implements the standard GRU equations:
/// - z = `σ(W_z·x` + `U_z·h` + `b_z`)  [update gate]
/// - r = `σ(W_r·x` + `U_r·h` + `b_r`)  [reset gate]
/// - h̃ = `tanh(W_h·x` + `U_h·(r⊙h)` + `b_h`)  (candidate)
/// - `h_new` = (1-z)⊙h + z⊙h̃  [new hidden state]
///
/// # Gate Purposes
///
/// - **Update gate (z)**: Controls how much of the new candidate state to use
///   vs retaining the previous hidden state. High z = more updating.
/// - **Reset gate (r)**: Controls how much of the previous hidden state to
///   expose when computing the candidate. Low r = forget more history.
pub struct GRUCell {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Input-to-hidden weights for update gate.
    pub w_z: Vec<f32>,
    /// Input-to-hidden weights for reset gate.
    pub w_r: Vec<f32>,
    /// Input-to-hidden weights for candidate.
    pub w_h: Vec<f32>,
    /// Hidden-to-hidden weights for update gate.
    pub u_z: Vec<f32>,
    /// Hidden-to-hidden weights for reset gate.
    pub u_r: Vec<f32>,
    /// Hidden-to-hidden weights for candidate.
    pub u_h: Vec<f32>,
    /// Biases for update gate.
    pub b_z: Vec<f32>,
    /// Biases for reset gate.
    pub b_r: Vec<f32>,
    /// Biases for candidate.
    pub b_h: Vec<f32>,
    /// Learning rate.
    pub learning_rate: f32,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f32,
}

impl GRUCell {
    /// Creates a new GRU cell with Xavier/Glorot initialization.
    ///
    /// Xavier initialization helps prevent vanishing/exploding gradients by
    /// initializing weights from U(-√(6/(fan_in + `fan_out`)), √(`6/(fan_in` + `fan_out`))).
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Dimension of hidden state
    /// * `learning_rate` - Learning rate for weight updates
    #[must_use]
    pub fn new(input_dim: usize, hidden_dim: usize, learning_rate: f32) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot uniform initialization
        let input_scale = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let recurrent_scale = (6.0 / (2.0 * hidden_dim as f32)).sqrt();

        let mut xavier_vec = |size: usize, scale: f32| -> Vec<f32> {
            (0..size).map(|_| rng.random_range(-scale..scale)).collect()
        };

        Self {
            input_dim,
            hidden_dim,
            w_z: xavier_vec(input_dim * hidden_dim, input_scale),
            w_r: xavier_vec(input_dim * hidden_dim, input_scale),
            w_h: xavier_vec(input_dim * hidden_dim, input_scale),
            u_z: xavier_vec(hidden_dim * hidden_dim, recurrent_scale),
            u_r: xavier_vec(hidden_dim * hidden_dim, recurrent_scale),
            u_h: xavier_vec(hidden_dim * hidden_dim, recurrent_scale),
            b_z: vec![0.0; hidden_dim],
            b_r: vec![0.0; hidden_dim],
            b_h: vec![0.0; hidden_dim],
            learning_rate,
            max_grad_norm: 5.0,
        }
    }

    /// Sigmoid activation: σ(x) = 1 / (1 + e^(-x)).
    ///
    /// Uses numerically stable implementation to avoid overflow.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }

    /// Tanh activation.
    #[inline]
    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    /// Matrix-vector multiplication for row-major matrices.
    ///
    /// Computes y = Wx where W is [`out_dim`, `in_dim`] stored row-major.
    fn matmul_vec(matrix: &[f32], vec: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
        for i in 0..out_dim {
            let row_start = i * in_dim;
            let row = &matrix[row_start..row_start + in_dim];
            out[i] = row.iter().zip(vec.iter()).map(|(&w, &x)| w * x).sum();
        }
    }

    /// Performs a single GRU forward step: `h_t` = `GRU(x_t`, h_{t-1}).
    ///
    /// # Arguments
    ///
    /// * `input` - Input features at timestep t, shape: \[`input_dim`\]
    /// * `hidden` - Hidden state from timestep t-1, shape: \[`hidden_dim`\]
    ///
    /// # Returns
    ///
    /// New hidden state at timestep t, shape: \[`hidden_dim`\]
    ///
    /// # Implementation
    ///
    /// Computes the GRU equations:
    /// 1. Update gate: z = `σ(W_z·x` + `U_z·h` + `b_z`) - controls how much to update
    /// 2. Reset gate: r = `σ(W_r·x` + `U_r·h` + `b_r`) - controls how much past to forget
    /// 3. Candidate: h̃ = `tanh(W_h·x` + `U_h·(r⊙h)` + `b_h`) - new candidate state
    /// 4. Output: `h_new` = (1-z)⊙h + z⊙h̃ - interpolate between old and new
    #[must_use]
    pub fn step(&self, input: &[f32], hidden: &[f32]) -> Vec<f32> {
        let h_dim = self.hidden_dim;
        let i_dim = self.input_dim;

        assert_eq!(input.len(), i_dim, "Input dimension mismatch");
        assert_eq!(hidden.len(), h_dim, "Hidden dimension mismatch");

        // Temporary buffers for intermediate computations
        let mut z_in = vec![0.0; h_dim];
        let mut z_h = vec![0.0; h_dim];
        let mut r_in = vec![0.0; h_dim];
        let mut r_h = vec![0.0; h_dim];
        let mut h_in = vec![0.0; h_dim];
        let mut h_h = vec![0.0; h_dim];

        // Update gate: z = σ(W_z·x + U_z·h + b_z)
        Self::matmul_vec(&self.w_z, input, &mut z_in, h_dim, i_dim);
        Self::matmul_vec(&self.u_z, hidden, &mut z_h, h_dim, h_dim);
        let z: Vec<f32> = (0..h_dim)
            .map(|i| Self::sigmoid(z_in[i] + z_h[i] + self.b_z[i]))
            .collect();

        // Reset gate: r = σ(W_r·x + U_r·h + b_r)
        Self::matmul_vec(&self.w_r, input, &mut r_in, h_dim, i_dim);
        Self::matmul_vec(&self.u_r, hidden, &mut r_h, h_dim, h_dim);
        let r: Vec<f32> = (0..h_dim)
            .map(|i| Self::sigmoid(r_in[i] + r_h[i] + self.b_r[i]))
            .collect();

        // Candidate hidden state: h̃ = tanh(W_h·x + U_h·(r⊙h) + b_h)
        let r_times_h: Vec<f32> = r.iter().zip(hidden.iter()).map(|(&r, &h)| r * h).collect();
        Self::matmul_vec(&self.w_h, input, &mut h_in, h_dim, i_dim);
        Self::matmul_vec(&self.u_h, &r_times_h, &mut h_h, h_dim, h_dim);
        let h_tilde: Vec<f32> = (0..h_dim)
            .map(|i| Self::tanh(h_in[i] + h_h[i] + self.b_h[i]))
            .collect();

        // New hidden state: h_new = (1-z)⊙h + z⊙h̃
        (0..h_dim)
            .map(|i| (1.0 - z[i]) * hidden[i] + z[i] * h_tilde[i])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_creation() {
        let gru = GRUCell::new(10, 20, 0.01);
        assert_eq!(gru.input_dim, 10);
        assert_eq!(gru.hidden_dim, 20);
        assert_eq!(gru.w_z.len(), 200);
        assert_eq!(gru.u_z.len(), 400);
        assert_eq!(gru.b_z.len(), 20);
    }

    #[test]
    fn test_xavier_initialization() {
        let gru = GRUCell::new(10, 20, 0.01);
        let input_scale = (6.0 / 30.0f32).sqrt();

        // Check that weights are in expected range
        for &w in &gru.w_z {
            assert!(w.abs() <= input_scale * 1.1); // Allow small tolerance
        }
    }

    #[test]
    fn test_gru_step() {
        let gru = GRUCell::new(5, 10, 0.01);
        let input = vec![0.5; 5];
        let hidden = vec![0.0; 10];

        let new_hidden = gru.step(&input, &hidden);

        assert_eq!(new_hidden.len(), 10);
        // New hidden state should be non-zero after processing input
        assert!(new_hidden.iter().any(|&h| h.abs() > 0.0));
    }

    #[test]
    fn test_sigmoid_activation() {
        assert!((GRUCell::sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(GRUCell::sigmoid(100.0) > 0.99);
        assert!(GRUCell::sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_tanh_activation() {
        assert!((GRUCell::tanh(0.0) - 0.0).abs() < 1e-6);
        assert!(GRUCell::tanh(100.0) > 0.99);
        assert!(GRUCell::tanh(-100.0) < -0.99);
    }

    #[test]
    fn test_matmul_vec() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let vec = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 2];

        GRUCell::matmul_vec(&matrix, &vec, &mut out, 2, 3);

        assert_eq!(out[0], 6.0); // 1+2+3
        assert_eq!(out[1], 15.0); // 4+5+6
    }

    #[test]
    fn test_gru_determinism() {
        let gru = GRUCell::new(3, 4, 0.01);
        let input = vec![0.5, 0.3, 0.7];
        let hidden = vec![0.1, 0.2, 0.3, 0.4];

        let h1 = gru.step(&input, &hidden);
        let h2 = gru.step(&input, &hidden);

        // Same input should produce same output
        assert_eq!(h1, h2);
    }
}
