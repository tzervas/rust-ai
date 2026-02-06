//! Output Prediction Analysis Module
//!
//! Analyzes model output logits and generates prediction visualizations.

use super::{InferenceVizError, Result};
use serde::{Deserialize, Serialize};

/// Entry for a single prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEntry {
    /// Token ID
    pub token_id: usize,
    /// Token text (if vocabulary is available)
    pub token_text: String,
    /// Raw logit value
    pub logit: f32,
    /// Probability (after softmax)
    pub probability: f32,
    /// Log probability
    pub log_prob: f32,
    /// Rank among all tokens
    pub rank: usize,
}

/// Comprehensive output analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputAnalysis {
    /// Top K predictions
    pub top_predictions: Vec<PredictionEntry>,
    /// Total number of tokens in vocabulary
    pub vocab_size: usize,
    /// Entropy of the probability distribution
    pub entropy: f32,
    /// Perplexity
    pub perplexity: f32,
    /// Max logit value
    pub max_logit: f32,
    /// Min logit value
    pub min_logit: f32,
    /// Temperature needed to achieve uniform distribution
    pub effective_temperature: f32,
    /// Confidence (probability of top prediction)
    pub confidence: f32,
    /// Probability mass in top 10
    pub top_10_mass: f32,
}

/// Get top K predictions from logits.
pub fn get_top_k_predictions(
    logits: &[f32],
    k: usize,
    vocab: Option<&[String]>,
) -> Result<Vec<(String, f32)>> {
    if logits.is_empty() {
        return Err(InferenceVizError::NoOutputLogits);
    }

    // Compute softmax
    let probabilities = softmax(logits);

    // Get top K
    let mut indexed: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let top_k: Vec<(String, f32)> = indexed
        .into_iter()
        .take(k)
        .map(|(idx, prob)| {
            let text = vocab
                .and_then(|v| v.get(idx).cloned())
                .unwrap_or_else(|| format!("token_{}", idx));
            (text, prob)
        })
        .collect();

    Ok(top_k)
}

/// Analyze output logits comprehensively.
pub fn analyze_output(
    logits: &[f32],
    vocab: Option<&[String]>,
    top_k: usize,
) -> Result<OutputAnalysis> {
    if logits.is_empty() {
        return Err(InferenceVizError::NoOutputLogits);
    }

    let vocab_size = logits.len();

    // Find min and max logits
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);

    // Compute softmax probabilities
    let probabilities = softmax(logits);

    // Sort by probability to get rankings
    let mut indexed: Vec<(usize, f32, f32)> = logits
        .iter()
        .zip(probabilities.iter())
        .enumerate()
        .map(|(i, (&logit, &prob))| (i, logit, prob))
        .collect();

    indexed.sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Build top predictions
    let top_predictions: Vec<PredictionEntry> = indexed
        .iter()
        .take(top_k)
        .enumerate()
        .map(|(rank, &(token_id, logit, probability))| {
            let token_text = vocab
                .and_then(|v| v.get(token_id).cloned())
                .unwrap_or_else(|| format!("token_{}", token_id));

            let log_prob = if probability > 0.0 {
                probability.ln()
            } else {
                f32::NEG_INFINITY
            };

            PredictionEntry {
                token_id,
                token_text,
                logit,
                probability,
                log_prob,
                rank,
            }
        })
        .collect();

    // Compute entropy: H = -sum(p * log(p))
    let entropy = -probabilities
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>();

    // Perplexity = exp(entropy)
    let perplexity = entropy.exp();

    // Confidence = top probability
    let confidence = top_predictions
        .first()
        .map(|p| p.probability)
        .unwrap_or(0.0);

    // Top 10 probability mass
    let top_10_mass: f32 = indexed.iter().take(10).map(|(_, _, p)| p).sum();

    // Effective temperature: the temperature that would make the distribution uniform
    // For uniform distribution over V tokens, entropy = ln(V)
    // If current entropy < ln(V), we need higher temperature
    let max_entropy = (vocab_size as f32).ln();
    let effective_temperature = if entropy < max_entropy {
        max_entropy / entropy
    } else {
        1.0
    };

    Ok(OutputAnalysis {
        top_predictions,
        vocab_size,
        entropy,
        perplexity,
        max_logit,
        min_logit,
        effective_temperature,
        confidence,
        top_10_mass,
    })
}

/// Compute softmax of logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    // Subtract max for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();

    if sum_exp < 1e-10 {
        // Avoid division by zero
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp_logits.iter().map(|&e| e / sum_exp).collect()
    }
}

/// Compute log softmax of logits.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max_logit).collect();
    let log_sum_exp = shifted.iter().map(|&s| s.exp()).sum::<f32>().ln();

    shifted.iter().map(|&s| s - log_sum_exp).collect()
}

/// Apply temperature scaling to logits.
pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 {
        // Temperature <= 0 returns argmax as one-hot
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut result = vec![0.0; logits.len()];
        result[max_idx] = 1.0;
        return result;
    }

    logits.iter().map(|&l| l / temperature).collect()
}

/// Compute nucleus (top-p) sampling threshold.
pub fn nucleus_threshold(probabilities: &[f32], p: f32) -> (Vec<usize>, f32) {
    let mut indexed: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut selected_indices = Vec::new();

    for (idx, prob) in indexed {
        cumulative += prob;
        selected_indices.push(idx);
        if cumulative >= p {
            break;
        }
    }

    (selected_indices, cumulative)
}

/// Compare two output distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison {
    /// KL divergence from p to q
    pub kl_divergence: f32,
    /// Jensen-Shannon divergence
    pub js_divergence: f32,
    /// Total variation distance
    pub tv_distance: f32,
    /// Correlation between probabilities
    pub correlation: f32,
    /// Number of tokens where ranking differs
    pub rank_disagreements: usize,
    /// Top prediction agreement
    pub top_1_agreement: bool,
    /// Top 5 prediction overlap
    pub top_5_overlap: f32,
}

/// Compare two probability distributions.
pub fn compare_distributions(p: &[f32], q: &[f32]) -> Option<DistributionComparison> {
    if p.len() != q.len() || p.is_empty() {
        return None;
    }

    let n = p.len();

    // KL divergence: sum(p * log(p/q))
    let kl_divergence: f32 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 1e-10 && qi > 1e-10)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum();

    // Jensen-Shannon divergence: 0.5 * (KL(p||m) + KL(q||m)) where m = 0.5*(p+q)
    let m: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| 0.5 * (pi + qi))
        .collect();
    let kl_pm: f32 = p
        .iter()
        .zip(m.iter())
        .filter(|(&pi, &mi)| pi > 1e-10 && mi > 1e-10)
        .map(|(&pi, &mi)| pi * (pi / mi).ln())
        .sum();
    let kl_qm: f32 = q
        .iter()
        .zip(m.iter())
        .filter(|(&qi, &mi)| qi > 1e-10 && mi > 1e-10)
        .map(|(&qi, &mi)| qi * (qi / mi).ln())
        .sum();
    let js_divergence = 0.5 * (kl_pm + kl_qm);

    // Total variation distance: 0.5 * sum(|p - q|)
    let tv_distance: f32 = 0.5
        * p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi - qi).abs())
            .sum::<f32>();

    // Correlation
    let mean_p: f32 = p.iter().sum::<f32>() / n as f32;
    let mean_q: f32 = q.iter().sum::<f32>() / n as f32;
    let var_p: f32 = p.iter().map(|&pi| (pi - mean_p).powi(2)).sum::<f32>();
    let var_q: f32 = q.iter().map(|&qi| (qi - mean_q).powi(2)).sum::<f32>();
    let cov: f32 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - mean_p) * (qi - mean_q))
        .sum();
    let correlation = if var_p > 1e-10 && var_q > 1e-10 {
        cov / (var_p.sqrt() * var_q.sqrt())
    } else {
        0.0
    };

    // Rankings
    let mut rank_p: Vec<(usize, f32)> = p.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut rank_q: Vec<(usize, f32)> = q.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    rank_p.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    rank_q.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Top 1 agreement
    let top_1_agreement = rank_p.first().map(|(i, _)| *i) == rank_q.first().map(|(i, _)| *i);

    // Top 5 overlap
    let top_5_p: std::collections::HashSet<usize> =
        rank_p.iter().take(5).map(|(i, _)| *i).collect();
    let top_5_q: std::collections::HashSet<usize> =
        rank_q.iter().take(5).map(|(i, _)| *i).collect();
    let top_5_overlap = top_5_p.intersection(&top_5_q).count() as f32 / 5.0;

    // Rank disagreements (positions where different tokens are ranked)
    let rank_map_p: std::collections::HashMap<usize, usize> = rank_p
        .iter()
        .enumerate()
        .map(|(rank, (idx, _))| (*idx, rank))
        .collect();
    let rank_disagreements = rank_q
        .iter()
        .enumerate()
        .filter(|(rank, (idx, _))| {
            rank_map_p
                .get(idx)
                .map(|&rp| (rp as i32 - *rank as i32).abs() > 5)
                .unwrap_or(true)
        })
        .count();

    Some(DistributionComparison {
        kl_divergence,
        js_divergence,
        tv_distance,
        correlation,
        rank_disagreements,
        top_1_agreement,
        top_5_overlap,
    })
}

/// Calibration analysis for model predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Bin index
    pub bin_idx: usize,
    /// Bin range (min, max)
    pub range: (f32, f32),
    /// Average confidence in this bin
    pub avg_confidence: f32,
    /// Accuracy in this bin (if labels available)
    pub accuracy: Option<f32>,
    /// Number of samples in this bin
    pub count: usize,
}

/// Compute calibration bins from predictions.
pub fn compute_calibration_bins(
    confidences: &[f32],
    correct: Option<&[bool]>,
    num_bins: usize,
) -> Vec<CalibrationBin> {
    let mut bins: Vec<CalibrationBin> = (0..num_bins)
        .map(|i| CalibrationBin {
            bin_idx: i,
            range: (i as f32 / num_bins as f32, (i + 1) as f32 / num_bins as f32),
            avg_confidence: 0.0,
            accuracy: None,
            count: 0,
        })
        .collect();

    // Populate bins
    for (i, &conf) in confidences.iter().enumerate() {
        let bin_idx = ((conf * num_bins as f32) as usize).min(num_bins - 1);
        bins[bin_idx].count += 1;
        bins[bin_idx].avg_confidence += conf;

        if let Some(correct_vec) = correct {
            if let Some(&is_correct) = correct_vec.get(i) {
                let acc = bins[bin_idx].accuracy.get_or_insert(0.0);
                if is_correct {
                    *acc += 1.0;
                }
            }
        }
    }

    // Finalize averages
    for bin in &mut bins {
        if bin.count > 0 {
            bin.avg_confidence /= bin.count as f32;
            if let Some(ref mut acc) = bin.accuracy {
                *acc /= bin.count as f32;
            }
        }
    }

    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Probabilities should be ordered
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large logits that could overflow without stabilization
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_top_k_predictions() {
        let logits = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let vocab = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        let top = get_top_k_predictions(&logits, 3, Some(&vocab)).unwrap();

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "b"); // Highest logit
        assert_eq!(top[1].0, "d"); // Second highest
    }

    #[test]
    fn test_analyze_output() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let analysis = analyze_output(&logits, None, 3).unwrap();

        assert_eq!(analysis.vocab_size, 5);
        assert_eq!(analysis.top_predictions.len(), 3);
        assert!(analysis.entropy > 0.0);
        assert!(analysis.perplexity > 1.0);
        assert!(analysis.confidence > 0.0 && analysis.confidence <= 1.0);
    }

    #[test]
    fn test_nucleus_threshold() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let (indices, cumulative) = nucleus_threshold(&probs, 0.9);

        assert!(cumulative >= 0.9);
        assert!(indices.contains(&0)); // Top probability
    }

    #[test]
    fn test_apply_temperature() {
        let logits = vec![1.0, 2.0, 3.0];

        // Temperature 1.0 should not change logits
        let temp1 = apply_temperature(&logits, 1.0);
        assert_eq!(temp1, logits);

        // Higher temperature flattens distribution
        let temp2 = apply_temperature(&logits, 2.0);
        assert!((temp2[2] - temp2[0]) < (logits[2] - logits[0]));
    }

    #[test]
    fn test_compare_distributions() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.5, 0.3, 0.2];

        let comparison = compare_distributions(&p, &q).unwrap();
        assert!(comparison.kl_divergence.abs() < 1e-6); // Same distribution
        assert!(comparison.correlation > 0.99);
        assert!(comparison.top_1_agreement);
    }

    #[test]
    fn test_calibration_bins() {
        let confidences = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let correct = vec![false, true, true, false, true];

        let bins = compute_calibration_bins(&confidences, Some(&correct), 5);

        assert_eq!(bins.len(), 5);
        assert!(bins[0].count > 0);
    }
}
