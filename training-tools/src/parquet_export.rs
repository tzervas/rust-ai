//! Parquet export for step metrics.
//!
//! Provides functionality to export training step metrics to Parquet format
//! for analysis in tools like pandas, DuckDB, or Apache Spark.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::training_state::StepMetrics;

/// Error type for parquet export operations.
#[derive(Debug, thiserror::Error)]
pub enum ParquetExportError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Arrow error
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Parquet error
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Empty metrics
    #[error("No metrics to export")]
    EmptyMetrics,
}

/// Build arrays from step metrics.
fn build_arrays(
    metrics: &[StepMetrics],
) -> (
    UInt64Array,
    Float32Array,
    Float32Array,
    StringArray,
    BooleanArray,
    Float32Array,
    Float64Array,
    Int64Array,
) {
    // For non-nullable columns, wrap in Some for Arrow's FromIterator
    let steps: UInt64Array = metrics.iter().map(|m| Some(m.step)).collect();
    let losses: Float32Array = metrics.iter().map(|m| Some(m.loss)).collect();
    let gradient_norms: Float32Array = metrics.iter().map(|m| Some(m.gradient_norm)).collect();
    let phases: StringArray = metrics.iter().map(|m| Some(m.phase.to_string())).collect();
    let was_predicted: BooleanArray = metrics.iter().map(|m| Some(m.was_predicted)).collect();
    // prediction_error is already Option<f32>
    let prediction_errors: Float32Array = metrics.iter().map(|m| m.prediction_error).collect();
    let step_times: Float64Array = metrics.iter().map(|m| Some(m.step_time_ms)).collect();
    let timestamps: Int64Array = metrics
        .iter()
        .map(|m| Some(m.timestamp.timestamp_millis()))
        .collect();

    (
        steps,
        losses,
        gradient_norms,
        phases,
        was_predicted,
        prediction_errors,
        step_times,
        timestamps,
    )
}

/// Create the schema for step metrics.
fn create_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("step", DataType::UInt64, false),
        Field::new("loss", DataType::Float32, false),
        Field::new("gradient_norm", DataType::Float32, false),
        Field::new("phase", DataType::Utf8, false),
        Field::new("was_predicted", DataType::Boolean, false),
        Field::new("prediction_error", DataType::Float32, true),
        Field::new("step_time_ms", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]))
}

/// Export step metrics to a Parquet file.
///
/// # Schema
///
/// The output Parquet file has the following schema:
/// - `step` (u64): Step number
/// - `loss` (f32): Loss value
/// - `gradient_norm` (f32): Gradient norm
/// - `phase` (string): Training phase (WARMUP, FULL, PREDICT, CORRECT)
/// - `was_predicted` (bool): Whether gradients were predicted
/// - `prediction_error` (f32, nullable): Prediction error if predicted
/// - `step_time_ms` (f64): Step duration in milliseconds
/// - `timestamp` (i64): Unix timestamp in milliseconds
///
/// # Example
///
/// ```no_run
/// use training_tools::parquet_export::export_metrics_to_parquet;
/// use training_tools::training_state::StepMetrics;
///
/// let metrics: Vec<StepMetrics> = vec![/* ... */];
/// export_metrics_to_parquet(&metrics, "metrics.parquet").unwrap();
/// ```
pub fn export_metrics_to_parquet<P: AsRef<Path>>(
    metrics: &[StepMetrics],
    path: P,
) -> Result<(), ParquetExportError> {
    if metrics.is_empty() {
        return Err(ParquetExportError::EmptyMetrics);
    }

    let schema = create_schema();
    let (
        steps,
        losses,
        gradient_norms,
        phases,
        was_predicted,
        prediction_errors,
        step_times,
        timestamps,
    ) = build_arrays(metrics);

    // Create record batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(steps) as Arc<dyn Array>,
            Arc::new(losses),
            Arc::new(gradient_norms),
            Arc::new(phases),
            Arc::new(was_predicted),
            Arc::new(prediction_errors),
            Arc::new(step_times),
            Arc::new(timestamps),
        ],
    )?;

    // Write to file with compression
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Export step metrics to Parquet with custom writer properties.
///
/// This variant allows specifying custom compression and other writer settings.
pub fn export_metrics_to_parquet_with_props<P: AsRef<Path>>(
    metrics: &[StepMetrics],
    path: P,
    props: WriterProperties,
) -> Result<(), ParquetExportError> {
    if metrics.is_empty() {
        return Err(ParquetExportError::EmptyMetrics);
    }

    let schema = create_schema();
    let (
        steps,
        losses,
        gradient_norms,
        phases,
        was_predicted,
        prediction_errors,
        step_times,
        timestamps,
    ) = build_arrays(metrics);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(steps) as Arc<dyn Array>,
            Arc::new(losses),
            Arc::new(gradient_norms),
            Arc::new(phases),
            Arc::new(was_predicted),
            Arc::new(prediction_errors),
            Arc::new(step_times),
            Arc::new(timestamps),
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_state::TrainingPhase;
    use chrono::Utc;
    use std::fs;
    use tempfile::tempdir;

    fn sample_metrics() -> Vec<StepMetrics> {
        vec![
            StepMetrics {
                step: 0,
                loss: 10.5,
                gradient_norm: 1.2,
                phase: TrainingPhase::Warmup,
                was_predicted: false,
                prediction_error: None,
                step_time_ms: 150.0,
                timestamp: Utc::now(),
                tokens_this_step: 2048,
                total_tokens_trained: 2048,
                tokens_remaining: 1_000_000,
                confidence: 0.0,
                learning_rate: 0.0,
                perplexity: 10.5_f32.exp(),
                train_val_gap: None,
                loss_velocity: 0.0,
                loss_acceleration: 0.0,
                gradient_entropy: None,
            },
            StepMetrics {
                step: 1,
                loss: 9.8,
                gradient_norm: 1.1,
                phase: TrainingPhase::Full,
                was_predicted: false,
                prediction_error: None,
                step_time_ms: 145.0,
                timestamp: Utc::now(),
                tokens_this_step: 2048,
                total_tokens_trained: 4096,
                tokens_remaining: 998_000,
                confidence: 0.0,
                learning_rate: 3e-4,
                perplexity: 9.8_f32.exp(),
                train_val_gap: None,
                loss_velocity: -0.05,
                loss_acceleration: 0.0,
                gradient_entropy: None,
            },
            StepMetrics {
                step: 2,
                loss: 9.2,
                gradient_norm: 0.9,
                phase: TrainingPhase::Predict,
                was_predicted: true,
                prediction_error: Some(0.05),
                step_time_ms: 50.0,
                timestamp: Utc::now(),
                tokens_this_step: 2048,
                total_tokens_trained: 6144,
                tokens_remaining: 996_000,
                confidence: 0.85,
                learning_rate: 3e-4,
                perplexity: 9.2_f32.exp(),
                train_val_gap: None,
                loss_velocity: -0.08,
                loss_acceleration: -0.003,
                gradient_entropy: Some(2.1),
            },
        ]
    }

    #[test]
    fn test_export_metrics_to_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metrics.parquet");

        let metrics = sample_metrics();
        export_metrics_to_parquet(&metrics, &path).unwrap();

        assert!(path.exists());
        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_empty_metrics_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.parquet");

        let result = export_metrics_to_parquet(&[], &path);
        assert!(matches!(result, Err(ParquetExportError::EmptyMetrics)));
    }
}
