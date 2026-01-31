//! Data loading utilities for Tritter training.
//!
//! This module provides memory-efficient streaming data loading from JSONL and Parquet files:
//! - [`DataConfig`] - Configuration for batch size, sequence length, workers
//! - [`JsonlDataset`] - Streaming iterator over JSONL files
//! - [`ParquetDataset`] - Streaming iterator over Parquet files (Arrow-based)
//! - [`DataLoader`] - Thread-safe batched iterator yielding [`TritterBatch`]
//! - Dynamic padding via [`collate_batch`]
//!
//! # Example
//!
//! ```no_run
//! use tritter_model_rs::data::{DataConfig, JsonlDataset, DataLoader};
//! use candle_core::Device;
//!
//! let config = DataConfig::default();
//! let dataset = JsonlDataset::new("data/train.jsonl", config.max_seq_length).unwrap();
//! let loader = DataLoader::new(Box::new(dataset), config, Device::Cpu);
//!
//! for batch in loader {
//!     let batch = batch.unwrap();
//!     println!("Batch size: {}, seq_len: {:?}", batch.input_ids.dims()[0], batch.input_ids.dims()[1]);
//! }
//! ```

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{TritterError, TritterResult};
use crate::trainer::TritterBatch;

/// Configuration for data loading.
///
/// Centralizes data hyperparameters for reproducibility and easy experimentation.
/// `max_seq_length` should match the model's `max_position_embeddings` but can be
/// reduced for memory-constrained training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Maximum sequence length (truncates longer sequences)
    pub max_seq_length: usize,
    /// Number of sequences per batch
    pub batch_size: usize,
    /// Number of worker threads for parallel loading
    pub num_workers: usize,
    /// Prefetch buffer size (batches to prefetch per worker)
    pub prefetch_factor: usize,
    /// Column name for text in JSONL/Parquet (auto-detected if None)
    pub text_column: Option<String>,
    /// Random seed for shuffling
    pub seed: u64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            max_seq_length: 2048,
            batch_size: 8,
            num_workers: 4,
            prefetch_factor: 2,
            text_column: None,
            seed: 42,
        }
    }
}

impl DataConfig {
    /// Create config for small-scale testing
    pub fn test() -> Self {
        Self {
            max_seq_length: 128,
            batch_size: 2,
            num_workers: 0,
            prefetch_factor: 1,
            text_column: None,
            seed: 42,
        }
    }

    /// Create config for training with custom batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Create config for training with custom sequence length
    pub fn with_max_seq_length(mut self, max_seq_length: usize) -> Self {
        self.max_seq_length = max_seq_length;
        self
    }

    /// Set the text column name
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }
}

/// A single tokenized example (before batching).
#[derive(Debug, Clone)]
pub struct TokenizedExample {
    /// Token IDs
    pub input_ids: Vec<u32>,
}

/// Trait for streaming datasets that yield tokenized examples.
pub trait StreamingDataset: Send {
    /// Get the next tokenized example, or None if exhausted.
    fn next_example(&mut self) -> Option<TritterResult<TokenizedExample>>;

    /// Reset the dataset iterator to the beginning.
    fn reset(&mut self) -> TritterResult<()>;

    /// Estimated total number of examples (may be approximate for streaming).
    fn len_hint(&self) -> Option<usize> {
        None
    }
}

/// Streaming dataset for JSONL files.
///
/// Reads JSONL files line by line, extracting text from the specified column
/// and tokenizing on the fly. Memory-efficient for large datasets.
pub struct JsonlDataset {
    /// Files to read from
    files: Vec<PathBuf>,
    /// Current file index
    current_file_idx: usize,
    /// Current file reader
    reader: Option<BufReader<File>>,
    /// Maximum sequence length
    max_seq_length: usize,
    /// Column name to extract text from
    text_column: Option<String>,
    /// Common text column names to auto-detect
    text_columns: Vec<String>,
}

impl JsonlDataset {
    /// Create a new JSONL streaming dataset.
    ///
    /// # Arguments
    /// * `path` - Path to JSONL file or directory of JSONL files
    /// * `max_seq_length` - Maximum sequence length (for truncation)
    pub fn new(path: impl AsRef<Path>, max_seq_length: usize) -> TritterResult<Self> {
        let path = path.as_ref();
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            Self::collect_jsonl_files(path)?
        } else {
            return Err(TritterError::Data(format!(
                "Path does not exist: {}",
                path.display()
            )));
        };

        if files.is_empty() {
            return Err(TritterError::Data(format!(
                "No JSONL files found in: {}",
                path.display()
            )));
        }

        let mut dataset = Self {
            files,
            current_file_idx: 0,
            reader: None,
            max_seq_length,
            text_column: None,
            text_columns: vec![
                "text".to_string(),
                "content".to_string(),
                "code".to_string(),
                "solution".to_string(),
                "output".to_string(),
                "response".to_string(),
                "input".to_string(),
                "question".to_string(),
            ],
        };

        dataset.open_current_file()?;
        Ok(dataset)
    }

    /// Set the text column name to use.
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }

    fn collect_jsonl_files(dir: &Path) -> TritterResult<Vec<PathBuf>> {
        let mut files = Vec::new();
        Self::collect_jsonl_files_recursive(dir, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn collect_jsonl_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> TritterResult<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            TritterError::Data(format!("Failed to read directory {}: {}", dir.display(), e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TritterError::Data(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();

            if path.is_dir() {
                Self::collect_jsonl_files_recursive(&path, files)?;
            } else if let Some(ext) = path.extension() {
                if ext == "jsonl" {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    fn open_current_file(&mut self) -> TritterResult<()> {
        if self.current_file_idx >= self.files.len() {
            self.reader = None;
            return Ok(());
        }

        let file_path = &self.files[self.current_file_idx];
        let file = File::open(file_path).map_err(|e| {
            TritterError::Data(format!("Failed to open file {}: {}", file_path.display(), e))
        })?;
        self.reader = Some(BufReader::new(file));
        Ok(())
    }

    fn extract_text(&self, data: &serde_json::Value) -> Option<String> {
        // Try specified column first
        if let Some(col) = &self.text_column {
            if let Some(text) = data.get(col).and_then(|v| v.as_str()) {
                return Some(text.to_string());
            }
        }

        // Auto-detect from common columns
        for col in &self.text_columns {
            if let Some(text) = data.get(col).and_then(|v| v.as_str()) {
                return Some(text.to_string());
            }
        }

        None
    }

    /// Simple byte-based tokenizer (placeholder for real tokenizer integration).
    /// In production, this should use a proper BPE/SentencePiece tokenizer.
    fn tokenize(&self, text: &str) -> Vec<u32> {
        // Simple UTF-8 byte tokenization as placeholder
        // Real implementation should integrate with a tokenizer
        let bytes: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        // Truncate if needed
        if bytes.len() > self.max_seq_length {
            bytes[..self.max_seq_length].to_vec()
        } else {
            bytes
        }
    }
}

impl StreamingDataset for JsonlDataset {
    fn next_example(&mut self) -> Option<TritterResult<TokenizedExample>> {
        loop {
            // Try to read from current file
            if let Some(reader) = &mut self.reader {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        // EOF, move to next file
                        self.current_file_idx += 1;
                        if let Err(e) = self.open_current_file() {
                            return Some(Err(e));
                        }
                        if self.reader.is_none() {
                            return None; // No more files
                        }
                        continue;
                    }
                    Ok(_) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        // Parse JSON
                        let data: serde_json::Value = match serde_json::from_str(line) {
                            Ok(v) => v,
                            Err(_) => continue, // Skip malformed JSON
                        };

                        // Extract text
                        let text = match self.extract_text(&data) {
                            Some(t) if !t.is_empty() => t,
                            _ => continue, // Skip if no text
                        };

                        // Tokenize
                        let input_ids = self.tokenize(&text);
                        if input_ids.is_empty() {
                            continue;
                        }

                        return Some(Ok(TokenizedExample { input_ids }));
                    }
                    Err(e) => {
                        return Some(Err(TritterError::Data(format!(
                            "Failed to read line: {}",
                            e
                        ))));
                    }
                }
            } else {
                return None;
            }
        }
    }

    fn reset(&mut self) -> TritterResult<()> {
        self.current_file_idx = 0;
        self.open_current_file()
    }
}

/// Streaming dataset for Parquet files.
///
/// Uses Arrow to efficiently read Parquet files in batches, extracting text
/// from the specified column. Memory-efficient for large datasets.
#[cfg(feature = "parquet")]
pub struct ParquetDataset {
    /// Files to read from
    files: Vec<PathBuf>,
    /// Current file index
    current_file_idx: usize,
    /// Current Arrow reader
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    /// Current batch being iterated
    current_batch: Option<arrow_array::RecordBatch>,
    /// Current row index in batch
    current_row: usize,
    /// Maximum sequence length
    max_seq_length: usize,
    /// Column name to extract text from
    text_column: Option<String>,
    /// Auto-detected text column name for current file
    detected_column: Option<String>,
    /// Common text column names to auto-detect
    text_columns: Vec<String>,
}

#[cfg(feature = "parquet")]
impl ParquetDataset {
    /// Create a new Parquet streaming dataset.
    ///
    /// # Arguments
    /// * `path` - Path to Parquet file or directory of Parquet files
    /// * `max_seq_length` - Maximum sequence length (for truncation)
    pub fn new(path: impl AsRef<Path>, max_seq_length: usize) -> TritterResult<Self> {
        let path = path.as_ref();
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            Self::collect_parquet_files(path)?
        } else {
            return Err(TritterError::Data(format!(
                "Path does not exist: {}",
                path.display()
            )));
        };

        if files.is_empty() {
            return Err(TritterError::Data(format!(
                "No Parquet files found in: {}",
                path.display()
            )));
        }

        let mut dataset = Self {
            files,
            current_file_idx: 0,
            reader: None,
            current_batch: None,
            current_row: 0,
            max_seq_length,
            text_column: None,
            detected_column: None,
            text_columns: vec![
                "text".to_string(),
                "content".to_string(),
                "code".to_string(),
                "solution".to_string(),
                "output".to_string(),
                "response".to_string(),
                "input".to_string(),
                "question".to_string(),
            ],
        };

        dataset.open_current_file()?;
        Ok(dataset)
    }

    /// Set the text column name to use.
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }

    fn collect_parquet_files(dir: &Path) -> TritterResult<Vec<PathBuf>> {
        let mut files = Vec::new();
        Self::collect_parquet_files_recursive(dir, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn collect_parquet_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> TritterResult<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            TritterError::Data(format!("Failed to read directory {}: {}", dir.display(), e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TritterError::Data(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();

            if path.is_dir() {
                Self::collect_parquet_files_recursive(&path, files)?;
            } else if let Some(ext) = path.extension() {
                if ext == "parquet" {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    fn open_current_file(&mut self) -> TritterResult<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        if self.current_file_idx >= self.files.len() {
            self.reader = None;
            self.current_batch = None;
            return Ok(());
        }

        let file_path = &self.files[self.current_file_idx];
        let file = File::open(file_path).map_err(|e| {
            TritterError::Data(format!("Failed to open file {}: {}", file_path.display(), e))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TritterError::Data(format!(
                "Failed to create Parquet reader for {}: {}",
                file_path.display(),
                e
            ))
        })?;

        // Detect text column from schema
        let schema = builder.schema();
        self.detected_column = self.detect_text_column(schema);

        if self.detected_column.is_none() && self.text_column.is_none() {
            // No text column found, skip this file
            self.current_file_idx += 1;
            return self.open_current_file();
        }

        // Build reader with batch size of 1024 rows
        let reader = builder.with_batch_size(1024).build().map_err(|e| {
            TritterError::Data(format!(
                "Failed to build Parquet reader for {}: {}",
                file_path.display(),
                e
            ))
        })?;

        self.reader = Some(reader);
        self.current_batch = None;
        self.current_row = 0;

        Ok(())
    }

    fn detect_text_column(&self, schema: &arrow_schema::SchemaRef) -> Option<String> {
        // Try specified column first
        if let Some(col) = &self.text_column {
            if schema.field_with_name(col).is_ok() {
                return Some(col.clone());
            }
        }

        // Auto-detect from common columns
        for col in &self.text_columns {
            if schema.field_with_name(col).is_ok() {
                return Some(col.clone());
            }
        }

        None
    }

    fn get_text_from_batch(&self, batch: &arrow_array::RecordBatch, row: usize) -> Option<String> {
        use arrow_array::cast::AsArray;
        use arrow_array::Array;

        let col_name = self.text_column.as_ref().or(self.detected_column.as_ref())?;

        let col_idx = batch.schema().index_of(col_name).ok()?;
        let column = batch.column(col_idx);

        // Try to get as string array
        if let Some(str_array) = column.as_string_opt::<i32>() {
            if row < str_array.len() && !str_array.is_null(row) {
                return Some(str_array.value(row).to_string());
            }
        }

        // Try large string array
        if let Some(str_array) = column.as_string_opt::<i64>() {
            if row < str_array.len() && !str_array.is_null(row) {
                return Some(str_array.value(row).to_string());
            }
        }

        None
    }

    /// Simple byte-based tokenizer (placeholder for real tokenizer integration).
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let bytes: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        if bytes.len() > self.max_seq_length {
            bytes[..self.max_seq_length].to_vec()
        } else {
            bytes
        }
    }
}

#[cfg(feature = "parquet")]
impl StreamingDataset for ParquetDataset {
    fn next_example(&mut self) -> Option<TritterResult<TokenizedExample>> {
        loop {
            // Try to read from current batch
            if let Some(batch) = &self.current_batch {
                if self.current_row < batch.num_rows() {
                    let text = self.get_text_from_batch(batch, self.current_row);
                    self.current_row += 1;

                    if let Some(text) = text {
                        if !text.trim().is_empty() {
                            let input_ids = self.tokenize(&text);
                            if !input_ids.is_empty() {
                                return Some(Ok(TokenizedExample { input_ids }));
                            }
                        }
                    }
                    continue;
                }
            }

            // Need to read next batch
            if let Some(reader) = &mut self.reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        self.current_batch = Some(batch);
                        self.current_row = 0;
                        continue;
                    }
                    Some(Err(e)) => {
                        return Some(Err(TritterError::Data(format!(
                            "Failed to read Parquet batch: {}",
                            e
                        ))));
                    }
                    None => {
                        // EOF, move to next file
                        self.current_file_idx += 1;
                        if let Err(e) = self.open_current_file() {
                            return Some(Err(e));
                        }
                        if self.reader.is_none() {
                            return None; // No more files
                        }
                        continue;
                    }
                }
            } else {
                return None;
            }
        }
    }

    fn reset(&mut self) -> TritterResult<()> {
        self.current_file_idx = 0;
        self.current_batch = None;
        self.current_row = 0;
        self.open_current_file()
    }
}

/// Thread-safe data loader that yields batches of [`TritterBatch`].
///
/// Supports parallel data loading with prefetching for high throughput.
/// Uses dynamic padding to pad sequences to the maximum length in each batch.
pub struct DataLoader {
    /// The underlying dataset (boxed for trait object)
    dataset: Arc<Mutex<Box<dyn StreamingDataset>>>,
    /// Configuration
    config: DataConfig,
    /// Device to create tensors on
    device: Device,
    /// Prefetch buffer (filled by background workers)
    buffer: Arc<Mutex<VecDeque<TritterResult<TritterBatch>>>>,
    /// Worker threads
    workers: Vec<JoinHandle<()>>,
    /// Flag to signal workers to stop
    stop_flag: Arc<Mutex<bool>>,
    /// Whether the dataset is exhausted
    exhausted: Arc<Mutex<bool>>,
}

impl DataLoader {
    /// Create a new data loader.
    ///
    /// # Arguments
    /// * `dataset` - Boxed streaming dataset
    /// * `config` - Data loading configuration
    /// * `device` - Device to create tensors on
    pub fn new(dataset: Box<dyn StreamingDataset>, config: DataConfig, device: Device) -> Self {
        let dataset = Arc::new(Mutex::new(dataset));
        let buffer = Arc::new(Mutex::new(VecDeque::new()));
        let stop_flag = Arc::new(Mutex::new(false));
        let exhausted = Arc::new(Mutex::new(false));
        let workers = Vec::new();

        let mut loader = Self {
            dataset,
            config,
            device,
            buffer,
            workers,
            stop_flag,
            exhausted,
        };

        // Start background workers if num_workers > 0
        if loader.config.num_workers > 0 {
            loader.start_workers();
        }

        loader
    }

    fn start_workers(&mut self) {
        let buffer_capacity = self.config.prefetch_factor * self.config.num_workers;

        for _ in 0..self.config.num_workers {
            let dataset = Arc::clone(&self.dataset);
            let buffer = Arc::clone(&self.buffer);
            let stop_flag = Arc::clone(&self.stop_flag);
            let exhausted = Arc::clone(&self.exhausted);
            let config = self.config.clone();
            let device = self.device.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Check stop flag
                    if *stop_flag.lock().unwrap() {
                        break;
                    }

                    // Check if buffer is full
                    {
                        let buf = buffer.lock().unwrap();
                        if buf.len() >= buffer_capacity {
                            drop(buf);
                            thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                    }

                    // Try to create a batch
                    let batch_result =
                        Self::create_batch_inner(&dataset, &config, &device, &exhausted);

                    match batch_result {
                        Some(batch) => {
                            let mut buf = buffer.lock().unwrap();
                            buf.push_back(batch);
                        }
                        None => {
                            // Dataset exhausted
                            *exhausted.lock().unwrap() = true;
                            break;
                        }
                    }
                }
            });

            self.workers.push(handle);
        }
    }

    fn create_batch_inner(
        dataset: &Arc<Mutex<Box<dyn StreamingDataset>>>,
        config: &DataConfig,
        device: &Device,
        _exhausted: &Arc<Mutex<bool>>,
    ) -> Option<TritterResult<TritterBatch>> {
        let mut examples = Vec::with_capacity(config.batch_size);

        // Collect batch_size examples
        {
            let mut ds = dataset.lock().unwrap();
            while examples.len() < config.batch_size {
                match ds.next_example() {
                    Some(Ok(ex)) => examples.push(ex),
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }
        }

        if examples.is_empty() {
            return None;
        }

        // Collate into batch
        Some(collate_batch(&examples, device))
    }

    /// Get the next batch (blocking).
    fn next_batch(&mut self) -> Option<TritterResult<TritterBatch>> {
        if self.config.num_workers > 0 {
            // Try to get from buffer
            loop {
                {
                    let mut buf = self.buffer.lock().unwrap();
                    if let Some(batch) = buf.pop_front() {
                        return Some(batch);
                    }
                }

                // Check if exhausted and buffer is empty
                if *self.exhausted.lock().unwrap() {
                    let buf = self.buffer.lock().unwrap();
                    if buf.is_empty() {
                        return None;
                    }
                }

                // Wait a bit for workers
                thread::sleep(std::time::Duration::from_millis(1));
            }
        } else {
            // Single-threaded mode
            Self::create_batch_inner(&self.dataset, &self.config, &self.device, &self.exhausted)
        }
    }

    /// Reset the data loader to iterate from the beginning.
    pub fn reset(&mut self) -> TritterResult<()> {
        // Stop workers
        *self.stop_flag.lock().unwrap() = true;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }

        // Clear buffer
        self.buffer.lock().unwrap().clear();

        // Reset dataset
        self.dataset.lock().unwrap().reset()?;

        // Reset flags
        *self.stop_flag.lock().unwrap() = false;
        *self.exhausted.lock().unwrap() = false;

        // Restart workers
        if self.config.num_workers > 0 {
            self.start_workers();
        }

        Ok(())
    }
}

impl Iterator for DataLoader {
    type Item = TritterResult<TritterBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

impl Drop for DataLoader {
    fn drop(&mut self) {
        // Signal workers to stop
        *self.stop_flag.lock().unwrap() = true;

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Collate a batch of tokenized examples with dynamic padding.
///
/// Pads sequences to the maximum length in the batch, not the global max.
/// This reduces wasted computation on padding tokens.
///
/// # Arguments
/// * `examples` - Slice of tokenized examples
/// * `device` - Device to create tensors on
///
/// # Returns
/// A [`TritterBatch`] with:
/// - `input_ids`: Padded tensor of shape `(batch_size, max_len_in_batch)`
/// - `attention_mask`: Binary mask where 1 = real token, 0 = padding
pub fn collate_batch(examples: &[TokenizedExample], device: &Device) -> TritterResult<TritterBatch> {
    if examples.is_empty() {
        return Err(TritterError::Data("Cannot collate empty batch".to_string()));
    }

    // Find max length in this batch
    let max_len = examples.iter().map(|ex| ex.input_ids.len()).max().unwrap();

    // Pad sequences and create attention masks
    let batch_size = examples.len();
    let mut input_ids_flat: Vec<u32> = Vec::with_capacity(batch_size * max_len);
    let mut attention_mask_flat: Vec<u8> = Vec::with_capacity(batch_size * max_len);

    for example in examples {
        let seq_len = example.input_ids.len();
        let padding_len = max_len - seq_len;

        // Add tokens
        input_ids_flat.extend_from_slice(&example.input_ids);
        // Add padding (token 0)
        input_ids_flat.extend(std::iter::repeat(0u32).take(padding_len));

        // Add attention mask (1 for real tokens, 0 for padding)
        attention_mask_flat.extend(std::iter::repeat(1u8).take(seq_len));
        attention_mask_flat.extend(std::iter::repeat(0u8).take(padding_len));
    }

    // Create tensors
    let input_ids = Tensor::from_slice(&input_ids_flat, (batch_size, max_len), device)?;

    // Convert attention mask to bool tensor
    let attention_mask_bool: Vec<u8> = attention_mask_flat;
    let attention_mask =
        Tensor::from_slice(&attention_mask_bool, (batch_size, max_len), device)?.to_dtype(DType::U8)?;

    Ok(TritterBatch::new(input_ids, Some(attention_mask)))
}

/// Create a data loader from a path, auto-detecting file format.
///
/// Supports:
/// - Single JSONL file
/// - Single Parquet file (requires `parquet` feature)
/// - Directory of JSONL files
/// - Directory of Parquet files (requires `parquet` feature)
pub fn create_data_loader(
    path: impl AsRef<Path>,
    config: DataConfig,
    device: Device,
) -> TritterResult<DataLoader> {
    let path = path.as_ref();

    // Determine file format
    let is_parquet = if path.is_file() {
        path.extension().map(|e| e == "parquet").unwrap_or(false)
    } else if path.is_dir() {
        // Check if directory contains parquet files
        std::fs::read_dir(path)
            .ok()
            .and_then(|entries| {
                entries.filter_map(|e| e.ok()).find(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "parquet")
                        .unwrap_or(false)
                })
            })
            .is_some()
    } else {
        false
    };

    if is_parquet {
        #[cfg(feature = "parquet")]
        {
            let mut dataset = ParquetDataset::new(path, config.max_seq_length)?;
            if let Some(col) = &config.text_column {
                dataset = dataset.with_text_column(col.clone());
            }
            Ok(DataLoader::new(Box::new(dataset), config, device))
        }

        #[cfg(not(feature = "parquet"))]
        {
            Err(TritterError::Data(
                "Parquet support requires the 'parquet' feature".to_string(),
            ))
        }
    } else {
        let mut dataset = JsonlDataset::new(path, config.max_seq_length)?;
        if let Some(col) = &config.text_column {
            dataset = dataset.with_text_column(col.clone());
        }
        Ok(DataLoader::new(Box::new(dataset), config, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_jsonl() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(file, r#"{{"text": "Hello world"}}"#).unwrap();
        writeln!(file, r#"{{"text": "Testing data loading"}}"#).unwrap();
        writeln!(file, r#"{{"text": "Rust is great"}}"#).unwrap();
        writeln!(file, r#"{{"content": "Alternative column name"}}"#).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_data_config_default() {
        let config = DataConfig::default();
        assert_eq!(config.max_seq_length, 2048);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.num_workers, 4);
    }

    #[test]
    fn test_data_config_builder() {
        let config = DataConfig::default()
            .with_batch_size(16)
            .with_max_seq_length(1024)
            .with_text_column("code");

        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_seq_length, 1024);
        assert_eq!(config.text_column, Some("code".to_string()));
    }

    #[test]
    fn test_jsonl_dataset_streaming() {
        let file = create_test_jsonl();
        let mut dataset = JsonlDataset::new(file.path(), 256).unwrap();

        let mut count = 0;
        while let Some(result) = dataset.next_example() {
            assert!(result.is_ok());
            let example = result.unwrap();
            assert!(!example.input_ids.is_empty());
            count += 1;
        }

        assert_eq!(count, 4); // 4 lines in test file
    }

    #[test]
    fn test_jsonl_dataset_reset() {
        let file = create_test_jsonl();
        let mut dataset = JsonlDataset::new(file.path(), 256).unwrap();

        // Exhaust dataset
        while dataset.next_example().is_some() {}

        // Reset and read again
        dataset.reset().unwrap();

        let mut count = 0;
        while dataset.next_example().is_some() {
            count += 1;
        }

        assert_eq!(count, 4);
    }

    #[test]
    fn test_collate_batch() {
        let examples = vec![
            TokenizedExample {
                input_ids: vec![1, 2, 3],
            },
            TokenizedExample {
                input_ids: vec![4, 5, 6, 7, 8],
            },
            TokenizedExample {
                input_ids: vec![9, 10],
            },
        ];

        let device = Device::Cpu;
        let batch = collate_batch(&examples, &device).unwrap();

        // Check shape
        assert_eq!(batch.input_ids.dims(), &[3, 5]); // batch_size=3, max_len=5

        // Check attention mask exists
        assert!(batch.attention_mask.is_some());
        let mask = batch.attention_mask.as_ref().unwrap();
        assert_eq!(mask.dims(), &[3, 5]);
    }

    #[test]
    fn test_data_loader_single_threaded() {
        let file = create_test_jsonl();
        let config = DataConfig::test().with_batch_size(2);
        let device = Device::Cpu;

        let dataset = JsonlDataset::new(file.path(), config.max_seq_length).unwrap();
        let loader = DataLoader::new(Box::new(dataset), config, device);

        let batches: Vec<_> = loader.collect();

        // 4 examples with batch_size=2 should yield 2 batches
        assert_eq!(batches.len(), 2);

        for batch in batches {
            assert!(batch.is_ok());
            let batch = batch.unwrap();
            assert_eq!(batch.input_ids.dims()[0], 2); // batch_size
        }
    }

    #[test]
    fn test_tokenized_example_truncation() {
        let file = create_test_jsonl();
        let mut dataset = JsonlDataset::new(file.path(), 5).unwrap(); // Very short max length

        let example = dataset.next_example().unwrap().unwrap();
        assert!(example.input_ids.len() <= 5);
    }
}
