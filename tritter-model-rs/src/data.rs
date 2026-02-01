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
use tokenizers::Tokenizer;

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
///
/// # Tokenization
///
/// By default, uses simple UTF-8 byte tokenization. For production training,
/// set a real tokenizer via [`with_tokenizer`]:
///
/// ```no_run
/// use tritter_model_rs::data::JsonlDataset;
/// use tokenizers::Tokenizer;
/// use std::sync::Arc;
///
/// let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
/// let dataset = JsonlDataset::new("data/train.jsonl", 2048)
///     .unwrap()
///     .with_tokenizer(Arc::new(tokenizer));
/// ```
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
    /// Optional tokenizer for real tokenization
    tokenizer: Option<Arc<Tokenizer>>,
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
            tokenizer: None,
        };

        dataset.open_current_file()?;
        Ok(dataset)
    }

    /// Set the text column name to use.
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }

    /// Set a tokenizer for real tokenization.
    ///
    /// When a tokenizer is set, text will be tokenized using the provided
    /// tokenizer instead of the simple byte tokenization fallback.
    ///
    /// # Arguments
    /// * `tokenizer` - Arc-wrapped tokenizer instance
    ///
    /// # Example
    /// ```no_run
    /// use tritter_model_rs::data::JsonlDataset;
    /// use tokenizers::Tokenizer;
    /// use std::sync::Arc;
    ///
    /// let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    /// let dataset = JsonlDataset::new("data.jsonl", 2048)
    ///     .unwrap()
    ///     .with_tokenizer(Arc::new(tokenizer));
    /// ```
    pub fn with_tokenizer(mut self, tokenizer: Arc<Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Check if a tokenizer is configured.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
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

    /// Tokenize text using the configured tokenizer or byte fallback.
    ///
    /// If a tokenizer is set via `with_tokenizer()`, uses real BPE tokenization.
    /// Otherwise, falls back to simple UTF-8 byte tokenization.
    fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(tokenizer) = &self.tokenizer {
            // Use real tokenizer
            match tokenizer.encode(text, false) {
                Ok(encoding) => {
                    let ids: Vec<u32> = encoding.get_ids().to_vec();
                    // Truncate if needed
                    if ids.len() > self.max_seq_length {
                        ids[..self.max_seq_length].to_vec()
                    } else {
                        ids
                    }
                }
                Err(_) => {
                    // Fallback to byte tokenization on error
                    self.tokenize_bytes(text)
                }
            }
        } else {
            // Fallback to byte tokenization
            self.tokenize_bytes(text)
        }
    }

    /// Simple byte-based tokenizer (fallback when no tokenizer is configured).
    fn tokenize_bytes(&self, text: &str) -> Vec<u32> {
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
///
/// # Tokenization
///
/// By default, uses simple UTF-8 byte tokenization. For production training,
/// set a real tokenizer via [`with_tokenizer`]:
///
/// ```no_run
/// use tritter_model_rs::data::ParquetDataset;
/// use tokenizers::Tokenizer;
/// use std::sync::Arc;
///
/// let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
/// let dataset = ParquetDataset::new("data/train.parquet", 2048)
///     .unwrap()
///     .with_tokenizer(Arc::new(tokenizer));
/// ```
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
    /// Optional tokenizer for real tokenization
    tokenizer: Option<Arc<Tokenizer>>,
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
            tokenizer: None,
        };

        dataset.open_current_file()?;
        Ok(dataset)
    }

    /// Set the text column name to use.
    pub fn with_text_column(mut self, column: impl Into<String>) -> Self {
        self.text_column = Some(column.into());
        self
    }

    /// Set a tokenizer for real tokenization.
    ///
    /// When a tokenizer is set, text will be tokenized using the provided
    /// tokenizer instead of the simple byte tokenization fallback.
    pub fn with_tokenizer(mut self, tokenizer: Arc<Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Check if a tokenizer is configured.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
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

    /// Tokenize text using the configured tokenizer or byte fallback.
    fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(tokenizer) = &self.tokenizer {
            // Use real tokenizer
            match tokenizer.encode(text, false) {
                Ok(encoding) => {
                    let ids: Vec<u32> = encoding.get_ids().to_vec();
                    if ids.len() > self.max_seq_length {
                        ids[..self.max_seq_length].to_vec()
                    } else {
                        ids
                    }
                }
                Err(_) => self.tokenize_bytes(text),
            }
        } else {
            self.tokenize_bytes(text)
        }
    }

    /// Simple byte-based tokenizer (fallback when no tokenizer is configured).
    fn tokenize_bytes(&self, text: &str) -> Vec<u32> {
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

        // Collate into batch with max_seq_length enforcement
        Some(collate_batch_with_max_len(&examples, device, Some(config.max_seq_length)))
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
    collate_batch_with_max_len(examples, device, None)
}

/// Collate a batch of tokenized examples with dynamic padding and optional max length enforcement.
///
/// Truncates sequences to `max_seq_length` if provided, then pads to the maximum length in the batch.
/// This ensures tensor dimensions stay within model limits while reducing wasted computation.
///
/// # Arguments
/// * `examples` - Slice of tokenized examples
/// * `device` - Device to create tensors on
/// * `max_seq_length` - Optional maximum sequence length (truncates longer sequences)
///
/// # Returns
/// A [`TritterBatch`] with:
/// - `input_ids`: Padded tensor of shape `(batch_size, effective_max_len)`
/// - `attention_mask`: Binary mask where 1 = real token, 0 = padding
///
/// # Shape Guarantees
/// - Output `input_ids` shape: `[batch_size, seq_len]` where `seq_len <= max_seq_length`
/// - Output `attention_mask` shape: same as `input_ids`
pub fn collate_batch_with_max_len(
    examples: &[TokenizedExample],
    device: &Device,
    max_seq_length: Option<usize>,
) -> TritterResult<TritterBatch> {
    if examples.is_empty() {
        return Err(TritterError::Data("Cannot collate empty batch".to_string()));
    }

    // Find max length in this batch, respecting max_seq_length limit
    let batch_max_len = examples.iter().map(|ex| ex.input_ids.len()).max().unwrap();
    let max_len = match max_seq_length {
        Some(limit) => batch_max_len.min(limit),
        None => batch_max_len,
    };

    // Ensure we have at least 2 tokens for next-token prediction
    let max_len = max_len.max(2);

    // Pad sequences and create attention masks
    let batch_size = examples.len();
    let mut input_ids_flat: Vec<u32> = Vec::with_capacity(batch_size * max_len);
    let mut attention_mask_flat: Vec<u8> = Vec::with_capacity(batch_size * max_len);

    for example in examples {
        // Truncate sequence if it exceeds max_len
        let seq_len = example.input_ids.len().min(max_len);
        let padding_len = max_len - seq_len;

        // Add tokens (truncated if necessary)
        input_ids_flat.extend_from_slice(&example.input_ids[..seq_len]);
        // Add padding (token 0)
        input_ids_flat.extend(std::iter::repeat(0u32).take(padding_len));

        // Add attention mask (1 for real tokens, 0 for padding)
        attention_mask_flat.extend(std::iter::repeat(1u8).take(seq_len));
        attention_mask_flat.extend(std::iter::repeat(0u8).take(padding_len));
    }

    // Create tensors with shape [batch_size, seq_len]
    let input_ids = Tensor::from_slice(&input_ids_flat, (batch_size, max_len), device)?;

    // Convert attention mask to U8 tensor
    let attention_mask =
        Tensor::from_slice(&attention_mask_flat, (batch_size, max_len), device)?.to_dtype(DType::U8)?;

    // Validate tensor shapes before returning
    let input_dims = input_ids.dims();
    if input_dims.len() != 2 {
        return Err(TritterError::Data(format!(
            "Expected input_ids to have 2 dimensions [batch_size, seq_len], got {:?}",
            input_dims
        )));
    }
    if input_dims[0] != batch_size || input_dims[1] != max_len {
        return Err(TritterError::Data(format!(
            "input_ids shape mismatch: expected [{}, {}], got {:?}",
            batch_size, max_len, input_dims
        )));
    }

    Ok(TritterBatch::new(input_ids, Some(attention_mask)))
}

/// Strategy for blending multiple datasets.
#[derive(Debug, Clone, Copy, Default)]
pub enum DatasetBlendingStrategy {
    /// Process each dataset sequentially until exhausted, then move to next.
    #[default]
    Sequential,
    /// Cycle through datasets in round-robin fashion.
    RoundRobin,
    /// Interleave datasets with weighted sampling probabilities.
    /// Weights should sum to 1.0 for proper distribution.
    Weighted,
}

/// Composite dataset that combines multiple streaming datasets.
///
/// Supports different blending strategies for training on heterogeneous data sources:
/// - Code repositories (largest portion)
/// - Instruction-following data
/// - Alignment data
/// - Domain-specific data (e.g., IaC)
///
/// # Example
/// ```no_run
/// use tritter_model_rs::data::{CompositeDataset, JsonlDataset, DatasetBlendingStrategy};
///
/// let code_ds = JsonlDataset::new("/data/code", 2048).unwrap();
/// let inst_ds = JsonlDataset::new("/data/instruction", 2048).unwrap();
///
/// let composite = CompositeDataset::new(vec![
///     Box::new(code_ds),
///     Box::new(inst_ds),
/// ]).with_strategy(DatasetBlendingStrategy::RoundRobin);
/// ```
pub struct CompositeDataset {
    /// Child datasets
    datasets: Vec<Box<dyn StreamingDataset>>,
    /// Current dataset index
    current_idx: usize,
    /// Blending strategy
    strategy: DatasetBlendingStrategy,
    /// Weights for weighted sampling (must sum to ~1.0)
    weights: Vec<f32>,
    /// RNG for weighted sampling
    rng_state: u64,
    /// Number of exhausted datasets (for sequential mode)
    exhausted_count: usize,
}

impl CompositeDataset {
    /// Create a new composite dataset from multiple child datasets.
    pub fn new(datasets: Vec<Box<dyn StreamingDataset>>) -> Self {
        let n = datasets.len();
        let weights = vec![1.0 / n as f32; n]; // Equal weights by default
        Self {
            datasets,
            current_idx: 0,
            strategy: DatasetBlendingStrategy::default(),
            weights,
            rng_state: 42,
            exhausted_count: 0,
        }
    }

    /// Set the blending strategy.
    pub fn with_strategy(mut self, strategy: DatasetBlendingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set weights for weighted sampling.
    /// Weights should roughly sum to 1.0 but will be normalized if not.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            self.weights = weights.iter().map(|w| w / sum).collect();
        }
        self
    }

    /// Get the number of child datasets.
    pub fn num_datasets(&self) -> usize {
        self.datasets.len()
    }

    /// Simple xorshift RNG for weighted sampling.
    fn next_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    /// Select next dataset index based on strategy.
    fn select_next_dataset(&mut self) -> Option<usize> {
        if self.datasets.is_empty() {
            return None;
        }

        match self.strategy {
            DatasetBlendingStrategy::Sequential => {
                // Already handled in next_example
                Some(self.current_idx)
            }
            DatasetBlendingStrategy::RoundRobin => {
                let idx = self.current_idx;
                self.current_idx = (self.current_idx + 1) % self.datasets.len();
                Some(idx)
            }
            DatasetBlendingStrategy::Weighted => {
                let r = self.next_random();
                let mut cumulative = 0.0;
                for (i, &w) in self.weights.iter().enumerate() {
                    cumulative += w;
                    if r < cumulative {
                        return Some(i);
                    }
                }
                // Fallback to last dataset
                Some(self.datasets.len() - 1)
            }
        }
    }
}

impl StreamingDataset for CompositeDataset {
    fn next_example(&mut self) -> Option<TritterResult<TokenizedExample>> {
        if self.datasets.is_empty() || self.exhausted_count >= self.datasets.len() {
            return None;
        }

        match self.strategy {
            DatasetBlendingStrategy::Sequential => {
                // Try current dataset
                loop {
                    if self.current_idx >= self.datasets.len() {
                        return None;
                    }

                    if let Some(result) = self.datasets[self.current_idx].next_example() {
                        return Some(result);
                    }

                    // Current dataset exhausted, move to next
                    self.current_idx += 1;
                    self.exhausted_count += 1;
                }
            }
            DatasetBlendingStrategy::RoundRobin | DatasetBlendingStrategy::Weighted => {
                // Try up to N datasets before giving up
                let max_attempts = self.datasets.len() * 2;
                for _ in 0..max_attempts {
                    if let Some(idx) = self.select_next_dataset() {
                        if let Some(result) = self.datasets[idx].next_example() {
                            return Some(result);
                        }
                        // This dataset exhausted, try another
                    }
                }
                None
            }
        }
    }

    fn reset(&mut self) -> TritterResult<()> {
        self.current_idx = 0;
        self.exhausted_count = 0;
        for ds in &mut self.datasets {
            ds.reset()?;
        }
        Ok(())
    }

    fn len_hint(&self) -> Option<usize> {
        // Sum of all child hints
        let mut total = 0usize;
        for ds in &self.datasets {
            if let Some(len) = ds.len_hint() {
                total += len;
            } else {
                return None; // Can't determine if any child is unknown
            }
        }
        Some(total)
    }
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

    #[test]
    fn test_collate_batch_with_max_len_truncation() {
        // Create examples that exceed the max length
        let examples = vec![
            TokenizedExample {
                input_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // 10 tokens
            },
            TokenizedExample {
                input_ids: vec![11, 12, 13, 14, 15], // 5 tokens
            },
        ];

        let device = Device::Cpu;

        // Collate with max_seq_length = 6 (should truncate first example)
        let batch = collate_batch_with_max_len(&examples, &device, Some(6)).unwrap();

        // Check shape: [batch_size=2, seq_len=6]
        assert_eq!(batch.input_ids.dims(), &[2, 6]);

        // Check attention mask shape
        let mask = batch.attention_mask.as_ref().unwrap();
        assert_eq!(mask.dims(), &[2, 6]);
    }

    #[test]
    fn test_collate_batch_shape_validation() {
        use hybrid_predict_trainer_rs::Batch;

        let examples = vec![
            TokenizedExample {
                input_ids: vec![1, 2, 3],
            },
        ];

        let device = Device::Cpu;
        let batch = collate_batch(&examples, &device).unwrap();

        // Shape should be [1, 3]
        assert_eq!(batch.input_ids.dims(), &[1, 3]);

        // Verify batch_size() method from Batch trait
        assert_eq!(batch.batch_size(), 1);
    }

    #[test]
    fn test_data_loader_enforces_max_seq_length() {
        let file = create_test_jsonl();
        // Use a very short max_seq_length to test truncation
        let config = DataConfig::test().with_max_seq_length(10).with_batch_size(2);
        let device = Device::Cpu;

        let dataset = JsonlDataset::new(file.path(), config.max_seq_length).unwrap();
        let loader = DataLoader::new(Box::new(dataset), config, device);

        for batch in loader {
            let batch = batch.unwrap();
            // All batches should have seq_len <= 10
            assert!(batch.input_ids.dims()[1] <= 10);
            // First dimension should be batch_size
            assert!(batch.input_ids.dims()[0] <= 2);
        }
    }

    #[test]
    fn test_minimum_sequence_length() {
        // Test that we get at least 2 tokens (for next-token prediction)
        let examples = vec![
            TokenizedExample {
                input_ids: vec![1], // Only 1 token
            },
        ];

        let device = Device::Cpu;
        let batch = collate_batch_with_max_len(&examples, &device, Some(100)).unwrap();

        // Should be padded to at least 2 tokens
        assert!(batch.input_ids.dims()[1] >= 2);
    }

    fn create_test_jsonl_2() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(file, r#"{{"text": "Dataset two line one"}}"#).unwrap();
        writeln!(file, r#"{{"text": "Dataset two line two"}}"#).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_composite_dataset_sequential() {
        let file1 = create_test_jsonl();
        let file2 = create_test_jsonl_2();

        let ds1 = JsonlDataset::new(file1.path(), 256).unwrap();
        let ds2 = JsonlDataset::new(file2.path(), 256).unwrap();

        let mut composite = CompositeDataset::new(vec![Box::new(ds1), Box::new(ds2)])
            .with_strategy(DatasetBlendingStrategy::Sequential);

        let mut count = 0;
        while let Some(result) = composite.next_example() {
            assert!(result.is_ok());
            count += 1;
        }

        // 4 from file1 + 2 from file2 = 6 total
        assert_eq!(count, 6);
    }

    #[test]
    fn test_composite_dataset_roundrobin() {
        let file1 = create_test_jsonl();
        let file2 = create_test_jsonl_2();

        let ds1 = JsonlDataset::new(file1.path(), 256).unwrap();
        let ds2 = JsonlDataset::new(file2.path(), 256).unwrap();

        let mut composite = CompositeDataset::new(vec![Box::new(ds1), Box::new(ds2)])
            .with_strategy(DatasetBlendingStrategy::RoundRobin);

        // Should interleave examples from both datasets
        let mut count = 0;
        while let Some(result) = composite.next_example() {
            assert!(result.is_ok());
            count += 1;
            if count > 10 {
                break; // Prevent infinite loop in case of bugs
            }
        }

        // Should get examples from both datasets
        assert!(count >= 4); // At least 4 examples
    }

    #[test]
    fn test_composite_dataset_reset() {
        let file1 = create_test_jsonl();
        let file2 = create_test_jsonl_2();

        let ds1 = JsonlDataset::new(file1.path(), 256).unwrap();
        let ds2 = JsonlDataset::new(file2.path(), 256).unwrap();

        let mut composite = CompositeDataset::new(vec![Box::new(ds1), Box::new(ds2)])
            .with_strategy(DatasetBlendingStrategy::Sequential);

        // Exhaust dataset
        while composite.next_example().is_some() {}

        // Reset and count again
        composite.reset().unwrap();

        let mut count = 0;
        while composite.next_example().is_some() {
            count += 1;
        }

        assert_eq!(count, 6);
    }

    #[test]
    fn test_composite_dataset_num_datasets() {
        let file1 = create_test_jsonl();
        let file2 = create_test_jsonl_2();

        let ds1 = JsonlDataset::new(file1.path(), 256).unwrap();
        let ds2 = JsonlDataset::new(file2.path(), 256).unwrap();

        let composite = CompositeDataset::new(vec![Box::new(ds1), Box::new(ds2)]);
        assert_eq!(composite.num_datasets(), 2);
    }
}
