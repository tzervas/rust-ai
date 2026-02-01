//! Multimodal tokenization compatible with Python tritter MultiModalTokenizer.
//!
//! This module provides a Rust implementation of the multimodal tokenizer that
//! supports text, code, image, and audio modalities with a unified embedding space.
//!
//! # Vocabulary Layout
//!
//! ```text
//! 0-7:        Special tokens (PAD, BOS, EOS, UNK, modality prefixes)
//! 8-263:      Reserved for BPE tokens (lower range)
//! 264-999:    Keyword tokens (AST encoding)
//! 1000-1499:  Operator tokens (AST encoding)
//! 1500-1501:  INDENT/DEDENT structural tokens
//! 1502-65279: BPE tokens (upper range)
//! 65280-65535: Byte fallback (256 values)
//! ```
//!
//! # Example
//!
//! ```no_run
//! use tritter_model_rs::tokenizer::{MultiModalTokenizer, ModalityType};
//!
//! let tokenizer = MultiModalTokenizer::new(65536, 131072);
//! let tokens = tokenizer.encode("Hello, world!", ModalityType::Text, true).unwrap();
//! let decoded = tokenizer.decode(&tokens, true).unwrap();
//! ```

use std::collections::HashMap;

use crate::error::{TritterError, TritterResult};

/// Supported modality types for multimodal processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityType {
    /// Text modality (BPE tokenization)
    Text,
    /// Code modality (AST-aware tokenization with BPE fallback)
    Code,
    /// Image modality (VQVAE tokens - placeholder)
    Image,
    /// Audio modality (SpeechTokenizer tokens - placeholder)
    Audio,
}

impl ModalityType {
    /// Get the string representation matching Python enum values.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModalityType::Text => "text",
            ModalityType::Code => "code",
            ModalityType::Image => "image",
            ModalityType::Audio => "audio",
        }
    }
}

impl std::fmt::Display for ModalityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Special token constants matching Python MultiModalTokenizer.
pub mod special_tokens {
    /// Padding token string
    pub const PAD_TOKEN: &str = "<pad>";
    /// Beginning of sequence token string
    pub const BOS_TOKEN: &str = "<bos>";
    /// End of sequence token string
    pub const EOS_TOKEN: &str = "<eos>";
    /// Unknown token string
    pub const UNK_TOKEN: &str = "<unk>";
    /// Text modality prefix
    pub const TEXT_PREFIX: &str = "<text>";
    /// Code modality prefix
    pub const CODE_PREFIX: &str = "<code>";
    /// Image modality prefix
    pub const IMAGE_PREFIX: &str = "<image>";
    /// Audio modality prefix
    pub const AUDIO_PREFIX: &str = "<audio>";

    /// Padding token ID
    pub const PAD_ID: u32 = 0;
    /// Beginning of sequence token ID
    pub const BOS_ID: u32 = 1;
    /// End of sequence token ID
    pub const EOS_ID: u32 = 2;
    /// Unknown token ID
    pub const UNK_ID: u32 = 3;
    /// Text modality prefix ID
    pub const TEXT_PREFIX_ID: u32 = 4;
    /// Code modality prefix ID
    pub const CODE_PREFIX_ID: u32 = 5;
    /// Image modality prefix ID
    pub const IMAGE_PREFIX_ID: u32 = 6;
    /// Audio modality prefix ID
    pub const AUDIO_PREFIX_ID: u32 = 7;
}

/// Vocabulary range constants.
mod vocab_ranges {
    /// Start of BPE offset (after special tokens)
    pub const BPE_OFFSET: u32 = 8;
    /// Start of keyword tokens (AST)
    pub const KEYWORD_START: u32 = 264;
    /// End of keyword tokens (exclusive)
    pub const KEYWORD_END: u32 = 1000;
    /// Start of operator tokens (AST)
    pub const OPERATOR_START: u32 = 1000;
    /// End of operator tokens (exclusive)
    pub const OPERATOR_END: u32 = 1500;
    /// Indent structural token
    pub const INDENT_TOKEN: u32 = 1500;
    /// Dedent structural token
    pub const DEDENT_TOKEN: u32 = 1501;
    /// End of AST token range (start of upper BPE range)
    pub const AST_TOKENS_END: u32 = 1502;
}

/// Unified multimodal tokenizer compatible with Python MultiModalTokenizer.
///
/// Maps all modalities to a shared token space enabling early fusion
/// for cross-modal attention and any-to-any generation.
#[derive(Debug, Clone)]
pub struct MultiModalTokenizer {
    /// Size of unified vocabulary (default 65536 = 2^16)
    vocab_size: u32,
    /// Maximum sequence length (default 131072 = 128K tokens)
    max_length: usize,
    /// HuggingFace tokenizer backend (lazy initialized)
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Special token mappings
    special_tokens: HashMap<String, u32>,
    /// Modality prefix mappings
    modality_prefixes: HashMap<ModalityType, u32>,
    /// Start of byte fallback range
    byte_fallback_start: u32,
}

impl Default for MultiModalTokenizer {
    fn default() -> Self {
        Self::new(65536, 131072)
    }
}

impl MultiModalTokenizer {
    /// Create a new multimodal tokenizer.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of unified vocabulary (default 65536)
    /// * `max_length` - Maximum sequence length (default 131072)
    ///
    /// # Panics
    ///
    /// Panics if vocab_size < 264 (minimum for special tokens + byte fallback).
    pub fn new(vocab_size: u32, max_length: usize) -> Self {
        assert!(
            vocab_size >= 264,
            "vocab_size must be at least 264 (8 special + 256 byte fallback)"
        );

        let mut special_tokens = HashMap::new();
        special_tokens.insert(special_tokens::PAD_TOKEN.to_string(), special_tokens::PAD_ID);
        special_tokens.insert(special_tokens::BOS_TOKEN.to_string(), special_tokens::BOS_ID);
        special_tokens.insert(special_tokens::EOS_TOKEN.to_string(), special_tokens::EOS_ID);
        special_tokens.insert(special_tokens::UNK_TOKEN.to_string(), special_tokens::UNK_ID);
        special_tokens.insert(
            special_tokens::TEXT_PREFIX.to_string(),
            special_tokens::TEXT_PREFIX_ID,
        );
        special_tokens.insert(
            special_tokens::CODE_PREFIX.to_string(),
            special_tokens::CODE_PREFIX_ID,
        );
        special_tokens.insert(
            special_tokens::IMAGE_PREFIX.to_string(),
            special_tokens::IMAGE_PREFIX_ID,
        );
        special_tokens.insert(
            special_tokens::AUDIO_PREFIX.to_string(),
            special_tokens::AUDIO_PREFIX_ID,
        );

        let mut modality_prefixes = HashMap::new();
        modality_prefixes.insert(ModalityType::Text, special_tokens::TEXT_PREFIX_ID);
        modality_prefixes.insert(ModalityType::Code, special_tokens::CODE_PREFIX_ID);
        modality_prefixes.insert(ModalityType::Image, special_tokens::IMAGE_PREFIX_ID);
        modality_prefixes.insert(ModalityType::Audio, special_tokens::AUDIO_PREFIX_ID);

        Self {
            vocab_size,
            max_length,
            tokenizer: None,
            special_tokens,
            modality_prefixes,
            byte_fallback_start: vocab_size - 256,
        }
    }

    /// Create tokenizer with a pre-trained HuggingFace tokenizer file.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of unified vocabulary
    /// * `max_length` - Maximum sequence length
    /// * `tokenizer_path` - Path to HuggingFace tokenizer.json file
    pub fn from_file(
        vocab_size: u32,
        max_length: usize,
        tokenizer_path: &str,
    ) -> TritterResult<Self> {
        let mut tokenizer = Self::new(vocab_size, max_length);
        let hf_tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            TritterError::InvalidConfig(format!("Failed to load tokenizer: {}", e))
        })?;
        tokenizer.tokenizer = Some(hf_tokenizer);
        Ok(tokenizer)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Get the maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get token ID for a special token string.
    pub fn get_special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// Get the modality prefix token ID.
    pub fn get_modality_prefix_id(&self, modality: ModalityType) -> u32 {
        self.modality_prefixes[&modality]
    }

    /// Check if a token ID is a special token.
    pub fn is_special_token(&self, token_id: u32) -> bool {
        token_id < vocab_ranges::BPE_OFFSET
    }

    /// Encode content to token IDs.
    ///
    /// # Arguments
    ///
    /// * `content` - Text/code content to encode
    /// * `modality` - Type of content being encoded
    /// * `add_special_tokens` - Whether to add BOS/EOS and modality prefix
    ///
    /// # Returns
    ///
    /// Vector of token IDs in the unified vocabulary space.
    pub fn encode(
        &self,
        content: &str,
        modality: ModalityType,
        add_special_tokens: bool,
    ) -> TritterResult<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(special_tokens::BOS_ID);
            tokens.push(self.modality_prefixes[&modality]);
        }

        // Modality-specific encoding
        let content_tokens = match modality {
            ModalityType::Text => self.encode_text(content)?,
            ModalityType::Code => self.encode_code(content)?,
            ModalityType::Image => self.encode_image_placeholder(),
            ModalityType::Audio => self.encode_audio_placeholder(),
        };
        tokens.extend(content_tokens);

        if add_special_tokens {
            tokens.push(special_tokens::EOS_ID);
        }

        // Truncate if necessary
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
        }

        Ok(tokens)
    }

    /// Encode text using byte-level fallback (BPE backend optional).
    ///
    /// If a HuggingFace tokenizer is loaded, uses BPE encoding.
    /// Otherwise, falls back to byte-level encoding for simplicity.
    fn encode_text(&self, text: &str) -> TritterResult<Vec<u32>> {
        if let Some(ref tokenizer) = self.tokenizer {
            // Use HuggingFace tokenizer
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|e| TritterError::InvalidConfig(format!("Tokenization failed: {}", e)))?;

            let mut unified_tokens = Vec::with_capacity(encoding.get_ids().len());
            for &tk_id in encoding.get_ids() {
                let unified_id = self.map_bpe_to_unified(tk_id);
                unified_tokens.push(unified_id);
            }
            Ok(unified_tokens)
        } else {
            // Byte-level fallback encoding
            self.encode_bytes(text.as_bytes())
        }
    }

    /// Encode code (falls back to text encoding without AST support).
    fn encode_code(&self, code: &str) -> TritterResult<Vec<u32>> {
        // For now, treat code as text
        // TODO: Add AST-aware tokenization with tree-sitter
        self.encode_text(code)
    }

    /// Placeholder for image encoding (VQVAE tokens).
    fn encode_image_placeholder(&self) -> Vec<u32> {
        // Return 256 image prefix tokens as placeholder
        vec![special_tokens::IMAGE_PREFIX_ID; 256]
    }

    /// Placeholder for audio encoding (SpeechTokenizer tokens).
    fn encode_audio_placeholder(&self) -> Vec<u32> {
        // Return 128 audio prefix tokens as placeholder
        vec![special_tokens::AUDIO_PREFIX_ID; 128]
    }

    /// Encode raw bytes to byte fallback tokens.
    fn encode_bytes(&self, bytes: &[u8]) -> TritterResult<Vec<u32>> {
        Ok(bytes
            .iter()
            .map(|&b| self.byte_fallback_start + u32::from(b))
            .collect())
    }

    /// Map BPE token ID to unified vocabulary space.
    ///
    /// BPE tokens are mapped to avoid collision with AST token ranges:
    /// - Lower range [8, 264): first 256 BPE slots
    /// - Upper range [1502, byte_fallback_start): remaining BPE slots
    fn map_bpe_to_unified(&self, bpe_id: u32) -> u32 {
        let lower_range_size = vocab_ranges::KEYWORD_START - vocab_ranges::BPE_OFFSET; // 256
        let upper_range_size = self.byte_fallback_start - vocab_ranges::AST_TOKENS_END;
        let total_bpe_space = lower_range_size + upper_range_size;

        let bpe_slot = bpe_id % total_bpe_space;

        if bpe_slot < lower_range_size {
            vocab_ranges::BPE_OFFSET + bpe_slot
        } else {
            vocab_ranges::AST_TOKENS_END + (bpe_slot - lower_range_size)
        }
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Token IDs from the unified vocabulary
    /// * `skip_special_tokens` - Whether to filter out special tokens
    ///
    /// # Returns
    ///
    /// Decoded string representation.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> TritterResult<String> {
        let filtered: Vec<u32> = if skip_special_tokens {
            token_ids
                .iter()
                .copied()
                .filter(|&t| !self.is_special_token(t))
                .collect()
        } else {
            token_ids.to_vec()
        };

        if filtered.is_empty() {
            return Ok(String::new());
        }

        // Try HuggingFace tokenizer first
        if let Some(ref tokenizer) = self.tokenizer {
            return self.decode_with_tokenizer(tokenizer, &filtered);
        }

        // Byte-level fallback decoding
        self.decode_bytes(&filtered)
    }

    /// Decode using HuggingFace tokenizer.
    fn decode_with_tokenizer(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        token_ids: &[u32],
    ) -> TritterResult<String> {
        let mut decoded_parts = Vec::new();
        let mut bpe_batch = Vec::new();
        let mut byte_chars = Vec::new();

        for &token_id in token_ids {
            if self.byte_fallback_start <= token_id && token_id < self.vocab_size {
                // Byte fallback token
                if !bpe_batch.is_empty() {
                    // Flush BPE batch
                    let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
                    decoded_parts.push(bpe_decoded);
                    bpe_batch.clear();
                }
                byte_chars.push((token_id - self.byte_fallback_start) as u8);
            } else if self.is_bpe_token(token_id) {
                // BPE token
                if !byte_chars.is_empty() {
                    // Flush byte chars
                    decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
                    byte_chars.clear();
                }
                let tiktoken_slot = self.map_unified_to_bpe(token_id);
                bpe_batch.push(tiktoken_slot);
            } else if vocab_ranges::KEYWORD_START <= token_id && token_id < vocab_ranges::KEYWORD_END
            {
                // Keyword token
                if !bpe_batch.is_empty() {
                    let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
                    decoded_parts.push(bpe_decoded);
                    bpe_batch.clear();
                }
                if !byte_chars.is_empty() {
                    decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
                    byte_chars.clear();
                }
                decoded_parts.push(" <kw> ".to_string());
            } else if vocab_ranges::OPERATOR_START <= token_id
                && token_id < vocab_ranges::OPERATOR_END
            {
                // Operator token
                if !bpe_batch.is_empty() {
                    let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
                    decoded_parts.push(bpe_decoded);
                    bpe_batch.clear();
                }
                if !byte_chars.is_empty() {
                    decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
                    byte_chars.clear();
                }
                decoded_parts.push(" ".to_string());
            } else if token_id == vocab_ranges::INDENT_TOKEN {
                if !bpe_batch.is_empty() {
                    let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
                    decoded_parts.push(bpe_decoded);
                    bpe_batch.clear();
                }
                if !byte_chars.is_empty() {
                    decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
                    byte_chars.clear();
                }
                decoded_parts.push("    ".to_string());
            } else if token_id == vocab_ranges::DEDENT_TOKEN {
                // Dedent doesn't add text
                if !bpe_batch.is_empty() {
                    let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
                    decoded_parts.push(bpe_decoded);
                    bpe_batch.clear();
                }
                if !byte_chars.is_empty() {
                    decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
                    byte_chars.clear();
                }
            }
            // Skip unknown tokens
        }

        // Flush remaining
        if !bpe_batch.is_empty() {
            let bpe_decoded = self.decode_bpe_batch(tokenizer, &bpe_batch)?;
            decoded_parts.push(bpe_decoded);
        }
        if !byte_chars.is_empty() {
            decoded_parts.push(String::from_utf8_lossy(&byte_chars).to_string());
        }

        Ok(decoded_parts.join(""))
    }

    /// Check if token ID is in BPE range.
    fn is_bpe_token(&self, token_id: u32) -> bool {
        (vocab_ranges::BPE_OFFSET <= token_id && token_id < vocab_ranges::KEYWORD_START)
            || (vocab_ranges::AST_TOKENS_END <= token_id && token_id < self.byte_fallback_start)
    }

    /// Map unified vocabulary ID back to BPE slot.
    fn map_unified_to_bpe(&self, unified_id: u32) -> u32 {
        let lower_range_size = vocab_ranges::KEYWORD_START - vocab_ranges::BPE_OFFSET;

        if unified_id < vocab_ranges::KEYWORD_START {
            unified_id - vocab_ranges::BPE_OFFSET
        } else {
            lower_range_size + (unified_id - vocab_ranges::AST_TOKENS_END)
        }
    }

    /// Decode BPE token batch using HuggingFace tokenizer.
    fn decode_bpe_batch(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        bpe_ids: &[u32],
    ) -> TritterResult<String> {
        tokenizer
            .decode(bpe_ids, false)
            .map_err(|e| TritterError::InvalidConfig(format!("BPE decode failed: {}", e)))
    }

    /// Decode using byte-level fallback only.
    fn decode_bytes(&self, token_ids: &[u32]) -> TritterResult<String> {
        let mut bytes = Vec::new();

        for &token_id in token_ids {
            if self.byte_fallback_start <= token_id && token_id < self.vocab_size {
                bytes.push((token_id - self.byte_fallback_start) as u8);
            } else if vocab_ranges::INDENT_TOKEN == token_id {
                bytes.extend_from_slice(b"    ");
            } else if vocab_ranges::KEYWORD_START <= token_id
                && token_id < vocab_ranges::KEYWORD_END
            {
                bytes.extend_from_slice(b" <kw> ");
            } else if vocab_ranges::OPERATOR_START <= token_id
                && token_id < vocab_ranges::OPERATOR_END
            {
                bytes.push(b' ');
            }
            // Skip other tokens in byte-only mode
        }

        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Batch encode multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to encode
    /// * `modality` - Type of content being encoded
    /// * `add_special_tokens` - Whether to add BOS/EOS and modality prefix
    ///
    /// # Returns
    ///
    /// Vector of token ID vectors, one per input text.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        modality: ModalityType,
        add_special_tokens: bool,
    ) -> TritterResult<Vec<Vec<u32>>> {
        texts
            .iter()
            .map(|text| self.encode(text, modality, add_special_tokens))
            .collect()
    }

    /// Batch decode multiple token sequences.
    ///
    /// # Arguments
    ///
    /// * `token_ids_batch` - Slice of token ID vectors to decode
    /// * `skip_special_tokens` - Whether to filter out special tokens
    ///
    /// # Returns
    ///
    /// Vector of decoded strings, one per input sequence.
    pub fn decode_batch(
        &self,
        token_ids_batch: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> TritterResult<Vec<String>> {
        token_ids_batch
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Pad a batch of token sequences to the same length.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Mutable slice of token ID vectors to pad in place
    /// * `max_len` - Target length (if None, uses longest sequence)
    /// * `pad_right` - If true, pad on right; otherwise pad on left
    pub fn pad_batch(&self, sequences: &mut [Vec<u32>], max_len: Option<usize>, pad_right: bool) {
        let target_len = max_len.unwrap_or_else(|| sequences.iter().map(|s| s.len()).max().unwrap_or(0));

        for seq in sequences.iter_mut() {
            if seq.len() < target_len {
                let pad_count = target_len - seq.len();
                let padding = vec![special_tokens::PAD_ID; pad_count];
                if pad_right {
                    seq.extend(padding);
                } else {
                    let mut new_seq = padding;
                    new_seq.append(seq);
                    *seq = new_seq;
                }
            }
        }
    }

    /// Create attention mask for padded sequences.
    ///
    /// Returns 1 for non-padding tokens, 0 for padding tokens.
    pub fn create_attention_mask(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids
            .iter()
            .map(|&id| if id == special_tokens::PAD_ID { 0 } else { 1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        assert_eq!(tokenizer.vocab_size(), 65536);
        assert_eq!(tokenizer.max_length(), 131072);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);

        assert_eq!(tokenizer.get_special_token_id("<pad>"), Some(0));
        assert_eq!(tokenizer.get_special_token_id("<bos>"), Some(1));
        assert_eq!(tokenizer.get_special_token_id("<eos>"), Some(2));
        assert_eq!(tokenizer.get_special_token_id("<unk>"), Some(3));
        assert_eq!(tokenizer.get_special_token_id("<text>"), Some(4));
        assert_eq!(tokenizer.get_special_token_id("<code>"), Some(5));
        assert_eq!(tokenizer.get_special_token_id("<image>"), Some(6));
        assert_eq!(tokenizer.get_special_token_id("<audio>"), Some(7));
    }

    #[test]
    fn test_modality_prefixes() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);

        assert_eq!(tokenizer.get_modality_prefix_id(ModalityType::Text), 4);
        assert_eq!(tokenizer.get_modality_prefix_id(ModalityType::Code), 5);
        assert_eq!(tokenizer.get_modality_prefix_id(ModalityType::Image), 6);
        assert_eq!(tokenizer.get_modality_prefix_id(ModalityType::Audio), 7);
    }

    #[test]
    fn test_encode_text_with_special_tokens() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let tokens = tokenizer.encode("Hello", ModalityType::Text, true).unwrap();

        // Should have BOS, TEXT_PREFIX, content tokens, EOS
        assert!(tokens.len() >= 4);
        assert_eq!(tokens[0], special_tokens::BOS_ID);
        assert_eq!(tokens[1], special_tokens::TEXT_PREFIX_ID);
        assert_eq!(*tokens.last().unwrap(), special_tokens::EOS_ID);
    }

    #[test]
    fn test_encode_decode_roundtrip_bytes() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let original = "Hello, world!";

        let tokens = tokenizer.encode(original, ModalityType::Text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, true).unwrap();

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_code() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let code = "fn main() {}";

        let tokens = tokenizer.encode(code, ModalityType::Code, true).unwrap();

        // Should have BOS, CODE_PREFIX, content, EOS
        assert!(tokens.len() >= 4);
        assert_eq!(tokens[0], special_tokens::BOS_ID);
        assert_eq!(tokens[1], special_tokens::CODE_PREFIX_ID);
    }

    #[test]
    fn test_image_placeholder() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let tokens = tokenizer.encode("", ModalityType::Image, true).unwrap();

        // BOS + IMAGE_PREFIX + 256 placeholder tokens + EOS
        assert_eq!(tokens.len(), 1 + 1 + 256 + 1);

        // Check placeholder tokens
        for &t in &tokens[2..258] {
            assert_eq!(t, special_tokens::IMAGE_PREFIX_ID);
        }
    }

    #[test]
    fn test_audio_placeholder() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let tokens = tokenizer.encode("", ModalityType::Audio, true).unwrap();

        // BOS + AUDIO_PREFIX + 128 placeholder tokens + EOS
        assert_eq!(tokens.len(), 1 + 1 + 128 + 1);
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let texts = &["Hello", "World", "Test"];

        let batch = tokenizer.encode_batch(texts, ModalityType::Text, true).unwrap();

        assert_eq!(batch.len(), 3);
        for tokens in &batch {
            assert_eq!(tokens[0], special_tokens::BOS_ID);
        }
    }

    #[test]
    fn test_padding() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let mut sequences = vec![
            vec![1, 2, 3],
            vec![1, 2],
            vec![1],
        ];

        tokenizer.pad_batch(&mut sequences, Some(5), true);

        assert_eq!(sequences[0].len(), 5);
        assert_eq!(sequences[1].len(), 5);
        assert_eq!(sequences[2].len(), 5);

        // Check padding is on the right
        assert_eq!(sequences[0], vec![1, 2, 3, 0, 0]);
        assert_eq!(sequences[1], vec![1, 2, 0, 0, 0]);
        assert_eq!(sequences[2], vec![1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_attention_mask() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let tokens = vec![1, 2, 3, 0, 0];  // Last two are padding

        let mask = tokenizer.create_attention_mask(&tokens);

        assert_eq!(mask, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_truncation() {
        let tokenizer = MultiModalTokenizer::new(65536, 20);  // Small max_length
        let long_text = "This is a very long text that should be truncated";

        let tokens = tokenizer.encode(long_text, ModalityType::Text, true).unwrap();

        assert!(tokens.len() <= 20);
    }

    #[test]
    fn test_is_special_token() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);

        assert!(tokenizer.is_special_token(0));
        assert!(tokenizer.is_special_token(7));
        assert!(!tokenizer.is_special_token(8));
        assert!(!tokenizer.is_special_token(1000));
    }

    #[test]
    fn test_modality_type_display() {
        assert_eq!(ModalityType::Text.as_str(), "text");
        assert_eq!(ModalityType::Code.as_str(), "code");
        assert_eq!(ModalityType::Image.as_str(), "image");
        assert_eq!(ModalityType::Audio.as_str(), "audio");
    }

    #[test]
    fn test_unicode_roundtrip() {
        let tokenizer = MultiModalTokenizer::new(65536, 131072);
        let unicode_text = "Hello, \u{4e16}\u{754c}! \u{1f600}";  // "Hello, 世界! 😀"

        let tokens = tokenizer.encode(unicode_text, ModalityType::Text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, true).unwrap();

        assert_eq!(decoded, unicode_text);
    }
}
