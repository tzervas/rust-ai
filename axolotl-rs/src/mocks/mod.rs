//! Mock implementations for testing without external dependencies

#[cfg(feature = "mock-peft")]
pub mod peft;

#[cfg(feature = "mock-qlora")]
pub mod qlora;

#[cfg(feature = "mock-unsloth")]
pub mod unsloth;
