# Axolotl-RS Roadmap

## Success Criteria
100% functional parity with Python Axolotl for core fine-tuning (LoRA/QLoRA on LLaMA/Mistral models), validated via end-to-end training tests on Alpaca dataset.

## Requirements
- Integrate peft-rs for adapters, qlora-rs for quantization, unsloth-rs for optimized kernels.
- Add Candle-based model loading and training loop.

## Deliverables
- Complete trainer module with forward/backward passes.
- Checkpoint/save/load functionality.
- Adapter merging.
- Multi-GPU support via Candle.

## Remaining Tasks
- **Phase 2**: Implement model loading and adapter management (LoRA/QLoRA).
- **Phase 3**: Add training loop and checkpoints.
- **Phase 4**: Multi-GPU and advanced features (DPO, GRPO).
- **Estimated Timeline**: 3-6 months for MVP training.