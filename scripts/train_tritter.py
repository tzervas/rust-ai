#!/usr/bin/env python3
"""
Tritter Model Training Automation Script

Implements proper hybrid predictive training with:
- Warmup → Full → Predict → Correct phase cycling
- Data sequencing and curriculum learning
- Mixed dataset batching with proper ratios
- Learning rate scheduling (WSD: Warmup-Stable-Decay)
- Training efficiency monitoring

This script orchestrates the training process to maximize efficiency
while maintaining model quality.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict
import random
import math

# Training Configuration
@dataclass
class TrainingConfig:
    """Configuration for hybrid predictive training."""

    # Model
    model_size: str = "100m"  # "100m", "500m", "1b"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # Training
    total_steps: int = 100_000
    batch_size: int = 32
    gradient_accumulation: int = 4
    max_seq_length: int = 2048

    # Learning Rate (WSD Schedule)
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.03    # 3% of steps for LR warmup
    stable_ratio: float = 0.85    # 85% stable
    decay_ratio: float = 0.12     # 12% decay

    # Hybrid Predictive Training Phases
    phase_warmup_steps: int = 1000      # Initial warmup (no prediction)
    phase_full_steps: int = 20          # Full training between predict phases
    phase_max_predict_steps: int = 80   # Max prediction steps
    phase_correct_steps: int = 5        # Correction validation steps
    confidence_threshold: float = 0.85  # Min confidence for prediction

    # Data Mixing Ratios
    mix_ratios: Dict[str, float] = field(default_factory=lambda: {
        # Natural language (35%)
        "fineweb-edu": 0.22,
        "wikipedia": 0.13,
        # Code (20%)
        "code-python": 0.08,
        "code-rust": 0.04,
        "code-typescript": 0.04,
        "code-go": 0.02,
        "code-other": 0.02,
        # Math & Reasoning (15%)
        "finemath": 0.08,
        "openwebmath": 0.05,
        "tinystories": 0.02,
        # Ethics & Alignment (10%)
        "anthropic-hh": 0.03,
        "ethics": 0.02,
        "truthfulqa": 0.02,
        "saferlhf": 0.02,
        "prosocial": 0.01,
        # IaC & Technical (10%)
        "terraform": 0.03,
        "kubernetes": 0.03,
        "ansible": 0.02,
        "docker": 0.01,
        "github-actions": 0.01,
        # Instruction (10%)
        "alpaca": 0.03,
        "dolly": 0.02,
        "selfinstruct": 0.02,
        "wizardlm": 0.02,
        "openorca": 0.01,
    })

    # Curriculum Learning Phases
    curriculum_phases: List[Dict[str, float]] = field(default_factory=lambda: [
        # Phase 1: Foundation (0-20%)
        # Focus on high-quality web text and basic code
        {
            "start_ratio": 0.0,
            "end_ratio": 0.2,
            "focus": ["fineweb-edu", "wikipedia", "tinystories"],
            "boost": 1.5,  # 50% more weight to these
        },
        # Phase 2: Skill Building (20-50%)
        # Introduce code, math, and reasoning
        {
            "start_ratio": 0.2,
            "end_ratio": 0.5,
            "focus": ["code-python", "code-rust", "finemath", "openwebmath"],
            "boost": 1.3,
        },
        # Phase 3: Specialization (50-80%)
        # Heavy on IaC, instructions, alignment
        {
            "start_ratio": 0.5,
            "end_ratio": 0.8,
            "focus": ["terraform", "kubernetes", "alpaca", "anthropic-hh", "ethics"],
            "boost": 1.4,
        },
        # Phase 4: Refinement (80-100%)
        # Balanced mix with emphasis on quality
        {
            "start_ratio": 0.8,
            "end_ratio": 1.0,
            "focus": ["anthropic-hh", "truthfulqa", "wizardlm"],
            "boost": 1.2,
        },
    ])

    # Efficiency Targets
    target_predict_ratio: float = 0.6   # Target 60% predicted steps
    min_predict_ratio: float = 0.4      # Minimum 40% for speedup
    max_loss_degradation: float = 0.02  # Max 2% loss quality loss

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    checkpoint_dir: str = "/data/models/tritter"

    # Logging
    log_steps: int = 100
    wandb_project: str = "tritter-training"
    wandb_run_name: str = ""

    def get_model_config(self) -> Dict:
        """Get model size configuration."""
        configs = {
            "100m": {"hidden_size": 768, "num_layers": 12, "num_heads": 12},
            "500m": {"hidden_size": 1536, "num_layers": 24, "num_heads": 16},
            "1b": {"hidden_size": 2048, "num_layers": 32, "num_heads": 24},
        }
        return configs.get(self.model_size, configs["100m"])


@dataclass
class TrainingState:
    """Current training state."""
    step: int = 0
    epoch: int = 0
    total_tokens: int = 0

    # Phase tracking
    current_phase: str = "warmup"
    phase_step: int = 0
    warmup_complete: bool = False

    # Loss tracking
    loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    loss_ema: float = 0.0

    # Prediction tracking
    predicted_steps: int = 0
    full_steps: int = 0
    prediction_errors: List[float] = field(default_factory=list)
    predictor_confidence: float = 0.0

    # Efficiency metrics
    tokens_per_second: float = 0.0
    time_elapsed: float = 0.0
    compute_saved: float = 0.0  # Estimated compute saved via prediction


class WSDLearningRateScheduler:
    """Warmup-Stable-Decay learning rate scheduler."""

    def __init__(self, config: TrainingConfig):
        self.base_lr = config.learning_rate
        self.total_steps = config.total_steps

        # Calculate phase boundaries
        self.warmup_steps = int(config.total_steps * config.warmup_ratio)
        self.stable_end = int(config.total_steps * (config.warmup_ratio + config.stable_ratio))
        self.decay_steps = config.total_steps - self.stable_end

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        elif step < self.stable_end:
            # Stable phase
            return self.base_lr
        else:
            # Cosine decay
            decay_progress = (step - self.stable_end) / max(1, self.decay_steps)
            return self.base_lr * (1 + math.cos(math.pi * decay_progress)) / 2


class DataMixer:
    """Mixes data from multiple sources according to specified ratios."""

    def __init__(self, config: TrainingConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.mix_ratios = config.mix_ratios.copy()
        self.curriculum_phases = config.curriculum_phases

        # Dataset iterators
        self.iterators: Dict[str, Iterator] = {}
        self.dataset_sizes: Dict[str, int] = {}

    def get_curriculum_adjusted_ratios(self, progress: float) -> Dict[str, float]:
        """Adjust mix ratios based on curriculum learning phase."""
        ratios = self.mix_ratios.copy()

        # Find current curriculum phase
        current_phase = None
        for phase in self.curriculum_phases:
            if phase["start_ratio"] <= progress < phase["end_ratio"]:
                current_phase = phase
                break

        if current_phase:
            # Boost focus datasets
            focus_datasets = current_phase.get("focus", [])
            boost = current_phase.get("boost", 1.0)

            for dataset in focus_datasets:
                if dataset in ratios:
                    ratios[dataset] *= boost

            # Renormalize
            total = sum(ratios.values())
            ratios = {k: v / total for k, v in ratios.items()}

        return ratios

    def sample_batch(self, batch_size: int, progress: float) -> List[Dict]:
        """Sample a batch with curriculum-adjusted mixing."""
        ratios = self.get_curriculum_adjusted_ratios(progress)

        # Determine samples per dataset
        samples_per_dataset = {}
        remaining = batch_size

        for dataset, ratio in sorted(ratios.items(), key=lambda x: -x[1]):
            count = max(1, int(batch_size * ratio))
            count = min(count, remaining)
            samples_per_dataset[dataset] = count
            remaining -= count
            if remaining <= 0:
                break

        # Collect samples
        batch = []
        for dataset, count in samples_per_dataset.items():
            samples = self._get_samples(dataset, count)
            batch.extend(samples)

        # Shuffle
        random.shuffle(batch)
        return batch[:batch_size]

    def _get_samples(self, dataset: str, count: int) -> List[Dict]:
        """Get samples from a specific dataset."""
        # Placeholder - actual implementation would read from files
        return [{"text": f"[{dataset}] Sample {i}", "source": dataset} for i in range(count)]


class PhaseController:
    """Controls phase transitions in hybrid predictive training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_phase = "warmup"
        self.phase_step = 0
        self.consecutive_predict_phases = 0
        self.predictor_confidence = 0.0
        self.prediction_errors = []

    def select_next_phase(self, state: TrainingState) -> Tuple[str, int]:
        """Select the next training phase and its duration."""

        # Warmup phase
        if not state.warmup_complete:
            if state.step < self.config.phase_warmup_steps:
                return "warmup", self.config.phase_warmup_steps - state.step
            state.warmup_complete = True
            self.current_phase = "full"

        # Phase cycling: Full → Predict → Correct → Full
        if self.current_phase == "warmup":
            self.current_phase = "full"
            return "full", self.config.phase_full_steps

        elif self.current_phase in ("full", "correct"):
            # Decide whether to predict or continue full training
            if (self.predictor_confidence >= self.config.confidence_threshold
                    and self.consecutive_predict_phases < 5):
                self.current_phase = "predict"
                self.consecutive_predict_phases += 1

                # Adaptive prediction length based on confidence
                base_steps = self.config.phase_max_predict_steps
                confidence_factor = self.predictor_confidence ** 2
                adaptive_steps = int(base_steps * confidence_factor)
                return "predict", max(10, adaptive_steps)
            else:
                self.current_phase = "full"
                self.consecutive_predict_phases = 0
                return "full", self.config.phase_full_steps

        elif self.current_phase == "predict":
            # After predict, go to correct
            self.current_phase = "correct"
            return "correct", self.config.phase_correct_steps

        return "full", self.config.phase_full_steps

    def update_confidence(self, confidence: float):
        """Update predictor confidence."""
        self.predictor_confidence = confidence

    def record_prediction_error(self, error: float):
        """Record a prediction error."""
        self.prediction_errors.append(error)
        # Keep last 100 errors
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

        # Update confidence based on recent errors
        if self.prediction_errors:
            avg_error = sum(self.prediction_errors) / len(self.prediction_errors)
            # Lower confidence if errors are high
            self.predictor_confidence = max(0.0, 1.0 - avg_error * 10)


class EfficiencyMonitor:
    """Monitors training efficiency and prediction ratio."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.full_steps = 0
        self.predicted_steps = 0
        self.total_compute_time = 0.0
        self.saved_compute_time = 0.0

    def record_step(self, phase: str, step_time: float, predicted_time: float = None):
        """Record a training step."""
        if phase in ("warmup", "full"):
            self.full_steps += 1
            self.total_compute_time += step_time
        elif phase == "predict":
            self.predicted_steps += 1
            self.total_compute_time += step_time
            if predicted_time:
                self.saved_compute_time += predicted_time - step_time

    @property
    def predict_ratio(self) -> float:
        """Get current prediction ratio."""
        total = self.full_steps + self.predicted_steps
        if total == 0:
            return 0.0
        return self.predicted_steps / total

    @property
    def speedup_factor(self) -> float:
        """Get current speedup factor."""
        if self.total_compute_time == 0:
            return 1.0
        theoretical_time = self.total_compute_time + self.saved_compute_time
        return theoretical_time / self.total_compute_time

    def meets_targets(self) -> bool:
        """Check if efficiency targets are being met."""
        return self.predict_ratio >= self.config.min_predict_ratio

    def get_report(self) -> Dict:
        """Get efficiency report."""
        return {
            "full_steps": self.full_steps,
            "predicted_steps": self.predicted_steps,
            "predict_ratio": self.predict_ratio,
            "speedup_factor": self.speedup_factor,
            "saved_compute_time": self.saved_compute_time,
            "meets_targets": self.meets_targets(),
        }


def execute_phase_step(
    phase: str,
    batch: List[Dict],
    state: TrainingState,
    config: TrainingConfig,
) -> Tuple[float, bool, Optional[float]]:
    """
    Execute a single training step in the given phase.

    Returns: (loss, was_predicted, prediction_error)
    """
    # This is a placeholder - actual implementation would call into
    # the Rust hybrid-predict-trainer-rs via PyO3 bindings

    if phase in ("warmup", "full"):
        # Full forward + backward pass
        loss = 0.5 + random.random() * 0.1  # Simulated loss
        return loss, False, None

    elif phase == "predict":
        # Predictive step - skip backward pass
        predicted_loss = state.loss + random.gauss(0, 0.01)
        actual_loss = 0.5 + random.random() * 0.1
        prediction_error = abs(actual_loss - predicted_loss)
        return actual_loss, True, prediction_error

    elif phase == "correct":
        # Correction step
        loss = state.loss - random.random() * 0.01  # Slight improvement
        return max(0.1, loss), False, None

    return 0.5, False, None


def train(config: TrainingConfig):
    """Main training loop with hybrid predictive training."""

    print("=" * 60)
    print("Tritter Model Training - Hybrid Predictive Mode")
    print("=" * 60)
    print(f"Model size: {config.model_size}")
    print(f"Total steps: {config.total_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Phase config: warmup={config.phase_warmup_steps}, full={config.phase_full_steps}, predict={config.phase_max_predict_steps}")
    print("=" * 60)

    # Initialize components
    lr_scheduler = WSDLearningRateScheduler(config)
    phase_controller = PhaseController(config)
    efficiency_monitor = EfficiencyMonitor(config)
    data_mixer = DataMixer(config, Path("/data/datasets/tritter"))

    state = TrainingState()
    start_time = time.time()

    # Training loop
    for step in range(config.total_steps):
        state.step = step
        progress = step / config.total_steps

        # Get current phase
        phase, phase_duration = phase_controller.select_next_phase(state)
        state.current_phase = phase

        # Get learning rate
        lr = lr_scheduler.get_lr(step)

        # Get batch with curriculum-adjusted mixing
        batch = data_mixer.sample_batch(config.batch_size, progress)

        # Execute step
        step_start = time.time()
        loss, was_predicted, prediction_error = execute_phase_step(
            phase, batch, state, config
        )
        step_time = time.time() - step_start

        # Update state
        state.loss = loss
        state.loss_history.append(loss)
        if len(state.loss_history) > 100:
            state.loss_history.pop(0)

        # Update loss EMA
        alpha = 0.1
        state.loss_ema = alpha * loss + (1 - alpha) * state.loss_ema if state.loss_ema else loss

        # Track prediction metrics
        if was_predicted:
            state.predicted_steps += 1
            if prediction_error is not None:
                state.prediction_errors.append(prediction_error)
                phase_controller.record_prediction_error(prediction_error)
        else:
            state.full_steps += 1

        # Record efficiency
        efficiency_monitor.record_step(phase, step_time)

        # Update tokens processed
        state.total_tokens += config.batch_size * config.max_seq_length
        state.time_elapsed = time.time() - start_time
        state.tokens_per_second = state.total_tokens / state.time_elapsed if state.time_elapsed > 0 else 0

        # Logging
        if step % config.log_steps == 0:
            efficiency = efficiency_monitor.get_report()
            print(f"Step {step:6d} | Phase: {phase:8s} | Loss: {loss:.4f} (EMA: {state.loss_ema:.4f}) | "
                  f"LR: {lr:.2e} | Predict: {efficiency['predict_ratio']*100:.1f}% | "
                  f"Speed: {efficiency['speedup_factor']:.2f}x")

        # Checkpointing
        if step > 0 and step % config.save_steps == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint-{step}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            # Save checkpoint (placeholder)
            with open(checkpoint_path / "state.json", "w") as f:
                json.dump(asdict(state), f, indent=2, default=str)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    efficiency = efficiency_monitor.get_report()
    print(f"Total steps: {state.step}")
    print(f"Total tokens: {state.total_tokens:,}")
    print(f"Final loss: {state.loss:.4f}")
    print(f"Final loss EMA: {state.loss_ema:.4f}")
    print(f"Full steps: {efficiency['full_steps']}")
    print(f"Predicted steps: {efficiency['predicted_steps']}")
    print(f"Prediction ratio: {efficiency['predict_ratio']*100:.1f}%")
    print(f"Speedup factor: {efficiency['speedup_factor']:.2f}x")
    print(f"Total time: {state.time_elapsed:.1f}s")
    print(f"Tokens/second: {state.tokens_per_second:.0f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Tritter model with hybrid predictive training")
    parser.add_argument("--model-size", choices=["100m", "500m", "1b"], default="100m")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="/data/models/tritter")
    parser.add_argument("--predict-ratio-target", type=float, default=0.6)
    args = parser.parse_args()

    config = TrainingConfig(
        model_size=args.model_size,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        phase_warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        target_predict_ratio=args.predict_ratio_target,
    )

    train(config)


if __name__ == "__main__":
    main()
