# Learning Rate Advisor - Integration Guide

The Learning Rate Advisor analyzes training dynamics in real-time to suggest learning rate adjustments based on loss curve and gradient behavior.

## Quick Start

```rust
use training_tools::lr_advisor::{analyze_lr, LRAdvice, TrainingPhase, Urgency};

// In your training loop
let mut loss_history: Vec<f32> = Vec::new();
let mut grad_history: Vec<f32> = Vec::new();
let mut current_lr = 1e-4;

for step in 0..max_steps {
    // ... perform training step ...

    // Track metrics (keep last 50-100 steps)
    loss_history.push(loss);
    grad_history.push(grad_norm);
    if loss_history.len() > 100 {
        loss_history.remove(0);
        grad_history.remove(0);
    }

    // Check for LR advice every N steps
    if step % 10 == 0 && loss_history.len() >= 10 {
        if let Some(advice) = analyze_lr(
            &loss_history,
            &grad_history,
            current_lr,
            step,
            TrainingPhase::Stable,
        ) {
            handle_lr_advice(advice, &mut current_lr, &mut optimizer);
        }
    }
}
```

## Integration Patterns

### Pattern 1: Automatic Application (Critical/High Only)

Apply critical and high-priority suggestions immediately:

```rust
fn handle_lr_advice(advice: LRAdvice, current_lr: &mut f32, optimizer: &mut Optimizer) {
    match advice.urgency {
        Urgency::Critical => {
            println!("🚨 CRITICAL: {}", advice.format());
            *current_lr = advice.suggested_lr;
            optimizer.set_lr(*current_lr);
        }
        Urgency::High => {
            println!("⚠️  HIGH: {}", advice.format());
            *current_lr = advice.suggested_lr;
            optimizer.set_lr(*current_lr);
        }
        Urgency::Medium | Urgency::Low => {
            println!("ℹ️  INFO: {}", advice.format());
            // Log but don't auto-apply
        }
    }
}
```

### Pattern 2: Logging Only (Manual Review)

Log all suggestions for post-training analysis:

```rust
fn log_lr_advice(advice: &LRAdvice, step: u64, log_file: &mut File) {
    let entry = serde_json::json!({
        "step": step,
        "current_lr": advice.current_lr,
        "suggested_lr": advice.suggested_lr,
        "change_pct": advice.percentage_change(),
        "urgency": advice.urgency,
        "issue": advice.issue,
        "reason": advice.reason,
    });
    writeln!(log_file, "{}", entry).ok();
}
```

### Pattern 3: Hybrid (Critical Auto, Others Manual)

Automatically apply critical changes, log others for manual review:

```rust
fn handle_lr_advice_hybrid(
    advice: LRAdvice,
    current_lr: &mut f32,
    optimizer: &mut Optimizer,
    log: &mut File,
) {
    // Always log
    log_lr_advice(&advice, step, log);

    // Auto-apply only critical issues
    if advice.urgency == Urgency::Critical {
        println!("🚨 AUTO-APPLYING: {}", advice.format());
        *current_lr = advice.suggested_lr;
        optimizer.set_lr(*current_lr);
    } else {
        println!("📊 Logged suggestion: {}", advice.format());
    }
}
```

## Training Phase Considerations

The advisor adjusts tolerance based on training phase:

```rust
let phase = if step < warmup_steps {
    TrainingPhase::Warmup  // Higher tolerance for oscillation
} else if step < stable_end {
    TrainingPhase::Stable  // Standard analysis
} else {
    TrainingPhase::Correct // Standard analysis
};

if let Some(advice) = analyze_lr(&losses, &grads, lr, step, phase) {
    // Warmup suggestions will have lower urgency for same issues
}
```

## Issue-Specific Handling

Different issues may warrant different responses:

```rust
use training_tools::lr_advisor::Issue;

fn handle_by_issue(advice: LRAdvice) {
    match advice.issue {
        Issue::GradientExplosion => {
            // Always apply - training will diverge otherwise
            apply_lr_change(advice.suggested_lr);
            // Maybe also clip gradients
            enable_gradient_clipping();
        }
        Issue::LossOscillation => {
            // Apply if urgency is high
            if advice.urgency >= Urgency::High {
                apply_lr_change(advice.suggested_lr);
            }
        }
        Issue::LossPlateau => {
            // Consider applying, but also check for other issues
            // (could be architecture problem, not just LR)
            log_for_review(advice);
        }
        Issue::GradientVanishing => {
            // Might be architectural - increase LR cautiously
            let new_lr = advice.current_lr * 1.2; // More conservative than suggestion
            apply_lr_change(new_lr);
        }
        _ => {
            // Other issues - log for review
            log_for_review(advice);
        }
    }
}
```

## Metrics Collection Best Practices

```rust
// Efficient circular buffer for metrics
struct MetricsBuffer {
    losses: Vec<f32>,
    gradients: Vec<f32>,
    capacity: usize,
}

impl MetricsBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            losses: Vec::with_capacity(capacity),
            gradients: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, loss: f32, grad_norm: f32) {
        if self.losses.len() >= self.capacity {
            self.losses.remove(0);
            self.gradients.remove(0);
        }
        self.losses.push(loss);
        self.gradients.push(grad_norm);
    }

    fn check_lr(&self, current_lr: f32, step: u64, phase: TrainingPhase) -> Option<LRAdvice> {
        if self.losses.len() < 10 {
            return None;
        }
        analyze_lr(&self.losses, &self.gradients, current_lr, step, phase)
    }
}
```

## Analysis Criteria Reference

| Issue | Threshold | Suggested Action | Default Urgency |
|-------|-----------|------------------|-----------------|
| Gradient Explosion | >10x baseline | Reduce LR by 75% | Critical |
| Gradient Growth | >5x baseline | Reduce LR by 50% | High |
| Loss Oscillation | Amplitude >10% of mean | Reduce LR by 50-60% | High |
| Loss Increase | >5% increase | Reduce LR by 40-60% | High |
| Gradient Instability | CV > 0.5 | Reduce LR by 30% | Medium |
| Gradient Vanishing | <0.01x baseline | Increase LR by 100% | Medium |
| Loss Plateau | Slope < 0.001 for 50+ steps | Increase LR by 20-50% | Medium/Low |

**Note:** Warmup phase has relaxed thresholds (20% oscillation tolerance vs 10% in stable phase).

## Example: Full Training Loop Integration

```rust
use training_tools::lr_advisor::{analyze_lr, TrainingPhase, Urgency};

fn train_with_lr_advisor() -> Result<()> {
    let mut model = create_model()?;
    let mut optimizer = create_optimizer(1e-4)?;
    let mut metrics = MetricsBuffer::new(100);

    let warmup_steps = 1000;
    let max_steps = 10000;

    for step in 0..max_steps {
        // Training step
        let (loss, grad_norm) = train_step(&mut model, &mut optimizer)?;

        // Track metrics
        metrics.push(loss, grad_norm);

        // Determine phase
        let phase = if step < warmup_steps {
            TrainingPhase::Warmup
        } else {
            TrainingPhase::Stable
        };

        // Check LR every 10 steps after warmup
        if step > warmup_steps && step % 10 == 0 {
            if let Some(advice) = metrics.check_lr(optimizer.lr(), step, phase) {
                // Auto-apply critical/high, log others
                match advice.urgency {
                    Urgency::Critical | Urgency::High => {
                        println!("Step {}: Applying LR change: {:.2e} → {:.2e}",
                            step, advice.current_lr, advice.suggested_lr);
                        println!("  Reason: {}", advice.reason);
                        optimizer.set_lr(advice.suggested_lr);
                    }
                    _ => {
                        println!("Step {}: LR suggestion: {}", step, advice.format());
                    }
                }
            }
        }

        // Regular logging
        if step % 100 == 0 {
            println!("Step {}: loss={:.4}, grad_norm={:.4}, lr={:.2e}",
                step, loss, grad_norm, optimizer.lr());
        }
    }

    Ok(())
}
```

## Testing Your Integration

Run the demo to see the advisor in action:

```bash
cargo run --example lr_advisor_demo
```

This shows 6 scenarios:
1. Healthy training (no advice)
2. Loss oscillation detection
3. Loss plateau detection
4. Gradient explosion (critical)
5. Warmup phase tolerance
6. Simulated training loop with automatic adjustments

## Performance Impact

The advisor is lightweight:
- **Analysis time**: <0.1ms for 50 samples
- **Memory**: ~400 bytes per metric sample (50 samples = 20KB)
- **CPU**: Negligible compared to training step time

Recommended check frequency: every 10-50 steps (not every step).

## Troubleshooting

### "Getting too many plateau suggestions"

Increase the slope threshold or window size:
```rust
// Modify in lr_advisor.rs if needed, or
// Filter low-urgency plateau suggestions in your handler
if advice.issue == Issue::LossPlateau && advice.urgency == Urgency::Low {
    // Ignore or just log
}
```

### "Oscillation detected during warmup"

This is expected - warmup phase has 20% tolerance vs 10% in stable phase. If still too sensitive, the advice will have `Urgency::Medium` instead of `High` during warmup.

### "Not detecting issues early enough"

Reduce minimum sample requirement (currently 10) or check more frequently. Note: too few samples reduce reliability.

## Advanced: Custom Thresholds

If you need custom detection thresholds, you can copy the analysis functions from `lr_advisor.rs` and modify them. The module is designed to be extensible.

## See Also

- `/home/kang/Documents/projects/rust-ai/training-tools/src/lr_advisor.rs` - Source code with detailed comments
- `/home/kang/Documents/projects/rust-ai/training-tools/examples/lr_advisor_demo.rs` - Working examples
- `LRScheduler` trait and `WSDScheduler` - Complementary scheduled LR changes
