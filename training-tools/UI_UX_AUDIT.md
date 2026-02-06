# Live Monitor UI/UX Audit Report

**Date**: 2026-02-01
**File**: training-tools/src/live_monitor.rs
**Lines**: 3,950

---

## Executive Summary

The live monitor TUI is a comprehensive, feature-rich interface with excellent foundational design. This audit identified 18 issues across 7 categories, ranging from critical navigation inconsistencies to minor spacing improvements. Overall score: **7.5/10** - solid implementation with room for polish.

---

## 1. Color Consistency: 8/10

### Strengths
- Well-defined color palette in `colors` module with semantic meaning
- Consistent use of green/yellow/red for status (good/warning/bad)
- Memory thresholds: >90% red, >70% yellow, <70% green (CONSISTENT)
- Temperature thresholds: >80°C red, >70°C yellow, <70°C green (CONSISTENT)
- Phase colors are distinct and meaningful

### Issues Found

#### ISSUE 1.1: CurveQuality Color Inconsistency (MEDIUM)
**Location**: Lines 255-264
**Problem**: `CurveQuality::Plateau` uses `Color::Gray` instead of semantic color
```rust
CurveQuality::Plateau => Color::Gray,  // Should be colors::MEMORY_WARN
```
**Impact**: Plateaus are warnings that need intervention, but gray suggests neutral/inactive
**Fix**: Change to `colors::MEMORY_WARN` (yellow)

#### ISSUE 1.2: Tab Number Mismatch in Help (MINOR)
**Location**: Lines 3615, 3747-3749
**Problem**: Help text says "1-5" but tabs are actually 0-8
```
Line 3615: "1-5" should be "0-8"
Line 3749: "1=Overview...8=Help" should be "0=Dashboard, 1=Overview...8=Help"
```
**Impact**: Confusing for new users trying to navigate
**Fix**: Update help text to match actual keybindings

#### ISSUE 1.3: Inconsistent Border Colors
**Location**: Throughout
**Problem**: Most borders use `colors::BORDER` but some use phase colors (e.g., line 1682)
**Impact**: Visual inconsistency - phase-colored borders can be distracting
**Recommendation**: Consider standardizing unless phase color is intentional highlight

---

## 2. Layout Balance: 7/10

### Strengths
- Three-tier vertical structure (header/content/footer) is clean
- Dashboard uses balanced 25/50/25 column split
- Consistent use of Layout constraints

### Issues Found

#### ISSUE 2.1: Dashboard Column Proportions (MEDIUM)
**Location**: Lines 1139-1143
**Problem**: Center column (50%) dominates; left/right (25% each) feel cramped at smaller terminal sizes
```rust
Constraint::Percentage(25),  // Left: Run list + Training Health
Constraint::Percentage(50),  // Center: Charts
Constraint::Percentage(25),  // Right: GPU + Metrics
```
**Impact**: At 80-column terminals, 25% = 20 cols (barely enough for text)
**Recommendation**: Consider 30/40/30 or add minimum constraints:
```rust
Constraint::Min(25).max(Percentage(30))
```

#### ISSUE 2.2: Left Column Vertical Split Imbalance (MINOR)
**Location**: Lines 1147-1150
**Problem**: Run list (40%) vs Training Health (60%) feels inverted
```rust
Constraint::Percentage(40),  // Run list - should be larger
Constraint::Percentage(60),  // Training Health - too much space
```
**Impact**: Run list often truncated when many runs exist; health has excess whitespace
**Fix**: Swap to 60/40 or make dynamic based on run count

#### ISSUE 2.3: Center Column Chart Heights (MINOR)
**Location**: Lines 1158-1162
```rust
Constraint::Percentage(40),  // Loss chart
Constraint::Percentage(30),  // Gradient chart
Constraint::Percentage(30),  // Key metrics bar
```
**Problem**: Metrics bar gets same height as gradient chart despite being text-only
**Recommendation**: 50/30/20 gives more space to main loss chart

#### ISSUE 2.4: Help Overlay Size Hardcoded (MINOR)
**Location**: Lines 3722-3723
```rust
let popup_width = 60.min(area.width - 4);
let popup_height = 28.min(area.height - 4);
```
**Problem**: At 24-line terminals, overlay covers entire screen
**Recommendation**: Use percentage-based sizing:
```rust
let popup_width = (area.width * 70 / 100).min(60).max(40);
let popup_height = (area.height * 80 / 100).min(28).max(20);
```

---

## 3. Information Hierarchy: 8/10

### Strengths
- Most important info (loss, step, status) prominently displayed
- Bold styling used effectively for emphasis
- Titles are clear and descriptive

### Issues Found

#### ISSUE 3.1: Footer Shortcut Hint Hierarchy (LOW)
**Location**: Lines 3818-3832
**Problem**: All shortcuts shown with equal weight; no visual hierarchy
```rust
Span::styled("Tab", Style::default().fg(colors::HELP_KEY)),
Span::raw(":tabs  "),
Span::styled("j/k", Style::default().fg(colors::HELP_KEY)),
```
**Impact**: Users don't know which shortcuts are most important
**Recommendation**: Bold the most critical ones (q, ?, Tab):
```rust
Span::styled("q", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD))
```

#### ISSUE 3.2: Dashboard Title Missing Visual Weight (MINOR)
**Location**: Line 1222
**Problem**: "TRAINING HEALTH" title uses only white + bold, not standing out enough
**Recommendation**: Add background or use brighter color:
```rust
Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
```

---

## 4. Readability: 9/10

### Strengths
- Excellent use of spacing and whitespace
- Consistent use of `│` separator for visual grouping
- Smart number formatting (format_tokens, format_duration)
- High-contrast color palette

### Issues Found

#### ISSUE 4.1: Dense GPU Summary Text (MINOR)
**Location**: Lines 2318-2332
**Problem**: Long line with many separators; hard to scan quickly
```
GPU: 10.5/24.0GB (43%)  │  68°C  │  150.0W  │  75%
```
**Recommendation**: Break into multiple lines or add more spacing

#### ISSUE 4.2: Loss Value Precision Inconsistency (LOW)
**Location**: Various
**Problem**: Loss shown as {:.3} in run list (line 1635) but {:.4} in some panels
**Impact**: Minor confusion about actual precision
**Fix**: Standardize to {:.4} everywhere or document why different

---

## 5. Navigation: 6/10 (NEEDS IMPROVEMENT)

### Strengths
- Multiple navigation methods (Tab, numbers, vim keys, arrows)
- Context-aware [/] for different views
- Good use of modal overlay for help

### Issues Found

#### ISSUE 5.1: Number Key Tab Mapping Confusion (CRITICAL)
**Location**: Lines 917-925
**Problem**: Tab indices don't match visible tab positions due to Dashboard being added
```rust
KeyCode::Char('0') => MainTab::Dashboard,  // Correct
KeyCode::Char('1') => MainTab::Overview,   // Correct
// But visible tabs show: Dashboard(1) Overview(2) Charts(3)...
```
**Impact**: HIGH - Users press "1" expecting first visible tab but get second
**Current Mapping**:
- 0 = Dashboard (tab 1)
- 1 = Overview (tab 2)
- 2 = Charts (tab 3)

**CRITICAL FIX REQUIRED**: Either:
1. Make Dashboard accessible via '1' and shift others, OR
2. Show "(0)" label in tab header to indicate Dashboard is "tab 0"

#### ISSUE 5.2: Missing Zoom Documentation (MEDIUM)
**Location**: Lines 3770-3778 (help overlay)
**Problem**: Zoom controls (+/-/f) shown but not explained clearly
**Fix**: Add chart range display to footer or chart title showing current zoom level

#### ISSUE 5.3: No Visual Indicator for Sub-Views (MINOR)
**Location**: GPU/Charts/Concepts tabs
**Problem**: User doesn't know [/] will cycle views until they try
**Fix**: Add hint to title bar: "Charts [←/→ to cycle]"

#### ISSUE 5.4: Esc Key Dual Function (LOW)
**Location**: Lines 887, 897
**Problem**: Esc closes help overlay OR quits app depending on context
```rust
KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('h') => {  // Close help
    self.show_help_overlay = false;
}
KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,  // Quit
```
**Impact**: Low risk but could be surprising
**Recommendation**: Document in help that Esc closes overlay first, then quits

---

## 6. Help Text Coverage: 7/10

### Strengths
- Comprehensive keyboard shortcuts documented
- Both overlay and dedicated Help tab
- Color-coded by category

### Issues Found

#### ISSUE 6.1: Zoom Controls Not in Help Tab (MEDIUM)
**Location**: Line 3600+ (help tab)
**Problem**: +/-/f zoom controls only in overlay, not in Help tab
**Fix**: Add "Zoom/Range" section to help tab

#### ISSUE 6.2: Dashboard Tab Not Mentioned (MEDIUM)
**Location**: Lines 3760-3768
**Problem**: Help overlay mentions tabs 4-5 (Network/Concepts) but not 0 (Dashboard)
**Fix**: Add line:
```rust
Line::from(vec![
    Span::styled(" 0        ", Style::default().fg(colors::HELP_KEY)),
    Span::raw("Dashboard - unified view of all key metrics"),
]),
```

#### ISSUE 6.3: Clear Command Undocumented Risk (HIGH)
**Location**: Line 3643-3644
**Problem**: 'c' clears metrics cache but doesn't warn user this is destructive
**Fix**: Change help text:
```rust
Span::raw("Clear cached metrics for selected run (reloads from disk)"),
```

---

## 7. Responsive Layout: 5/10 (NEEDS WORK)

### Strengths
- Uses percentage-based constraints
- Minimum constraints prevent total collapse

### Issues Found

#### ISSUE 7.1: No Small Terminal Handling (CRITICAL)
**Location**: Throughout
**Problem**: No graceful degradation for terminals <80x24
**Impact**: Layout breaks badly at 40-column or 12-line terminals
**Recommendation**: Add terminal size check in `draw()`:
```rust
let (width, height) = (f.area().width, f.area().height);
if width < 80 || height < 24 {
    // Show "Terminal too small" message
    return;
}
```

#### ISSUE 7.2: Dashboard Not Responsive (HIGH)
**Location**: Lines 1137-1203
**Problem**: 3-column layout doesn't stack at narrow widths
**Fix**: Detect width and switch to 2-column or vertical stack:
```rust
let cols = if area.width < 120 {
    // Stack vertically instead
    Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Percentage(33); 3])
} else {
    // Original 3-column layout
}
```

#### ISSUE 7.3: Chart Axis Labels Overlap at Small Sizes (MEDIUM)
**Location**: Lines 1885-1901
**Problem**: Y-axis labels show {:.2} precision always, causing overlap in narrow charts
**Recommendation**: Reduce precision at small widths or rotate labels

---

## Summary of Critical Issues

| Priority | Issue | Impact | Fix Complexity |
|----------|-------|--------|----------------|
| CRITICAL | 5.1 - Tab number mapping confusion | High user frustration | Easy (1 line) |
| CRITICAL | 7.1 - No small terminal handling | App unusable <80x24 | Medium (10-20 lines) |
| HIGH | 6.3 - Clear command not documented | Data loss risk | Easy (1 line) |
| HIGH | 7.2 - Dashboard not responsive | Poor UX at 80-100 cols | Hard (50+ lines) |
| MEDIUM | 1.2 - Tab number mismatch in help | Confusing docs | Easy (2 lines) |
| MEDIUM | 2.1 - Dashboard column proportions | Cramped at small sizes | Easy (3 lines) |
| MEDIUM | 5.2 - Missing zoom documentation | Feature discoverability | Easy (5 lines) |

---

## Recommended Immediate Fixes (High ROI)

1. **Fix tab number mapping** (Issue 5.1) - 1 line change, huge UX improvement
2. **Update help text for tabs** (Issues 1.2, 6.2) - 3 lines, prevents confusion
3. **Add terminal size check** (Issue 7.1) - 15 lines, prevents broken layout
4. **Fix CurveQuality plateau color** (Issue 1.1) - 1 line, semantic consistency
5. **Clarify clear command in help** (Issue 6.3) - 1 line, prevents accidents
6. **Add zoom level indicator** (Issue 5.2) - 5 lines, improves discoverability

**Total: ~26 lines of code for 6 high-impact fixes**

---

## Overall Assessment

**Strengths**:
- Clean, professional visual design
- Comprehensive feature set
- Good keyboard-driven navigation
- Excellent color semantic consistency

**Weaknesses**:
- Tab numbering is confusing (critical usability issue)
- Not tested for small terminal sizes
- Help documentation has gaps
- Some layout proportions need tuning

**Grade**: 7.5/10 - Excellent foundation, needs polish for production readiness

---

## Accessibility Notes

- High contrast colors are good for visibility
- No color-only information (icons + text used)
- Keyboard-only navigation works well
- Consider adding screen reader hints in future (not critical for TUI)

---

## Testing Recommendations

1. Test at multiple terminal sizes: 40x12, 80x24, 120x40, 200x60
2. Test with different color schemes (light/dark backgrounds)
3. Test all navigation paths (Tab, numbers, arrows, vim keys)
4. Test help overlay at minimum terminal size
5. Verify all documented shortcuts work as described
