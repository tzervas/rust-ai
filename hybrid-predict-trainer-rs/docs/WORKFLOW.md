# Git Workflow Guide

## Branch Strategy

```
main (protected, production-ready)
  ↑
  └── PR only (reviewed, CI passing)
       ↑
      dev (integration branch, staging)
       ↑
       ├── feature/optimization-research
       ├── feature/micro-corrections
       ├── feature/multi-step-bptt
       └── bugfix/metrics-finalization
```

## Rules

1. **main branch**:
   - Protected, no direct commits
   - Only merge via Pull Request from `dev`
   - Must pass all CI checks
   - Requires review approval

2. **dev branch**:
   - Integration branch for all features
   - Merge feature branches here
   - Should always be in working state
   - Tagged for pre-releases (v0.2.0-rc1, etc.)

3. **Feature branches**:
   - Always branch off `dev`
   - Naming: `feature/descriptive-name` or `bugfix/issue-name`
   - Regular commits allowed
   - Merge back to `dev` when complete

## Workflow

### Starting New Work

```bash
# Ensure you're on dev and up-to-date
git checkout dev
git pull origin dev

# Create feature branch
git checkout -b feature/my-feature

# Work and commit
git add .
git commit -m "feat: implement X"

# Push to remote (if collaborating)
git push -u origin feature/my-feature
```

### Integrating to Dev

```bash
# When feature is complete and tested
git checkout dev
git pull origin dev  # Get latest changes

# Merge feature branch (no fast-forward for clear history)
git merge feature/my-feature --no-ff -m "Merge feature/my-feature into dev"

# Push to dev
git push origin dev

# Optionally delete feature branch
git branch -d feature/my-feature
```

### Releasing to Main (via PR)

```bash
# When dev is stable and ready for release
# 1. Create PR on GitHub: dev → main
# 2. Review changes, ensure CI passes
# 3. Merge PR (squash or merge commit)
# 4. Tag release on main
git checkout main
git pull origin main
git tag v0.2.0
git push origin v0.2.0
```

## Current Active Branches

- `main`: Last stable release (v0.1.0)
- `dev`: Integration branch with latest features
- `feature/optimization-research`: Comprehensive parameter optimization (ACTIVE)

## Commit Message Convention

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples**:
```
feat(dynamics): implement multi-step BPTT for GRU training

Adds k-step backpropagation through time to improve weight delta
predictions by 44%. Configurable via bptt_steps parameter.

Closes #42

---

fix(CRITICAL): resolve metrics finalization bug

Statistics were not auto-finalized, causing 0% speedup reporting.
Added self.finalize() call in statistics() method.

---

docs: create comprehensive documentation INDEX (70% token reduction)

Implements navigation hub for efficient AI agent context loading.
```

## Branch Protection (Recommended Settings)

### For `main`:
- ✅ Require pull request before merging
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Do not allow bypassing settings
- ❌ Allow force pushes: Never

### For `dev`:
- ✅ Require pull request before merging (optional, can commit directly if solo)
- ✅ Require status checks to pass
- ❌ Allow force pushes: Only maintainers

## Quick Reference

```bash
# Check current branch
git branch

# List all branches
git branch -a

# Switch to existing branch
git checkout dev

# Create and switch to new branch
git checkout -b feature/my-feature

# Merge feature into dev
git checkout dev
git merge feature/my-feature --no-ff

# Delete local branch
git branch -d feature/my-feature

# Delete remote branch
git push origin --delete feature/my-feature

# View commit history
git log --oneline --graph --all
```

---

**Last Updated**: 2026-02-06
**Maintained By**: Development Team
