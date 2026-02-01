# IaC Dataset Collection - Quick Start Guide

## TL;DR - Get Started in 5 Minutes

```bash
# 1. Download datasets (30-120 minutes depending on network)
bash /home/kang/Documents/projects/rust-ai/scripts/download_iac_datasets.sh

# 2. Process datasets (5-10 minutes)
python3 /home/kang/Documents/projects/rust-ai/scripts/process_iac_datasets.py \
    --iac-dir /data/datasets/tritter/iac

# 3. Check results
ls -lah /data/datasets/tritter/iac/processed/
cat /data/datasets/tritter/iac/processed/statistics.json
```

---

## What You Get

After running the scripts, you'll have:

- **~48,000 infrastructure code files** across 5 domains
- **2-5 GB** of deduplicated, organized data
- **Metadata and statistics** for each domain
- **ML-ready datasets** for training language models

### Directory Structure After Download

```
/data/datasets/tritter/iac/
├── repos/                          # Source repositories (41 total)
├── datasets/                       # Collected IaC files
│   ├── terraform-files/            # ~10k .tf files
│   ├── kubernetes-manifests/       # ~15k .yaml files
│   ├── ansible-playbooks/          # ~10k playbooks
│   ├── docker-files/               # ~8k Dockerfiles
│   └── github-actions-workflows/   # ~5k workflows
├── processed/                      # ML-ready datasets
├── dataset-index.md                # Full documentation
└── DATASET_SUMMARY.txt             # Quick summary
```

---

## The 5 Domains

### 1. Terraform (~10,000 files)
**What:** Infrastructure provisioning language (HCL)
**Example:** AWS EC2 instances, VPCs, databases
**Source:** HashiCorp providers + community modules

### 2. Kubernetes (~15,000 files)
**What:** Container orchestration manifests (YAML)
**Example:** Deployments, Services, StatefulSets, Ingress
**Source:** Official examples + Helm charts + operators

### 3. Ansible (~10,000 files)
**What:** Configuration management playbooks (YAML)
**Example:** System configuration, package installation, app deployment
**Source:** Official examples + community collections

### 4. Docker (~8,000 files)
**What:** Container definitions + orchestration (Dockerfile, YAML)
**Example:** Application containers, multi-container setups
**Source:** 1000+ official images + real-world examples

### 5. GitHub Actions (~5,000 files)
**What:** CI/CD workflow automation (YAML)
**Example:** Build, test, deploy workflows
**Source:** Official actions + community workflows

---

## Installation Requirements

```bash
# Check Python 3
python3 --version  # Should be 3.8+

# Install required package
pip install pyyaml

# Verify HuggingFace CLI
/home/kang/.local/bin/hf auth whoami

# Verify disk space (need 5-10 GB)
df -h /data
```

---

## Running the Scripts

### Step 1: Download Datasets

```bash
bash /home/kang/Documents/projects/rust-ai/scripts/download_iac_datasets.sh
```

**What it does:**
- Clones 41 GitHub repositories
- Collects configuration files by domain
- Creates metadata for each domain
- Logs all operations to download.log

**Time:** 30-120 minutes (depends on network and disk speed)
**Output:** `/data/datasets/tritter/iac/datasets/`

**Monitor progress:**
```bash
tail -f /data/datasets/tritter/iac/download.log
```

### Step 2: Process Datasets

```bash
python3 /home/kang/Documents/projects/rust-ai/scripts/process_iac_datasets.py \
    --iac-dir /data/datasets/tritter/iac \
    --output-dir /data/datasets/tritter/iac/processed \
    --verbose
```

**What it does:**
- Deduplicates files by content hash
- Combines related files
- Validates syntax
- Generates statistics
- Creates ML-ready datasets

**Time:** 5-10 minutes
**Output:** `/data/datasets/tritter/iac/processed/`

### Step 3: Verify Results

```bash
# Check if processing completed successfully
ls -la /data/datasets/tritter/iac/processed/

# Review statistics
cat /data/datasets/tritter/iac/processed/statistics.json | python3 -m json.tool

# Check individual domains
find /data/datasets/tritter/iac/processed -type f | head -20
```

---

## Usage Examples

### Load in Python for Training

```python
from pathlib import Path
import json

iac_base = Path("/data/datasets/tritter/iac/processed")

# Load statistics
with open(iac_base / "statistics.json") as f:
    stats = json.load(f)
    print(f"Total files: {stats['totals']['total_files']}")

# Read Terraform configs
terraform_files = (iac_base / "terraform").glob("*.tf")
for tf_file in terraform_files:
    with open(tf_file) as f:
        content = f.read()
        # Use content for training
```

### Tokenize for Model Training

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Read IaC files and tokenize
iac_base = Path("/data/datasets/tritter/iac/processed")

for domain in ["terraform", "kubernetes", "ansible", "docker", "github-actions"]:
    domain_dir = iac_base / domain
    if domain_dir.exists():
        for file_path in domain_dir.glob("*.yaml") or domain_dir.glob("*.tf"):
            with open(file_path) as f:
                content = f.read()
                tokens = tokenizer.encode(content)
                # Store or process tokens
```

### Create Train/Val/Test Split

```python
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

iac_base = Path("/data/datasets/tritter/iac/processed")

# Collect all files
all_files = []
for domain in ["terraform", "kubernetes", "ansible", "docker", "github-actions"]:
    domain_dir = iac_base / domain
    if domain_dir.exists():
        all_files.extend(domain_dir.glob("*"))

# Create splits (70/15/15)
train, temp = train_test_split(all_files, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
splits = {
    'train': [str(f) for f in train],
    'val': [str(f) for f in val],
    'test': [str(f) for f in test],
}

with open(iac_base / "splits.json", 'w') as f:
    json.dump(splits, f, indent=2)
```

---

## Common Issues & Fixes

### Issue: "Disk space exhausted"
```bash
# Check available space
df -h /data

# If needed, delete repos directory (can re-download)
rm -rf /data/datasets/tritter/iac/repos

# Keep only processed/ and datasets/
```

### Issue: "GitHub rate limit exceeded"
```bash
# Wait a few hours or authenticate with GitHub token
export GITHUB_TOKEN=your_token_here
# Re-run the download script
```

### Issue: "Network timeout during download"
```bash
# Increase timeouts in the script:
# Edit download_iac_datasets.sh
# Change: git clone --depth 1 ...
# To: git clone --depth 1 -v ... (with retry logic)

# Or clone individual repositories manually:
cd /data/datasets/tritter/iac/repos/terraform
git clone https://github.com/hashicorp/terraform-provider-aws.git
```

### Issue: "Python module 'yaml' not found"
```bash
pip install pyyaml
```

### Issue: "Permission denied" on script files
```bash
chmod +x /home/kang/Documents/projects/rust-ai/scripts/download_iac_datasets.sh
chmod +x /home/kang/Documents/projects/rust-ai/scripts/process_iac_datasets.py
```

---

## Performance Tips

### Faster Downloads

1. **Use better internet connection** (wired > wireless)
2. **Run during off-peak hours** to avoid GitHub rate limits
3. **Increase --depth** in script if you want full history:
   ```bash
   # Instead of --depth 1, use --depth 5 (takes longer but gets more context)
   ```

### Faster Processing

1. **Reduce--verbose output:**
   ```bash
   python3 process_iac_datasets.py --iac-dir /data/datasets/tritter/iac
   ```

2. **Process in parallel** (for advanced users):
   ```bash
   # Process domains separately
   for domain in terraform kubernetes ansible docker github-actions; do
       python3 process_iac_datasets.py --domain $domain &
   done
   ```

### Reduce Storage

1. **Delete repos/ after processing:**
   ```bash
   rm -rf /data/datasets/tritter/iac/repos
   # Saves ~5-10 GB, keeps processed datasets
   ```

2. **Delete HuggingFace downloads if not used:**
   ```bash
   rm -rf /data/datasets/tritter/iac/datasets/huggingface-iac
   ```

---

## File Statistics

### After Download (all files)

| Domain | Files | Size |
|--------|-------|------|
| Terraform | ~10,000 | 200-500 MB |
| Kubernetes | ~15,000 | 300-700 MB |
| Ansible | ~10,000 | 150-400 MB |
| Docker | ~8,000 | 100-300 MB |
| GitHub Actions | ~5,000 | 50-150 MB |
| **Total** | **~48,000** | **2-5 GB** |

### After Processing (deduplicated)

- **Unique files:** ~45,000 (95% unique)
- **Exact duplicates removed:** ~3,000 files
- **Combined size:** ~1.8-4 GB

---

## Next Steps After Collection

### For LLM Fine-tuning

1. Create vocabulary from IaC files
2. Tokenize all content
3. Create training batches
4. Fine-tune on model of choice

### For Code Analysis

1. Extract syntax trees from configurations
2. Learn configuration patterns
3. Build pattern recognizers
4. Create domain-specific tools

### For Infrastructure Understanding

1. Learn resource relationships
2. Extract deployment topologies
3. Understand best practices
4. Generate recommendations

---

## Advanced Usage

### Filter by Language

```bash
# Extract only Terraform files
find /data/datasets/tritter/iac/processed/terraform -name "*.tf"

# Extract only Kubernetes YAML
find /data/datasets/tritter/iac/processed/kubernetes -name "*.yaml"
```

### Analyze File Sizes

```bash
# Find largest files
find /data/datasets/tritter/iac/processed -type f -exec du -h {} \; | sort -rh | head -20

# Calculate distribution
for domain in terraform kubernetes ansible docker github-actions; do
    size=$(du -sh /data/datasets/tritter/iac/processed/$domain | cut -f1)
    echo "$domain: $size"
done
```

### Extract Specific Patterns

```python
import re
from pathlib import Path

pattern = re.compile(r'resource\s+"([^"]+)"')  # Find Terraform resource types

for tf_file in Path("/data/datasets/tritter/iac/processed/terraform").glob("*.tf"):
    with open(tf_file) as f:
        matches = pattern.findall(f.read())
        for match in matches:
            print(f"{tf_file.name}: {match}")
```

---

## Documentation

### Full Guides
- **Complete Dataset Guide:** `/home/kang/Documents/projects/rust-ai/docs/IaC-DATASET-GUIDE.md`
- **Sources Reference:** `/home/kang/Documents/projects/rust-ai/docs/IaC-SOURCES-REFERENCE.md`

### Generated Outputs
- **Download Log:** `/data/datasets/tritter/iac/download.log`
- **Dataset Index:** `/data/datasets/tritter/iac/dataset-index.md`
- **Statistics:** `/data/datasets/tritter/iac/processed/statistics.json`

---

## Support

### Check Logs for Errors
```bash
# Download operation log
tail -100 /data/datasets/tritter/iac/download.log

# Processing output
cat /data/datasets/tritter/iac/processed/statistics.json | python3 -m json.tool
```

### Verify Dataset Integrity
```bash
# Count files by domain
for domain in terraform kubernetes ansible docker github-actions; do
    echo "$domain: $(find /data/datasets/tritter/iac/processed/$domain -type f 2>/dev/null | wc -l) files"
done

# Check for errors in statistics
cat /data/datasets/tritter/iac/processed/statistics.json | grep -i error
```

---

## Troubleshooting Script

Quick diagnostic script:

```bash
#!/bin/bash
echo "=== IaC Dataset Health Check ==="
echo ""
echo "1. Check directories exist:"
for dir in repos datasets processed; do
    if [ -d "/data/datasets/tritter/iac/$dir" ]; then
        echo "  ✓ $dir exists"
    else
        echo "  ✗ $dir missing"
    fi
done

echo ""
echo "2. Check files by domain:"
for domain in terraform kubernetes ansible docker github-actions; do
    count=$(find /data/datasets/tritter/iac/processed/$domain -type f 2>/dev/null | wc -l)
    echo "  $domain: $count files"
done

echo ""
echo "3. Check disk usage:"
du -sh /data/datasets/tritter/iac/*

echo ""
echo "4. Check statistics file:"
if [ -f "/data/datasets/tritter/iac/processed/statistics.json" ]; then
    echo "  ✓ statistics.json exists"
    cat /data/datasets/tritter/iac/processed/statistics.json | python3 -m json.tool | head -20
else
    echo "  ✗ statistics.json missing"
fi
```

---

## Success Criteria

You've successfully completed the collection if you see:

- ✅ ~48,000 total files collected
- ✅ 2-5 GB total dataset size
- ✅ All 5 domains represented
- ✅ `statistics.json` generated with metrics
- ✅ Deduplication summary in output
- ✅ No critical errors in logs

---

**Ready to train your IaC model!** 🚀

For detailed information, see the full guides in `/home/kang/Documents/projects/rust-ai/docs/`

**Last Updated:** 2026-01-31
