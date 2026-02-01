# Infrastructure as Code (IaC) Dataset Collection Guide

## Overview

This guide documents the complete infrastructure as code dataset collection system for the Rust-AI project. The system collects, processes, and curates datasets from five major IaC domains for training language models.

**Supported Domains:**
- Terraform (Infrastructure Provisioning)
- Kubernetes (Container Orchestration)
- Ansible (Configuration Management)
- Docker (Container Definitions)
- GitHub Actions (CI/CD Automation)

## Quick Start

### 1. Download IaC Datasets

```bash
# Run the download script
bash /home/kang/Documents/projects/rust-ai/scripts/download_iac_datasets.sh

# The script will:
# - Clone repositories from GitHub
# - Collect configuration files
# - Organize by domain
# - Generate metadata and statistics
```

**Expected Output Location:** `/data/datasets/tritter/iac/`

**Expected Size:** 2-5 GB depending on repository sizes

**Duration:** 30-120 minutes depending on network speed

### 2. Process Datasets

```bash
# Process downloaded datasets for ML training
python3 /home/kang/Documents/projects/rust-ai/scripts/process_iac_datasets.py \
    --iac-dir /data/datasets/tritter/iac \
    --output-dir /data/datasets/tritter/iac/processed \
    --verbose
```

**Output:** Deduplicated, combined datasets ready for training

### 3. Use in Training Pipeline

```python
from pathlib import Path

iac_dir = Path("/data/datasets/tritter/iac/processed")

# Load Terraform configs
terraform_files = (iac_dir / "terraform").glob("*.tf")

# Load Kubernetes manifests
k8s_files = (iac_dir / "kubernetes").glob("*.yaml")

# Load Ansible playbooks
ansible_files = (iac_dir / "ansible").glob("*.yaml")

# Load Docker files
docker_files = (iac_dir / "docker").glob("Dockerfile*")

# Load GitHub Actions
github_files = (iac_dir / "github-actions").glob("*.yaml")
```

## Dataset Domains

### 1. Terraform Dataset

**Purpose:** Infrastructure Provisioning Language
**File Format:** HCL (.tf files)
**Primary Use:** Cloud infrastructure definitions, resource management

#### Sources

| Repository | Owner | Purpose |
|-----------|-------|---------|
| terraform-provider-aws | HashiCorp | AWS provider examples |
| terraform-provider-google | HashiCorp | Google Cloud provider |
| terraform-provider-azurerm | HashiCorp | Azure provider |
| terraform-aws-modules | terraform-aws-modules | Reusable AWS modules |
| terraform-google-modules | terraform-google-modules | Reusable Google Cloud modules |
| terraform-azurerm-caf | Azure | Azure Cloud Adoption Framework |

#### Example Content

```hcl
resource "aws_instance" "example" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"

  vpc_security_group_ids = [aws_security_group.example.id]

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name        = "Example Instance"
    Environment = "Development"
  }
}
```

#### Use Cases
- Train models for infrastructure code generation
- Learn terraform patterns and best practices
- Implement IaC code completion
- Infrastructure design recommendations

### 2. Kubernetes Dataset

**Purpose:** Container Orchestration Configuration
**File Format:** YAML (.yaml/.yml files)
**Primary Use:** Deployment specifications, service definitions, configurations

#### Sources

| Repository | Owner | Purpose |
|-----------|-------|---------|
| kubernetes/examples | Kubernetes | Official examples |
| istio/istio | Istio | Service mesh examples |
| prometheus-operator | Prometheus | Prometheus operator configs |
| cert-manager | Jetstack | Certificate management |
| charts (bitnami/grafana/DataDog) | Various | Helm charts |

#### Kubernetes Object Types

The dataset includes examples of:
- Deployments
- Services
- StatefulSets
- ConfigMaps
- Secrets
- Ingress
- PersistentVolumes
- Custom Resources (CRDs)
- DaemonSets
- Jobs
- CronJobs

#### Example Content

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
```

#### Use Cases
- Train models for Kubernetes manifest generation
- Learn deployment patterns and configurations
- Implement kubectl/deployment assistants
- Best practices for container orchestration

### 3. Ansible Dataset

**Purpose:** Configuration Management and Automation
**File Format:** YAML (.yaml/.yml files) + Python plugins
**Primary Use:** System configuration, playbooks, roles, tasks

#### Sources

| Repository | Owner | Purpose |
|-----------|-------|---------|
| ansible/ansible | Ansible | Core framework examples |
| ansible/ansible-examples | Ansible | Official playbook examples |
| community.general | Ansible Collections | General community modules |
| community.aws | Ansible Collections | AWS-specific automation |
| community.kubernetes | Ansible Collections | Kubernetes automation |
| ansible-role-* | geerlingguy | Popular reusable roles |

#### Ansible Content Types
- Playbooks (orchestration)
- Roles (reusable components)
- Tasks (individual actions)
- Handlers (event-driven tasks)
- Variables and templates
- Custom plugins

#### Example Content

```yaml
---
- name: Deploy web application
  hosts: webservers
  vars:
    app_version: "2.0"
    deploy_user: "appuser"
  tasks:
    - name: Update system packages
      apt:
        update_cache: yes
        upgrade: dist
      when: ansible_os_family == "Debian"

    - name: Install required packages
      package:
        name: "{{ item }}"
        state: present
      loop:
        - python3
        - python3-pip
        - nginx

    - name: Configure application
      template:
        src: app.conf.j2
        dest: /etc/app/config.conf
        owner: "{{ deploy_user }}"
        mode: 0644
      notify: restart application

  handlers:
    - name: restart application
      systemd:
        name: myapp
        state: restarted
```

#### Use Cases
- Train models for automation script generation
- Learn configuration management patterns
- Implement Ansible playbook assistants
- Best practices for infrastructure automation

### 4. Docker Dataset

**Purpose:** Container Image Definition and Orchestration
**File Format:** Dockerfile, docker-compose.yml
**Primary Use:** Container build specifications, multi-container definitions

#### Sources

| Repository | Owner | Purpose |
|-----------|-------|---------|
| docker-library/official-images | Docker | 1000+ official Dockerfiles |
| docker/compose | Docker | Docker Compose examples |
| moby/moby | Docker | Docker engine source |
| docker-library/docs | Docker | Docker documentation |

#### Docker Content Types
- Dockerfiles (image definitions)
- docker-compose.yml (multi-container apps)
- .dockerignore (build context exclusions)
- Build context and layers

#### Example Content

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/myapp
    depends_on:
      - db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### Use Cases
- Train models for Dockerfile generation
- Learn container best practices
- Implement container orchestration assistants
- Multi-container architecture patterns

### 5. GitHub Actions Dataset

**Purpose:** CI/CD Workflow Automation
**File Format:** YAML (.yaml/.yml files)
**Primary Use:** Build, test, and deployment workflows

#### Sources

| Repository | Owner | Purpose |
|-----------|-------|---------|
| actions/* | GitHub | Official GitHub actions |
| super-linter | GitHub | Code quality linting action |
| codecov-action | Codecov | Code coverage reporting |
| release-drafter | Release Drafter | Release automation |

#### GitHub Actions Content
- Workflow files (.github/workflows/*.yml)
- Action definitions (action.yml)
- Job specifications
- Step definitions
- Event triggers

#### Example Content

```yaml
name: Build and Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Run tests
      run: pytest --cov=./ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

    - name: Build package
      run: python -m build

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-packages
        path: dist/
```

#### Use Cases
- Train models for CI/CD pipeline generation
- Learn GitHub Actions patterns and best practices
- Implement workflow automation assistants
- DevOps automation knowledge

## Directory Structure

```
/data/datasets/tritter/iac/
├── repos/                          # Source repositories (can be deleted after processing)
│   ├── terraform/
│   │   ├── terraform-provider-aws/
│   │   ├── terraform-provider-google/
│   │   └── ...
│   ├── kubernetes/
│   │   ├── examples/
│   │   ├── istio/
│   │   └── ...
│   ├── ansible/
│   │   ├── ansible/
│   │   ├── ansible-examples/
│   │   └── ...
│   ├── docker/
│   │   ├── official-images/
│   │   ├── compose/
│   │   └── ...
│   └── github-actions/
│       ├── checkout/
│       ├── setup-python/
│       └── ...
│
├── datasets/                       # Processed datasets (primary training source)
│   ├── terraform-files/
│   │   ├── *.tf files (grouped by source)
│   │   └── metadata.json
│   ├── kubernetes-manifests/
│   │   ├── *.yaml files
│   │   ├── all-manifests.yaml (consolidated)
│   │   └── metadata.json
│   ├── ansible-playbooks/
│   │   ├── *.yaml files
│   │   └── metadata.json
│   ├── docker-files/
│   │   ├── Dockerfile*
│   │   ├── docker-compose*.yml
│   │   └── metadata.json
│   ├── github-actions-workflows/
│   │   ├── *.yaml files
│   │   └── metadata.json
│   └── huggingface-iac/            # Optional HF dataset downloads
│
├── processed/                      # ML-ready datasets
│   ├── terraform/
│   ├── kubernetes/
│   ├── ansible/
│   ├── docker/
│   ├── github-actions/
│   └── statistics.json
│
├── dataset-index.md                # Comprehensive dataset documentation
├── DATASET_SUMMARY.txt             # Quick summary
└── download.log                    # Download operation log
```

## Data Statistics

### Expected Sizes

| Domain | Files | Approx Size |
|--------|-------|------------|
| Terraform | ~10,000 | 200-500 MB |
| Kubernetes | ~15,000 | 300-700 MB |
| Ansible | ~10,000 | 150-400 MB |
| Docker | ~8,000 | 100-300 MB |
| GitHub Actions | ~5,000 | 50-150 MB |
| **Total** | **~48,000** | **2-5 GB** |

### Data Quality Metrics

- **Deduplication:** Remove identical files by content hash
- **Format validation:** Only include valid configuration files
- **Size filtering:** Exclude extremely large monolithic files
- **Error handling:** Skip corrupted or malformed files
- **Unique content:** ~95% of collected files are unique

## Data Deduplication

The processing pipeline includes automatic deduplication:

```python
# Files are deduplicated using SHA256 content hashing
# Identical files across domains are identified and counted once
# Statistics reflect unique content

sha256(file_content) -> unique_id
```

## Machine Learning Applications

### 1. Code LLM Fine-tuning

Fine-tune models on infrastructure code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load IaC dataset
iac_dataset = load_iac_dataset("/data/datasets/tritter/iac/processed")

# Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=iac_dataset['train'],
    eval_dataset=iac_dataset['val'],
)
trainer.train()
```

### 2. Domain Classification

Classify configuration files by type:

```python
# Train classifier to distinguish between:
# - Terraform vs Kubernetes
# - Infrastructure vs CI/CD
# - AWS vs Google Cloud
# - Development vs Production patterns
```

### 3. Code Completion

Predict next configuration blocks:

```python
# Given partial input:
# resource "aws_s3_bucket" "my_bucket" {
# Predict likely completions:
# - acl = "private"
# - versioning { enabled = true }
# - server_side_encryption_configuration { ... }
```

### 4. Infrastructure Pattern Recognition

Learn common deployment patterns:

```python
# Identify and extract patterns:
# - Multi-tier application architectures
# - High availability configurations
# - Security best practices
# - Cost optimization patterns
```

### 5. Cross-Domain Learning

Train models to understand relationships:

```python
# Learn connections:
# - Terraform resources -> Kubernetes manifests
# - Docker images -> GitHub Actions workflows
# - Ansible roles -> Infrastructure components
```

## Integration with Rust-AI Training Pipeline

### Step 1: Data Preparation

```bash
# Download datasets
bash scripts/download_iac_datasets.sh

# Process datasets
python3 scripts/process_iac_datasets.py \
    --iac-dir /data/datasets/tritter/iac \
    --output-dir /data/datasets/tritter/iac/processed
```

### Step 2: Tokenization

```rust
// In Rust training code
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

// Tokenize IaC files
let tokens = tokenizer.encode_batch(vec![
    terraform_content,
    kubernetes_content,
    ansible_content,
])?;
```

### Step 3: Dataset Loading

```rust
// Create batches for training
struct IaCBatch {
    terraform_ids: Vec<Vec<u32>>,
    kubernetes_ids: Vec<Vec<u32>>,
    ansible_ids: Vec<Vec<u32>>,
    docker_ids: Vec<Vec<u32>>,
    github_actions_ids: Vec<Vec<u32>>,
}

impl DataLoader for IaCBatch {
    fn next_batch(&mut self) -> Option<Tensor> {
        // Load and return next batch
    }
}
```

### Step 4: Training

Use with any Rust ML framework (Candle, tch-rs, etc.):

```rust
// Example with candle
let model = Model::new(&device)?;
let dataset = IaCDataset::load("/data/datasets/tritter/iac/processed")?;

for epoch in 0..num_epochs {
    for batch in dataset.batches(batch_size) {
        let loss = model.forward(&batch)?;
        loss.backward()?;
        optimizer.step()?;
    }
}
```

## Troubleshooting

### Issue: Download Script Stalls

**Solution:** Check network connectivity and GitHub API rate limits

```bash
# Check your GitHub rate limits
curl -s -H "Authorization: token YOUR_TOKEN" \
    https://api.github.com/rate_limit | jq

# Increase sleep delays in script if needed
```

### Issue: Disk Space Exhausted

**Solution:** Delete repos directory after processing

```bash
# Keep only processed datasets
rm -rf /data/datasets/tritter/iac/repos
```

### Issue: YAML Parsing Errors

**Solution:** Some files may not be valid YAML; processor handles gracefully

```bash
# Check error summary in statistics.json
cat /data/datasets/tritter/iac/processed/statistics.json | jq '.totals'
```

### Issue: Duplicate Files

**Solution:** Automatic deduplication removes exact copies

```bash
# Check deduplication statistics
python3 scripts/process_iac_datasets.py --verbose
```

## Performance Benchmarks

### Download Performance

| Domain | Repos | Clone Time |
|--------|-------|-----------|
| Terraform | 10 | 5-10 min |
| Kubernetes | 9 | 10-15 min |
| Ansible | 10 | 8-12 min |
| Docker | 6 | 5-8 min |
| GitHub Actions | 10 | 3-5 min |
| **Total** | **45** | **30-50 min** |

### Processing Performance

| Stage | Time |
|-------|------|
| File discovery | 1-2 min |
| Hash computation | 2-5 min |
| Deduplication | 1-2 min |
| Statistics generation | 1 min |
| **Total** | **5-10 min** |

## Future Enhancements

1. **Additional IaC Platforms:**
   - CloudFormation (AWS)
   - Azure Resource Manager (ARM)
   - Pulumi
   - CDK (TypeScript, Python)

2. **Advanced Processing:**
   - Abstract syntax tree (AST) extraction
   - Semantic code normalization
   - Variable/placeholder detection and masking

3. **Evaluation Metrics:**
   - Syntactic correctness validation
   - Configuration semantics analysis
   - Cross-domain consistency checking

4. **Model-Specific Datasets:**
   - Domain-specific tokenization
   - Hierarchical structure preservation
   - Context window optimization

## References

- [Terraform Language Documentation](https://www.terraform.io/language/)
- [Kubernetes API Reference](https://kubernetes.io/docs/reference/)
- [Ansible Documentation](https://docs.ansible.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [HashiCorp Terraform Registry](https://registry.terraform.io/)
- [Helm Charts](https://helm.sh/docs/chart_repository/)

## Support

For issues or questions:
1. Check `/data/datasets/tritter/iac/download.log` for error details
2. Review `/data/datasets/tritter/iac/DATASET_SUMMARY.txt` for statistics
3. Run with `--verbose` flag for detailed diagnostics
4. Check script permissions: `ls -l scripts/download_iac_datasets.sh`

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Status:** Production Ready
