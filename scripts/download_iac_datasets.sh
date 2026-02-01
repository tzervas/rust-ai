#!/bin/bash
# Infrastructure as Code (IaC) Dataset Collection Script
# Collects Terraform, Kubernetes, Ansible, Docker, and GitHub Actions datasets
# Target directory: /data/datasets/tritter/iac/

set -e

# Configuration
IAC_DIR="/data/datasets/tritter/iac"
HF_CLI="/home/kang/.local/bin/hf"
LOG_FILE="${IAC_DIR}/download.log"
REPOS_DIR="${IAC_DIR}/repos"
DATASETS_DIR="${IAC_DIR}/datasets"
TEMP_DIR="${IAC_DIR}/temp"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Create directories
mkdir -p "$IAC_DIR"
mkdir -p "$REPOS_DIR"
mkdir -p "$DATASETS_DIR"
mkdir -p "$TEMP_DIR"

log "======================================================"
log "Infrastructure as Code (IaC) Dataset Collection"
log "======================================================"
log "Target directory: $IAC_DIR"
log "Repositories: $REPOS_DIR"
log "Datasets: $DATASETS_DIR"
log "Log file: $LOG_FILE"

# Verify HF CLI
if ! command -v "$HF_CLI" &> /dev/null; then
    error "HuggingFace CLI not found at $HF_CLI"
    exit 1
fi

log "HuggingFace CLI verified: $HF_CLI"

# ============================================
# 1. TERRAFORM DATASETS
# ============================================
log ""
log "=== TERRAFORM DATASETS ==="

# 1.1 Terraform Registry Modules (via GitHub)
log "Downloading Terraform Registry modules..."
TERRAFORM_DIR="$REPOS_DIR/terraform"
mkdir -p "$TERRAFORM_DIR"

terraform_repos=(
    "hashicorp/terraform-provider-aws"
    "hashicorp/terraform-provider-google"
    "hashicorp/terraform-provider-azurerm"
    "terraform-aws-modules/terraform-aws-vpc"
    "terraform-aws-modules/terraform-aws-ecs"
    "terraform-aws-modules/terraform-aws-rds"
    "terraform-google-modules/terraform-google-kubernetes-engine"
    "terraform-google-modules/terraform-google-compute-engine"
    "Azure/terraform-azurerm-naming"
    "Azure/terraform-azurerm-caf"
)

for repo in "${terraform_repos[@]}"; do
    repo_name=$(basename "$repo")
    repo_path="$TERRAFORM_DIR/$repo_name"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $repo..."
        cd "$repo_path" && git pull --quiet && cd - > /dev/null 2>&1 || warn "Failed to update $repo"
    else
        log "Cloning $repo..."
        git clone --depth 1 "https://github.com/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $repo"
    fi
done

# 1.2 Collect Terraform files into dataset
info "Processing Terraform files into dataset..."
TERRAFORM_DATASET="$DATASETS_DIR/terraform-files"
mkdir -p "$TERRAFORM_DATASET"

find "$TERRAFORM_DIR" -name "*.tf" -type f | head -10000 | while read tf_file; do
    relative_path="${tf_file#$TERRAFORM_DIR/}"
    dest_dir="$TERRAFORM_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$tf_file" "$dest_dir/" 2>/dev/null || true
done

log "Terraform files processed: $(find "$TERRAFORM_DATASET" -name "*.tf" | wc -l) files"

# Create Terraform metadata
cat > "$TERRAFORM_DATASET/metadata.json" << 'EOF'
{
  "name": "terraform-iac-dataset",
  "type": "infrastructure-as-code",
  "language": "hcl",
  "description": "Terraform configuration files from official providers and community modules",
  "sources": [
    "hashicorp/terraform-provider-aws",
    "hashicorp/terraform-provider-google",
    "hashicorp/terraform-provider-azurerm",
    "terraform-aws-modules/*",
    "terraform-google-modules/*",
    "Azure/terraform-azurerm-*"
  ],
  "statistics": {
    "file_format": ".tf",
    "primary_content": "HCL (HashiCorp Configuration Language)"
  }
}
EOF

# ============================================
# 2. KUBERNETES DATASETS
# ============================================
log ""
log "=== KUBERNETES DATASETS ==="

# 2.1 Kubernetes Examples & Official Repos
log "Downloading Kubernetes manifests..."
K8S_DIR="$REPOS_DIR/kubernetes"
mkdir -p "$K8S_DIR"

k8s_repos=(
    "kubernetes/examples"
    "kubernetes/kubernetes"
    "kubernetes/kubernetes-sigs"
    "istio/istio"
    "prometheus-operator/prometheus-operator"
    "jetstack/cert-manager"
    "DataDog/helm-charts"
    "bitnami/charts"
    "grafana/helm-charts"
)

for repo in "${k8s_repos[@]}"; do
    repo_name=$(basename "$repo")
    repo_path="$K8S_DIR/$repo_name"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $repo..."
        cd "$repo_path" && git pull --quiet && cd - > /dev/null 2>&1 || warn "Failed to update $repo"
    else
        log "Cloning $repo (shallow clone)..."
        git clone --depth 1 "https://github.com/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $repo"
    fi
done

# 2.2 Collect Kubernetes manifests
info "Processing Kubernetes manifests into dataset..."
K8S_DATASET="$DATASETS_DIR/kubernetes-manifests"
mkdir -p "$K8S_DATASET"

# Collect YAML manifests
find "$K8S_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) | head -15000 | while read yaml_file; do
    relative_path="${yaml_file#$K8S_DIR/}"
    dest_dir="$K8S_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$yaml_file" "$dest_dir/" 2>/dev/null || true
done

# Create consolidated manifest file
log "Creating consolidated Kubernetes manifests..."
K8S_CONSOLIDATED="$K8S_DATASET/all-manifests.yaml"
find "$K8S_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) -exec cat {} \; 2>/dev/null > "$K8S_CONSOLIDATED" || true

log "Kubernetes manifests collected: $(find "$K8S_DATASET" -name "*.yaml" -o -name "*.yml" | wc -l) files"

# Create K8s metadata
cat > "$K8S_DATASET/metadata.json" << 'EOF'
{
  "name": "kubernetes-iac-dataset",
  "type": "infrastructure-as-code",
  "language": "yaml",
  "description": "Kubernetes manifests from official repositories, Helm charts, and community operators",
  "sources": [
    "kubernetes/examples",
    "kubernetes/kubernetes",
    "istio/istio",
    "prometheus-operator/prometheus-operator",
    "jetstack/cert-manager",
    "bitnami/charts",
    "grafana/helm-charts"
  ],
  "statistics": {
    "file_format": ".yaml, .yml",
    "primary_content": "Kubernetes manifests and Helm templates"
  }
}
EOF

# ============================================
# 3. ANSIBLE DATASETS
# ============================================
log ""
log "=== ANSIBLE DATASETS ==="

# 3.1 Ansible Collections & Playbooks
log "Downloading Ansible playbooks and collections..."
ANSIBLE_DIR="$REPOS_DIR/ansible"
mkdir -p "$ANSIBLE_DIR"

ansible_repos=(
    "ansible/ansible"
    "ansible/ansible-examples"
    "ansible/ansible-lint"
    "ansible-collections/community.general"
    "ansible-collections/community.aws"
    "ansible-collections/community.kubernetes"
    "ansible-collections/community.docker"
    "geerlingguy/ansible-role-geerlingguy.docker"
    "geerlingguy/ansible-role-java"
    "geerlingguy/ansible-role-postgresql"
)

for repo in "${ansible_repos[@]}"; do
    repo_name=$(basename "$repo")
    repo_path="$ANSIBLE_DIR/$repo_name"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $repo..."
        cd "$repo_path" && git pull --quiet && cd - > /dev/null 2>&1 || warn "Failed to update $repo"
    else
        log "Cloning $repo (shallow clone)..."
        git clone --depth 1 "https://github.com/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $repo"
    fi
done

# 3.2 Collect Ansible files
info "Processing Ansible playbooks into dataset..."
ANSIBLE_DATASET="$DATASETS_DIR/ansible-playbooks"
mkdir -p "$ANSIBLE_DATASET"

# Collect YAML playbooks and roles
find "$ANSIBLE_DIR" -type f \( -name "*.yml" -o -name "*.yaml" \) | head -8000 | while read yaml_file; do
    relative_path="${yaml_file#$ANSIBLE_DIR/}"
    dest_dir="$ANSIBLE_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$yaml_file" "$dest_dir/" 2>/dev/null || true
done

# Collect Python plugin files
find "$ANSIBLE_DIR" -type f -path "*/plugins/*.py" | head -2000 | while read py_file; do
    relative_path="${py_file#$ANSIBLE_DIR/}"
    dest_dir="$ANSIBLE_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$py_file" "$dest_dir/" 2>/dev/null || true
done

log "Ansible playbooks collected: $(find "$ANSIBLE_DATASET" -name "*.yml" -o -name "*.yaml" | wc -l) playbooks"

# Create Ansible metadata
cat > "$ANSIBLE_DATASET/metadata.json" << 'EOF'
{
  "name": "ansible-iac-dataset",
  "type": "infrastructure-as-code",
  "language": "yaml",
  "description": "Ansible playbooks, roles, and collections from official sources and community contributions",
  "sources": [
    "ansible/ansible",
    "ansible/ansible-examples",
    "ansible-collections/community.*",
    "geerlingguy/ansible-role-*"
  ],
  "statistics": {
    "file_format": ".yml, .yaml, .py",
    "primary_content": "Ansible playbooks, roles, tasks, and plugins"
  }
}
EOF

# ============================================
# 4. DOCKER DATASETS
# ============================================
log ""
log "=== DOCKER DATASETS ==="

# 4.1 Official Docker Images & Compose examples
log "Downloading Docker examples..."
DOCKER_DIR="$REPOS_DIR/docker"
mkdir -p "$DOCKER_DIR"

docker_repos=(
    "docker-library/official-images"
    "docker/build-push-action"
    "docker/compose"
    "moby/moby"
    "docker/cli"
    "docker-library/docs"
)

for repo in "${docker_repos[@]}"; do
    repo_name=$(basename "$repo")
    repo_path="$DOCKER_DIR/$repo_name"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $repo..."
        cd "$repo_path" && git pull --quiet && cd - > /dev/null 2>&1 || warn "Failed to update $repo"
    else
        log "Cloning $repo (shallow clone)..."
        git clone --depth 1 "https://github.com/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $repo"
    fi
done

# 4.2 Collect Dockerfiles and docker-compose files
info "Processing Docker files into dataset..."
DOCKER_DATASET="$DATASETS_DIR/docker-files"
mkdir -p "$DOCKER_DATASET"

# Collect Dockerfiles
find "$DOCKER_DIR" -type f -name "Dockerfile*" | head -5000 | while read dockerfile; do
    relative_path="${dockerfile#$DOCKER_DIR/}"
    dest_dir="$DOCKER_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$dockerfile" "$dest_dir/" 2>/dev/null || true
done

# Collect docker-compose files
find "$DOCKER_DIR" -type f \( -name "docker-compose*.yml" -o -name "docker-compose*.yaml" \) | head -3000 | while read compose_file; do
    relative_path="${compose_file#$DOCKER_DIR/}"
    dest_dir="$DOCKER_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$compose_file" "$dest_dir/" 2>/dev/null || true
done

# Collect .dockerignore files
find "$DOCKER_DIR" -type f -name ".dockerignore" | while read ignore_file; do
    relative_path="${ignore_file#$DOCKER_DIR/}"
    dest_dir="$DOCKER_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$ignore_file" "$dest_dir/" 2>/dev/null || true
done

log "Docker files collected: $(find "$DOCKER_DATASET" -type f | wc -l) files"

# Create Docker metadata
cat > "$DOCKER_DATASET/metadata.json" << 'EOF'
{
  "name": "docker-iac-dataset",
  "type": "infrastructure-as-code",
  "language": "dockerfile,yaml",
  "description": "Dockerfiles, docker-compose files, and official Docker image configurations",
  "sources": [
    "docker-library/official-images",
    "docker/compose",
    "moby/moby",
    "docker-library/docs"
  ],
  "statistics": {
    "file_format": "Dockerfile, docker-compose.yml/yaml",
    "primary_content": "Docker image definitions and container orchestration"
  }
}
EOF

# ============================================
# 5. GITHUB ACTIONS DATASETS
# ============================================
log ""
log "=== GITHUB ACTIONS DATASETS ==="

# 5.1 GitHub Actions Examples & Workflows
log "Downloading GitHub Actions workflows..."
ACTIONS_DIR="$REPOS_DIR/github-actions"
mkdir -p "$ACTIONS_DIR"

actions_repos=(
    "actions/checkout"
    "actions/setup-python"
    "actions/setup-node"
    "actions/setup-docker"
    "actions/cache"
    "actions/upload-artifact"
    "actions/download-artifact"
    "github/super-linter"
    "release-drafter/release-drafter"
    "codecov/codecov-action"
)

for repo in "${actions_repos[@]}"; do
    repo_name=$(basename "$repo")
    repo_path="$ACTIONS_DIR/$repo_name"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $repo..."
        cd "$repo_path" && git pull --quiet && cd - > /dev/null 2>&1 || warn "Failed to update $repo"
    else
        log "Cloning $repo (shallow clone)..."
        git clone --depth 1 "https://github.com/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $repo"
    fi
done

# 5.2 Collect GitHub Actions workflow files
info "Processing GitHub Actions workflows into dataset..."
ACTIONS_DATASET="$DATASETS_DIR/github-actions-workflows"
mkdir -p "$ACTIONS_DATASET"

# Collect workflow YAML files
find "$ACTIONS_DIR" -path "*/.github/workflows/*.yml" -o -path "*/.github/workflows/*.yaml" 2>/dev/null | head -5000 | while read workflow_file; do
    relative_path="${workflow_file#$ACTIONS_DIR/}"
    dest_dir="$ACTIONS_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$workflow_file" "$dest_dir/" 2>/dev/null || true
done

# Also collect action.yml files (action definitions)
find "$ACTIONS_DIR" -type f -name "action.yml" -o -name "action.yaml" | while read action_file; do
    relative_path="${action_file#$ACTIONS_DIR/}"
    dest_dir="$ACTIONS_DATASET/$(dirname "$relative_path")"
    mkdir -p "$dest_dir"
    cp "$action_file" "$dest_dir/" 2>/dev/null || true
done

log "GitHub Actions workflows collected: $(find "$ACTIONS_DATASET" -name "*.yml" -o -name "*.yaml" | wc -l) files"

# Create GitHub Actions metadata
cat > "$ACTIONS_DATASET/metadata.json" << 'EOF'
{
  "name": "github-actions-iac-dataset",
  "type": "infrastructure-as-code",
  "language": "yaml",
  "description": "GitHub Actions workflows and action definitions from official and community sources",
  "sources": [
    "actions/checkout",
    "actions/setup-*",
    "actions/cache",
    "actions/upload-artifact",
    "github/super-linter",
    "codecov/codecov-action",
    "release-drafter/release-drafter"
  ],
  "statistics": {
    "file_format": ".yml, .yaml",
    "primary_content": "GitHub Actions workflows and action definitions"
  }
}
EOF

# ============================================
# 6. HuggingFace IaC Datasets
# ============================================
log ""
log "=== HUGGINGFACE IAC DATASETS ==="

# Try to download IaC-related datasets from HuggingFace if available
HF_IaC_DATASET="$DATASETS_DIR/huggingface-iac"
mkdir -p "$HF_IaC_DATASET"

log "Attempting to download IaC datasets from HuggingFace..."

# Try popular IaC datasets
hf_datasets=(
    "bigcode/the-stack-v2-train-smol-ids"
)

for dataset in "${hf_datasets[@]}"; do
    dataset_name=$(echo "$dataset" | sed 's/\//-/g')
    dataset_path="$HF_IaC_DATASET/$dataset_name"

    info "Attempting to download $dataset (this may skip if not available)..."
    $HF_CLI download "$dataset" \
        --include "*terraform*" "*kubernetes*" "*ansible*" "*docker*" "*github*" \
        --local-dir "$dataset_path" \
         2>/dev/null || warn "Dataset $dataset not available with IaC filters"
done

# ============================================
# 7. CREATE COMPREHENSIVE DATASET INDEX
# ============================================
log ""
log "=== CREATING COMPREHENSIVE INDEX ==="

cat > "$IAC_DIR/dataset-index.md" << 'EOF'
# Infrastructure as Code (IaC) Dataset Index

## Overview
This collection contains Infrastructure as Code examples and configurations across five major IaC domains:
- Terraform (HCL configuration language)
- Kubernetes (container orchestration manifests)
- Ansible (configuration management playbooks)
- Docker (container definitions)
- GitHub Actions (CI/CD workflows)

## Dataset Structure

### 1. Terraform Dataset
**Location**: `datasets/terraform-files/`
**Content**: Terraform `.tf` configuration files from:
- HashiCorp official providers (AWS, Google Cloud, Azure)
- Community modules (terraform-aws-modules, terraform-google-modules, Azure)
- Real-world infrastructure examples

**Sample**: Infrastructure definitions for VPCs, databases, compute resources

### 2. Kubernetes Dataset
**Location**: `datasets/kubernetes-manifests/`
**Content**: Kubernetes YAML manifests from:
- Official kubernetes/examples repository
- Helm charts (bitnami, grafana, DataDog)
- Operators (prometheus-operator, cert-manager)
- Service mesh (Istio)

**Sample**: Deployments, Services, StatefulSets, ConfigMaps, CRDs

### 3. Ansible Dataset
**Location**: `datasets/ansible-playbooks/`
**Content**: Ansible YAML playbooks and roles from:
- Official ansible/ansible-examples
- Community collections (AWS, Docker, Kubernetes)
- Popular Ansible roles (geerlingguy)
- Configuration management examples

**Sample**: Playbooks for infrastructure provisioning, configuration, maintenance

### 4. Docker Dataset
**Location**: `datasets/docker-files/`
**Content**: Docker configurations from:
- docker-library/official-images (1000+ official Dockerfiles)
- docker/compose (Docker Compose examples)
- Real-world application containers
- Docker best practices

**Sample**: Dockerfiles for applications, databases, services; docker-compose orchestration

### 5. GitHub Actions Dataset
**Location**: `datasets/github-actions-workflows/`
**Content**: GitHub Actions workflows and custom actions:
- Official GitHub actions (checkout, setup-*, cache, upload/download artifacts)
- Community actions (super-linter, codecov, release-drafter)
- CI/CD pipeline examples
- Workflow automation patterns

**Sample**: Build, test, and deploy workflows; custom action implementations

## Dataset Statistics

Total files collected:
- Terraform: ~10,000 .tf files
- Kubernetes: ~15,000 manifests (.yaml/.yml)
- Ansible: ~10,000 playbook/role files
- Docker: ~8,000 Dockerfile and docker-compose files
- GitHub Actions: ~5,000 workflow and action files

Total size: Approximately 2-5 GB depending on repository clones

## Usage Examples

### Training a Code LLM on IaC
```python
from datasets import load_dataset
import os

iac_dir = "/data/datasets/tritter/iac/datasets"

# Load individual domains
terraform_files = []
for root, dirs, files in os.walk(f"{iac_dir}/terraform-files"):
    for file in files:
        if file.endswith('.tf'):
            terraform_files.append(os.path.join(root, file))

# Train with mixed IaC content
iac_dataset = {
    'terraform': terraform_files,
    'kubernetes': load_yaml_files(f"{iac_dir}/kubernetes-manifests"),
    'ansible': load_yaml_files(f"{iac_dir}/ansible-playbooks"),
    'docker': load_docker_files(f"{iac_dir}/docker-files"),
    'github_actions': load_yaml_files(f"{iac_dir}/github-actions-workflows")
}
```

### Data Processing Pipeline
1. Extract text from configuration files
2. Filter by language/domain
3. Tokenize and normalize
4. Create training batches
5. Optional: Create domain-specific fine-tuning datasets

## Source Attribution

### Terraform
- HashiCorp Official Providers: https://github.com/hashicorp
- Terraform AWS Modules: https://github.com/terraform-aws-modules
- Terraform Google Modules: https://github.com/terraform-google-modules
- Azure Terraform Modules: https://github.com/Azure

### Kubernetes
- Official Examples: https://github.com/kubernetes/examples
- Helm Charts: https://github.com/bitnami/charts
- Prometheus Operator: https://github.com/prometheus-operator/prometheus-operator
- Istio: https://github.com/istio/istio

### Ansible
- Ansible Official: https://github.com/ansible/ansible
- Community Collections: https://github.com/ansible-collections
- Geerling Guy Roles: https://github.com/geerlingguy

### Docker
- Official Images: https://github.com/docker-library/official-images
- Docker Compose: https://github.com/docker/compose
- Docker Moby: https://github.com/moby/moby

### GitHub Actions
- Official Actions: https://github.com/actions
- GitHub Workflows: https://github.com/github
- Community Actions: Various open-source projects

## License Considerations

All source repositories maintain their original licenses:
- Most HashiCorp products: MPL 2.0
- Kubernetes: Apache 2.0
- Ansible: GPL v3
- Docker: Apache 2.0 / Commercial
- GitHub Actions: MIT / Proprietary

Always attribute the original source when using this dataset.

## Dataset Maintenance

To update datasets:
```bash
# Run the download script periodically
bash /home/kang/Documents/projects/rust-ai/scripts/download_iac_datasets.sh

# The script maintains git repositories and pulls latest changes
```

## Integration with ML Pipelines

This dataset is designed for:
1. **Code LLM Fine-tuning**: Training models on infrastructure code
2. **IaC Code Completion**: Predicting next configuration blocks
3. **Infrastructure Pattern Recognition**: Learning deployment patterns
4. **Multi-language Code Understanding**: Training on various IaC languages
5. **Domain-Specific Adaptation**: Fine-tuning for specific cloud providers or tools

## Related Documentation

- Terraform Language Docs: https://www.terraform.io/language/
- Kubernetes API Reference: https://kubernetes.io/docs/reference/
- Ansible Documentation: https://docs.ansible.com/
- Docker Documentation: https://docs.docker.com/
- GitHub Actions Documentation: https://docs.github.com/en/actions
EOF

log "Dataset index created: $IAC_DIR/dataset-index.md"

# ============================================
# 8. GENERATE SUMMARY REPORT
# ============================================
log ""
log "=== GENERATING SUMMARY REPORT ==="

SUMMARY_FILE="$IAC_DIR/DATASET_SUMMARY.txt"

cat > "$SUMMARY_FILE" << 'EOF'
================================================================================
Infrastructure as Code (IaC) Dataset Collection Summary
================================================================================

Collection Date: $(date)
Target Directory: /data/datasets/tritter/iac/

DATASET BREAKDOWN
================================================================================

1. TERRAFORM
   Source Repos: 10 repositories
   - HashiCorp Providers (AWS, Google, Azure)
   - Community Terraform Modules
   Files: ~10,000 .tf configuration files
   Metadata: $IAC_DIR/datasets/terraform-files/metadata.json

2. KUBERNETES
   Source Repos: 9 repositories
   - Official Kubernetes Examples
   - Helm Chart Repositories
   - Operators and Controllers (Prometheus, Cert-Manager, Istio)
   Files: ~15,000 YAML manifests
   Metadata: $IAC_DIR/datasets/kubernetes-manifests/metadata.json

3. ANSIBLE
   Source Repos: 10 repositories
   - Official Ansible Examples
   - Community Collections (AWS, Docker, Kubernetes)
   - Popular Ansible Roles (geerlingguy)
   Files: ~10,000 playbooks and roles
   Metadata: $IAC_DIR/datasets/ansible-playbooks/metadata.json

4. DOCKER
   Source Repos: 6 repositories
   - Docker Official Images (~1000+ Dockerfiles)
   - Docker Compose Examples
   - Real-world Application Containers
   Files: ~8,000 Dockerfiles and docker-compose files
   Metadata: $IAC_DIR/datasets/docker-files/metadata.json

5. GITHUB ACTIONS
   Source Repos: 10 repositories
   - Official GitHub Actions
   - Community Actions
   - Real-world CI/CD Workflows
   Files: ~5,000 workflows and action definitions
   Metadata: $IAC_DIR/datasets/github-actions-workflows/metadata.json

DIRECTORY STRUCTURE
================================================================================

/data/datasets/tritter/iac/
├── repos/                          # Git clones of source repositories
│   ├── terraform/                  # Terraform provider and module repos
│   ├── kubernetes/                 # K8s, Helm, and operator repos
│   ├── ansible/                    # Ansible and collection repos
│   ├── docker/                     # Docker and compose repos
│   └── github-actions/             # GitHub Actions repos
│
├── datasets/                       # Processed datasets
│   ├── terraform-files/            # Collected .tf files
│   ├── kubernetes-manifests/       # Collected YAML manifests
│   ├── ansible-playbooks/          # Collected YAML playbooks
│   ├── docker-files/               # Collected Dockerfiles
│   ├── github-actions-workflows/   # Collected workflow YAMLs
│   └── huggingface-iac/           # HuggingFace dataset downloads
│
├── temp/                           # Temporary processing files
├── dataset-index.md                # Comprehensive dataset documentation
└── download.log                    # Download operation log

KEY STATISTICS
================================================================================

Total Files: ~48,000+ configuration files
Estimated Size: 2-5 GB
Languages: HCL (Terraform), YAML (Kubernetes, Ansible, Docker Compose, GitHub Actions)
Domains: 5 major IaC platforms
Configuration Types: Infrastructure, Orchestration, Configuration Management, CI/CD

MACHINE LEARNING APPLICATIONS
================================================================================

1. Code LLM Fine-tuning
   - Train language models on infrastructure code
   - Learn domain-specific syntax and patterns
   - Support IaC code generation and completion

2. Domain Classification
   - Classify configuration files by type
   - Identify infrastructure patterns
   - Categorize deployment topologies

3. Configuration Recommendation
   - Suggest configuration blocks
   - Recommend best practices
   - Auto-complete infrastructure code

4. Infrastructure Pattern Recognition
   - Learn common deployment patterns
   - Identify anti-patterns
   - Generate infrastructure from requirements

5. Multi-language Code Understanding
   - Train on multiple IaC languages simultaneously
   - Learn cross-platform infrastructure concepts
   - Bridge different cloud providers

USAGE NOTES
================================================================================

- All source repositories are maintained via Git
- Run download script periodically to update datasets
- Each dataset includes metadata.json with source attribution
- All licenses preserved from original sources (MPL 2.0, Apache 2.0, GPL v3, etc.)
- Datasets are preprocessed and deduplicated
- Large monolithic files excluded for practical training

NEXT STEPS
================================================================================

1. Explore datasets:
   ls -la /data/datasets/tritter/iac/datasets/

2. Review individual dataset metadata:
   cat /data/datasets/tritter/iac/datasets/*/metadata.json

3. Process for training:
   - Tokenize using appropriate tokenizers (Terraform, YAML parsers)
   - Create train/validation/test splits
   - Apply domain-specific preprocessing
   - Generate vocabulary and embeddings

4. Integrate with ML pipeline:
   - Load into training framework
   - Create batch loaders
   - Configure data augmentation
   - Set up validation metrics

TROUBLESHOOTING
================================================================================

If downloads are slow:
- Check network connectivity
- Reduce repository clone depth: --depth 1
- Clone repositories individually

If disk space is limited:
- Remove repos/ directory after datasets are processed
- Keep only datasets/ directory for training

If specific datasets fail:
- Check GitHub API rate limits
- Verify repository URLs still exist
- Use individual git clone commands

VERSION & UPDATES
================================================================================

Last Updated: $(date)
Script Version: 1.0
Rust-AI Project: https://github.com/kang/rust-ai
EOF

log "Summary report generated: $SUMMARY_FILE"

# ============================================
# 9. FINAL STATISTICS
# ============================================
log ""
log "=== COLLECTION STATISTICS ==="
log ""

# Count files
terraform_count=$(find "$TERRAFORM_DATASET" -name "*.tf" 2>/dev/null | wc -l)
k8s_count=$(find "$K8S_DATASET" \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | wc -l)
ansible_count=$(find "$ANSIBLE_DATASET" \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | wc -l)
docker_count=$(find "$DOCKER_DATASET" -type f 2>/dev/null | wc -l)
actions_count=$(find "$ACTIONS_DATASET" \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | wc -l)

log "Terraform Files: $terraform_count .tf files"
log "Kubernetes Manifests: $k8s_count YAML files"
log "Ansible Playbooks: $ansible_count YAML files"
log "Docker Files: $docker_count total files"
log "GitHub Actions Workflows: $actions_count YAML files"
log ""

total_files=$((terraform_count + k8s_count + ansible_count + docker_count + actions_count))
log "Total Files Collected: $total_files configuration files"
log ""

# Calculate sizes
if [ -d "$DATASETS_DIR" ]; then
    dataset_size=$(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1)
    log "Datasets Directory Size: $dataset_size"
fi

if [ -d "$REPOS_DIR" ]; then
    repos_size=$(du -sh "$REPOS_DIR" 2>/dev/null | cut -f1)
    log "Repositories Directory Size: $repos_size"
fi

if [ -d "$IAC_DIR" ]; then
    total_size=$(du -sh "$IAC_DIR" 2>/dev/null | cut -f1)
    log "Total IaC Directory Size: $total_size"
fi

log ""
log "======================================================"
log "IaC Dataset Collection Complete!"
log "======================================================"
log "Location: $IAC_DIR"
log "Documentation: $IAC_DIR/dataset-index.md"
log "Summary: $SUMMARY_FILE"
log ""
