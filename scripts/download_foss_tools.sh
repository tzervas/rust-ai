#!/bin/bash
# Download Industry-Standard FOSS Tools, Docs, and Examples
# ArgoCD, Flux, Crossplane, Vault, Traefik, Tekton, etc.

set -e

FOSS_DIR="/data/datasets/tritter/iac/repos/foss-tools"
LOG_FILE="/data/datasets/tritter/iac/foss_download.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $1" | tee -a "$LOG_FILE"; }
info() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"; }

mkdir -p "$FOSS_DIR"

log "=============================================="
log "Downloading Industry FOSS Tools & Examples"
log "=============================================="

# Function to clone or update repo
clone_repo() {
    local org=$1
    local repo=$2
    local category=$3
    local repo_path="$FOSS_DIR/$category/$repo"

    mkdir -p "$FOSS_DIR/$category"

    if [ -d "$repo_path/.git" ]; then
        log "Updating $org/$repo..."
        cd "$repo_path" && git pull --quiet 2>/dev/null && cd - > /dev/null || warn "Failed to update $repo"
    else
        log "Cloning $org/$repo..."
        git clone --depth 1 "https://github.com/$org/$repo.git" "$repo_path" 2>/dev/null || warn "Failed to clone $org/$repo"
    fi
}

# ============================================
# GitOps Tools
# ============================================
log ""
log "=== GitOps Tools ==="

# ArgoCD - GitOps continuous delivery
clone_repo "argoproj" "argo-cd" "gitops"
clone_repo "argoproj" "argo-workflows" "gitops"
clone_repo "argoproj" "argo-rollouts" "gitops"
clone_repo "argoproj" "argo-events" "gitops"
clone_repo "argoproj" "applicationset" "gitops"
clone_repo "argoproj-labs" "argocd-autopilot" "gitops"

# Flux - GitOps toolkit
clone_repo "fluxcd" "flux2" "gitops"
clone_repo "fluxcd" "flux2-kustomize-helm-example" "gitops"
clone_repo "fluxcd" "flux2-multi-tenancy" "gitops"

# ============================================
# Infrastructure Tools
# ============================================
log ""
log "=== Infrastructure Tools ==="

# Crossplane - Universal control plane
clone_repo "crossplane" "crossplane" "infrastructure"
clone_repo "crossplane-contrib" "provider-aws" "infrastructure"
clone_repo "crossplane-contrib" "provider-gcp" "infrastructure"
clone_repo "crossplane-contrib" "provider-azure" "infrastructure"

# Pulumi examples
clone_repo "pulumi" "examples" "infrastructure"
clone_repo "pulumi" "pulumi-kubernetes" "infrastructure"

# AWS CDK examples
clone_repo "aws-samples" "aws-cdk-examples" "infrastructure"

# ============================================
# Service Mesh & Networking
# ============================================
log ""
log "=== Service Mesh & Networking ==="

# Traefik
clone_repo "traefik" "traefik" "networking"
clone_repo "traefik" "traefik-helm-chart" "networking"

# Nginx Ingress
clone_repo "kubernetes" "ingress-nginx" "networking"

# Linkerd
clone_repo "linkerd" "linkerd2" "networking"

# Cilium
clone_repo "cilium" "cilium" "networking"

# ============================================
# Security & Secrets
# ============================================
log ""
log "=== Security & Secrets ==="

# HashiCorp Vault
clone_repo "hashicorp" "vault" "security"
clone_repo "hashicorp" "vault-helm" "security"
clone_repo "hashicorp" "vault-k8s" "security"

# External Secrets Operator
clone_repo "external-secrets" "external-secrets" "security"

# Sealed Secrets
clone_repo "bitnami-labs" "sealed-secrets" "security"

# Cert Manager (already have but ensure)
clone_repo "cert-manager" "cert-manager" "security"

# ============================================
# CI/CD Pipelines
# ============================================
log ""
log "=== CI/CD Pipelines ==="

# Tekton
clone_repo "tektoncd" "pipeline" "cicd"
clone_repo "tektoncd" "triggers" "cicd"
clone_repo "tektoncd" "catalog" "cicd"

# Jenkins
clone_repo "jenkinsci" "kubernetes-plugin" "cicd"
clone_repo "jenkinsci" "configuration-as-code-plugin" "cicd"

# Drone CI
clone_repo "harness" "drone" "cicd"

# Woodpecker CI (Drone fork)
clone_repo "woodpecker-ci" "woodpecker" "cicd"

# ============================================
# Observability
# ============================================
log ""
log "=== Observability ==="

# OpenTelemetry
clone_repo "open-telemetry" "opentelemetry-collector" "observability"
clone_repo "open-telemetry" "opentelemetry-helm-charts" "observability"

# Prometheus
clone_repo "prometheus" "prometheus" "observability"
clone_repo "prometheus-community" "helm-charts" "observability"

# Grafana
clone_repo "grafana" "grafana" "observability"
clone_repo "grafana" "loki" "observability"
clone_repo "grafana" "tempo" "observability"
clone_repo "grafana" "mimir" "observability"

# Jaeger
clone_repo "jaegertracing" "jaeger" "observability"

# ============================================
# Developer Tools
# ============================================
log ""
log "=== Developer Tools ==="

# Backstage (Developer Portal)
clone_repo "backstage" "backstage" "devtools"

# Kustomize
clone_repo "kubernetes-sigs" "kustomize" "devtools"

# Skaffold
clone_repo "GoogleContainerTools" "skaffold" "devtools"

# Tilt
clone_repo "tilt-dev" "tilt" "devtools"

# DevSpace
clone_repo "devspace-sh" "devspace" "devtools"

# ============================================
# Database Operators
# ============================================
log ""
log "=== Database Operators ==="

# PostgreSQL Operator
clone_repo "zalando" "postgres-operator" "databases"
clone_repo "CrunchyData" "postgres-operator" "databases"

# MySQL Operator
clone_repo "mysql" "mysql-operator" "databases"

# MongoDB Operator
clone_repo "mongodb" "mongodb-kubernetes-operator" "databases"

# Redis Operator
clone_repo "spotahome" "redis-operator" "databases"

# ============================================
# Message Queues
# ============================================
log ""
log "=== Message Queues ==="

# Kafka (Strimzi)
clone_repo "strimzi" "strimzi-kafka-operator" "messaging"

# RabbitMQ
clone_repo "rabbitmq" "cluster-operator" "messaging"

# NATS
clone_repo "nats-io" "nats-server" "messaging"
clone_repo "nats-io" "nats-operator" "messaging"

# ============================================
# Policy & Governance
# ============================================
log ""
log "=== Policy & Governance ==="

# Open Policy Agent
clone_repo "open-policy-agent" "opa" "policy"
clone_repo "open-policy-agent" "gatekeeper" "policy"

# Kyverno
clone_repo "kyverno" "kyverno" "policy"
clone_repo "kyverno" "policies" "policy"

# Falco
clone_repo "falcosecurity" "falco" "policy"

# ============================================
# Storage
# ============================================
log ""
log "=== Storage ==="

# Rook (Ceph)
clone_repo "rook" "rook" "storage"

# Longhorn
clone_repo "longhorn" "longhorn" "storage"

# OpenEBS
clone_repo "openebs" "openebs" "storage"

# MinIO
clone_repo "minio" "minio" "storage"
clone_repo "minio" "operator" "storage"

# ============================================
# Collect Statistics
# ============================================
log ""
log "=== Statistics ==="

total_repos=$(find "$FOSS_DIR" -maxdepth 2 -type d -name ".git" | wc -l)
total_size=$(du -sh "$FOSS_DIR" 2>/dev/null | cut -f1)

log "Total repositories cloned: $total_repos"
log "Total size: $total_size"

# Count files by type
yaml_count=$(find "$FOSS_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | wc -l)
json_count=$(find "$FOSS_DIR" -type f -name "*.json" 2>/dev/null | wc -l)
md_count=$(find "$FOSS_DIR" -type f -name "*.md" 2>/dev/null | wc -l)

log "YAML files: $yaml_count"
log "JSON files: $json_count"
log "Markdown (docs): $md_count"

log ""
log "=============================================="
log "FOSS Tools Download Complete!"
log "=============================================="
log "Location: $FOSS_DIR"
