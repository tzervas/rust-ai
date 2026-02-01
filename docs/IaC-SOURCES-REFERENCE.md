# Infrastructure as Code (IaC) Dataset Sources Reference

## Complete Source List for Dataset Collection

This document provides a comprehensive reference of all data sources used in the IaC dataset collection system.

---

## Terraform Sources

### Official HashiCorp Providers

1. **terraform-provider-aws**
   - URL: https://github.com/hashicorp/terraform-provider-aws
   - License: MPL 2.0
   - Content: AWS resource provider with extensive examples
   - Files: 100+ example configurations

2. **terraform-provider-google**
   - URL: https://github.com/hashicorp/terraform-provider-google
   - License: MPL 2.0
   - Content: Google Cloud Platform provider examples
   - Files: 50+ example configurations

3. **terraform-provider-azurerm**
   - URL: https://github.com/hashicorp/terraform-provider-azurerm
   - License: MPL 2.0
   - Content: Microsoft Azure resource provider examples
   - Files: 75+ example configurations

### Community Terraform Modules

4. **terraform-aws-modules**
   - URL: https://github.com/terraform-aws-modules
   - License: Apache 2.0
   - Content: Pre-built AWS infrastructure modules
   - Popular Modules:
     - terraform-aws-vpc (VPC networking)
     - terraform-aws-ecs (Elastic Container Service)
     - terraform-aws-rds (Relational Database Service)
     - terraform-aws-alb (Application Load Balancer)
     - terraform-aws-security-group (Security Groups)

5. **terraform-google-modules**
   - URL: https://github.com/terraform-google-modules
   - License: Apache 2.0
   - Content: Pre-built Google Cloud modules
   - Popular Modules:
     - terraform-google-kubernetes-engine (GKE)
     - terraform-google-compute-engine (Compute instances)
     - terraform-google-sql (Cloud SQL)
     - terraform-google-cloud-storage (Storage)

6. **Azure Terraform Modules (CAF)**
   - URL: https://github.com/Azure
   - License: MIT
   - Content: Azure Cloud Adoption Framework modules
   - Notable:
     - terraform-azurerm-naming (Naming conventions)
     - terraform-azurerm-caf (CAF patterns)

### Terraform Statistics
- **Total Repositories:** 6
- **Estimated Files:** ~10,000 .tf files
- **Estimated Size:** 200-500 MB
- **Primary Language:** HCL (HashiCorp Configuration Language)

---

## Kubernetes Sources

### Official Kubernetes

1. **kubernetes/examples**
   - URL: https://github.com/kubernetes/examples
   - License: Apache 2.0
   - Content: Official Kubernetes resource examples
   - Examples Include:
     - Deployments
     - StatefulSets
     - DaemonSets
     - Jobs and CronJobs
     - Services and Ingress
     - ConfigMaps and Secrets
     - RBAC configurations

2. **kubernetes/kubernetes**
   - URL: https://github.com/kubernetes/kubernetes
   - License: Apache 2.0
   - Content: Kubernetes core source with sample manifests
   - Files: Testing and documentation manifests

### Service Mesh & Observability

3. **istio/istio**
   - URL: https://github.com/istio/istio
   - License: Apache 2.0
   - Content: Istio service mesh installation and configuration examples
   - Includes: Virtual Services, Destination Rules, Gateways

4. **prometheus-operator/prometheus-operator**
   - URL: https://github.com/prometheus-operator/prometheus-operator
   - License: Apache 2.0
   - Content: Prometheus monitoring operator with CRDs and examples
   - Includes: Prometheus, Alertmanager, ServiceMonitor configs

5. **jetstack/cert-manager**
   - URL: https://github.com/jetstack/cert-manager
   - License: Apache 2.0
   - Content: Certificate management operator and examples
   - Includes: Certificate, Issuer, ClusterIssuer resources

### Helm Charts

6. **bitnami/charts**
   - URL: https://github.com/bitnami/charts
   - License: Apache 2.0
   - Content: 500+ Helm charts for popular applications
   - Applications: Database, cache, monitoring, logging, etc.

7. **grafana/helm-charts**
   - URL: https://github.com/grafana/helm-charts
   - License: AGPL 3.0
   - Content: Grafana stack Helm charts
   - Includes: Grafana, Loki, Tempo, Prometheus

8. **DataDog/helm-charts**
   - URL: https://github.com/DataDog/helm-charts
   - License: Apache 2.0
   - Content: Datadog agent and integration Helm charts

### Kubernetes Statistics
- **Total Repositories:** 9
- **Estimated Files:** ~15,000 YAML manifests
- **Estimated Size:** 300-700 MB
- **Primary Language:** YAML
- **Resource Types:** 40+ Kubernetes object types

---

## Ansible Sources

### Official Ansible

1. **ansible/ansible**
   - URL: https://github.com/ansible/ansible
   - License: GPL v3
   - Content: Ansible core framework with built-in modules
   - Includes: 500+ built-in modules and plugins

2. **ansible/ansible-examples**
   - URL: https://github.com/ansible/ansible-examples
   - License: GPL v3
   - Content: Official playbook and role examples
   - Examples:
     - Web server setup
     - Database configuration
     - Multi-tier deployments
     - Security hardening

3. **ansible/ansible-lint**
   - URL: https://github.com/ansible/ansible-lint
   - License: GPL v3
   - Content: Linting tool for Ansible playbooks
   - Includes: Rule definitions and playbook examples

### Community Collections

4. **ansible-collections/community.general**
   - URL: https://github.com/ansible-collections/community.general
   - License: GPL v3
   - Content: General-purpose community modules
   - Modules: 200+ third-party integrations

5. **ansible-collections/community.aws**
   - URL: https://github.com/ansible-collections/community.aws
   - License: GPL v3
   - Content: AWS-specific Ansible modules
   - Modules: 100+ AWS automation modules

6. **ansible-collections/community.kubernetes**
   - URL: https://github.com/ansible-collections/community.kubernetes
   - License: GPL v3
   - Content: Kubernetes automation modules
   - Modules: Kubectl, Helm integration, etc.

7. **ansible-collections/community.docker**
   - URL: https://github.com/ansible-collections/community.docker
   - License: GPL v3
   - Content: Docker container management modules
   - Modules: Docker, Podman, Docker Swarm

### Popular Ansible Roles

8. **geerlingguy/ansible-role-geerlingguy.docker**
   - URL: https://github.com/geerlingguy/ansible-role-docker
   - License: MIT
   - Content: Production-ready Docker installation role
   - Use Cases: Docker Engine setup on Linux systems

9. **geerlingguy/ansible-role-java**
   - URL: https://github.com/geerlingguy/ansible-role-java
   - License: MIT
   - Content: Java runtime installation and configuration
   - Includes: OpenJDK, Eclipse Temurin, Oracle JDK options

10. **geerlingguy/ansible-role-postgresql**
    - URL: https://github.com/geerlingguy/ansible-role-postgresql
    - License: MIT
    - Content: PostgreSQL database setup and configuration
    - Includes: Installation, user management, backups

### Ansible Statistics
- **Total Repositories:** 10
- **Estimated Files:** ~10,000 playbooks and roles
- **Estimated Size:** 150-400 MB
- **Primary Language:** YAML
- **Component Types:** Playbooks, Roles, Tasks, Handlers, Plugins, Modules

---

## Docker Sources

### Official Docker

1. **docker-library/official-images**
   - URL: https://github.com/docker-library/official-images
   - License: Apache 2.0
   - Content: 1000+ official Docker image Dockerfiles
   - Images Include:
     - Programming languages (Python, Node.js, Java, Go, Rust, etc.)
     - Databases (PostgreSQL, MySQL, MongoDB, Redis, etc.)
     - Web servers (Nginx, Apache, etc.)
     - Messaging (RabbitMQ, Kafka, etc.)
     - Monitoring (Prometheus, Grafana, etc.)
     - And 900+ more

2. **docker/compose**
   - URL: https://github.com/docker/compose
   - License: Apache 2.0
   - Content: Docker Compose tool with examples
   - Examples: Multi-container application definitions

3. **docker/cli**
   - URL: https://github.com/docker/cli
   - License: Apache 2.0
   - Content: Docker CLI source and examples
   - Includes: Command documentation and examples

4. **moby/moby**
   - URL: https://github.com/moby/moby
   - License: Apache 2.0
   - Content: Docker engine (moby) source code
   - Includes: Test fixtures and integration examples

5. **docker-library/docs**
   - URL: https://github.com/docker-library/docs
   - License: CC0 1.0 (Public Domain)
   - Content: Documentation for official images
   - Includes: Usage examples and best practices

### Docker Statistics
- **Total Repositories:** 6
- **Estimated Files:** ~8,000 (Dockerfiles and compose files)
- **Estimated Size:** 100-300 MB
- **File Types:**
  - Dockerfile
  - docker-compose.yml/yaml
  - .dockerignore
- **Image Coverage:** 1000+ official images

---

## GitHub Actions Sources

### Official GitHub Actions

1. **actions/checkout**
   - URL: https://github.com/actions/checkout
   - License: MIT
   - Content: Git repository checkout action
   - Usage: Check out code in workflows

2. **actions/setup-python**
   - URL: https://github.com/actions/setup-python
   - License: MIT
   - Content: Python environment setup action
   - Usage: Set up Python versions with caching

3. **actions/setup-node**
   - URL: https://github.com/actions/setup-node
   - License: MIT
   - Content: Node.js environment setup
   - Usage: Set up Node.js with npm/yarn caching

4. **actions/setup-docker**
   - URL: https://github.com/actions/setup-docker
   - License: MIT
   - Content: Docker setup action
   - Usage: Configure Docker in workflows

5. **actions/cache**
   - URL: https://github.com/actions/cache
   - License: MIT
   - Content: Cache management action
   - Usage: Cache dependencies between runs

6. **actions/upload-artifact**
   - URL: https://github.com/actions/upload-artifact
   - License: MIT
   - Content: Artifact upload action
   - Usage: Upload build artifacts

7. **actions/download-artifact**
   - URL: https://github.com/actions/download-artifact
   - License: MIT
   - Content: Artifact download action
   - Usage: Download artifacts from previous jobs

### Community Actions

8. **github/super-linter**
   - URL: https://github.com/github/super-linter
   - License: MIT
   - Content: Comprehensive code linting action
   - Supports: 40+ languages and linting tools

9. **release-drafter/release-drafter**
   - URL: https://github.com/release-drafter/release-drafter
   - License: MIT
   - Content: Release automation action
   - Usage: Draft releases from pull request labels

10. **codecov/codecov-action**
    - URL: https://github.com/codecov/codecov-action
    - License: MIT
    - Content: Code coverage reporting action
    - Usage: Upload coverage reports to Codecov

### GitHub Actions Statistics
- **Total Repositories:** 10
- **Estimated Files:** ~5,000 workflow and action files
- **Estimated Size:** 50-150 MB
- **Primary Language:** YAML
- **Coverage:** Official GitHub actions + popular community actions

---

## HuggingFace Datasets (Optional)

The script attempts to download relevant datasets from HuggingFace if available:

### Attempted HuggingFace Datasets

1. **bigcode/the-stack-v2-train-smol-ids**
   - URL: https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids
   - License: OpenRAIL
   - Content: Large code dataset with IaC samples
   - Filters Applied:
     - terraform*
     - kubernetes*
     - ansible*
     - docker*
     - github*

---

## Summary Statistics

### Total Repository Count
| Domain | Count |
|--------|-------|
| Terraform | 6 |
| Kubernetes | 9 |
| Ansible | 10 |
| Docker | 6 |
| GitHub Actions | 10 |
| **TOTAL** | **41** |

### Estimated Data Collection
| Domain | Est. Files | Est. Size | Language(s) |
|--------|-----------|-----------|------------|
| Terraform | ~10,000 | 200-500 MB | HCL |
| Kubernetes | ~15,000 | 300-700 MB | YAML |
| Ansible | ~10,000 | 150-400 MB | YAML, Python |
| Docker | ~8,000 | 100-300 MB | Dockerfile, YAML |
| GitHub Actions | ~5,000 | 50-150 MB | YAML |
| **TOTAL** | **~48,000** | **2-5 GB** | **Mixed** |

---

## License Attribution

### License Summary
| License | Repositories | Notes |
|---------|-------------|-------|
| Apache 2.0 | 20+ | Compatible with commercial use |
| MIT | 15+ | Permissive, commercial-friendly |
| MPL 2.0 | 3 | Weaker copyleft |
| GPL v3 | 7 | Strong copyleft, requires derivative works be open |
| AGPL 3.0 | 1 | Strong copyleft with network clause |
| Other | 5 | Various permissive licenses |

### Important Notes
1. **GPL v3 Compliance:** When using Ansible datasets for commercial products, ensure compliance with GPL v3 terms
2. **Attribution Required:** Always attribute sources when publishing derived work
3. **License Compatibility:** Verify license compatibility with your use case
4. **Commercial Use:** Most sources allow commercial use with proper attribution

---

## Data Quality Metrics

### Source Repository Statistics

| Source | Stars | Contributors | Last Update |
|--------|-------|--------------|------------|
| terraform-provider-aws | 10k+ | 500+ | Weekly |
| kubernetes/examples | 10k+ | 1000+ | Weekly |
| ansible/ansible | 60k+ | 1000+ | Weekly |
| docker-library/official-images | 8k+ | 200+ | Weekly |
| istio/istio | 35k+ | 800+ | Weekly |
| bitnami/charts | 10k+ | 100+ | Weekly |

### Community Engagement
- **Average Stars:** 10k-35k across major projects
- **Update Frequency:** Most projects updated weekly
- **Contributor Base:** 100-1000+ contributors per major project
- **Maturity Level:** Production-grade, actively maintained

---

## Integration with Training

### Recommended Usage Sequence

1. **Download Priority:**
   1. Terraform (most stable API)
   2. Kubernetes (most examples)
   3. Docker (highest quality)
   4. Ansible (comprehensive coverage)
   5. GitHub Actions (modern patterns)

2. **Processing Steps:**
   - Extract and validate syntax
   - Remove duplicates and corrupted files
   - Normalize formatting (optional)
   - Create domain-specific splits
   - Generate statistics

3. **Training Preparation:**
   - Tokenize per domain
   - Create mixed-domain batches
   - Split into train/val/test
   - Balance domain representation

---

## Maintenance & Updates

### Update Schedule

Recommended update frequency:
- **Daily:** Check GitHub Actions (rapid changes)
- **Weekly:** Update all repositories
- **Monthly:** Full dataset reprocessing
- **Quarterly:** Quality audit and deduplication

### Repository Health Checks

```bash
# Verify repository accessibility
for repo in terraform-provider-aws docker-library/official-images ...; do
    curl -sI "https://github.com/$repo" | grep HTTP
done

# Check clone sizes
for repo in repos/*; do
    du -sh "$repo"
done
```

---

## External References

### Official Documentation
- [Terraform Registry](https://registry.terraform.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Ansible Documentation](https://docs.ansible.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

### Community Resources
- [Terraform Community](https://discuss.hashicorp.com/c/terraform)
- [Kubernetes Community](https://kubernetes.io/community/)
- [Ansible Community](https://www.ansible.com/community)
- [Docker Community](https://www.docker.com/community)

---

## Contact & Support

For dataset-related issues or questions:
1. Check individual repository issues on GitHub
2. Review dataset documentation: `/home/kang/Documents/projects/rust-ai/docs/IaC-DATASET-GUIDE.md`
3. Check processing logs: `/data/datasets/tritter/iac/download.log`

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintenance Status:** Active
**Next Review:** 2026-04-30
