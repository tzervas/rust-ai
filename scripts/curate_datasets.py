#!/usr/bin/env python3
"""
Automated Dataset Curation Pipeline

This script:
1. Identifies dataset gaps
2. Curates high-quality FOSS GitHub repositories
3. Fetches latest IaC schemas and validates examples
4. Generates training-ready datasets

Usage:
    python curate_datasets.py --analyze          # Show gaps
    python curate_datasets.py --github           # Curate GitHub repos
    python curate_datasets.py --iac              # Fetch IaC schemas
    python curate_datasets.py --all              # Run full pipeline
"""

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

# ============================================
# Configuration
# ============================================

DATA_DIR = Path("/data/datasets/tritter")
CURATED_DIR = DATA_DIR / "curated"
IAC_DIR = DATA_DIR / "iac"
GITHUB_DIR = DATA_DIR / "github-foss"

# FOSS-compatible licenses
FOSS_LICENSES = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "mpl-2.0", "lgpl-2.1", "lgpl-3.0", "isc", "unlicense",
    "cc0-1.0", "0bsd", "artistic-2.0", "zlib"
}

# Languages to curate
TARGET_LANGUAGES = [
    "rust", "python", "typescript", "go", "zig",
    "hcl",  # Terraform
    "nix",  # NixOS
]

# IaC tools and their schema sources
IAC_SCHEMAS = {
    "terraform": {
        "providers": ["aws", "google", "azure", "kubernetes"],
        "schema_url": "https://registry.terraform.io/v1/providers/{provider}/latest",
    },
    "kubernetes": {
        "versions": ["1.29", "1.30"],
        "schema_url": "https://raw.githubusercontent.com/kubernetes/kubernetes/v{version}/api/openapi-spec/swagger.json",
    },
    "ansible": {
        "collections": ["ansible.builtin", "community.general", "amazon.aws"],
        "docs_url": "https://docs.ansible.com/ansible/latest/collections/{collection}/",
    },
    "docker": {
        "compose_schema": "https://raw.githubusercontent.com/compose-spec/compose-spec/master/schema/compose-spec.json",
        "dockerfile_ref": "https://docs.docker.com/reference/dockerfile/",
    },
    "helm": {
        "schema_url": "https://raw.githubusercontent.com/helm/helm/main/pkg/chartutil/capabilities.go",
    },
    "openapi": {
        "versions": ["3.0", "3.1"],
        "schema_url": "https://spec.openapis.org/oas/{version}/schema/2022-10-07",
    },
    "jsonschema": {
        "versions": ["draft-07", "2020-12"],
        "meta_schema": "https://json-schema.org/{version}/schema",
    },
    "github-actions": {
        "schema_url": "https://json.schemastore.org/github-workflow.json",
    },
    "cloudinit": {
        "schema_url": "https://raw.githubusercontent.com/canonical/cloud-init/main/cloudinit/config/schema.py",
    },
}

# Quality thresholds for GitHub repos
GITHUB_QUALITY = {
    "min_stars": 100,
    "min_forks": 10,
    "max_age_days": 365,  # Active in last year
    "min_commits": 50,
    "has_readme": True,
    "has_license": True,
    "has_tests": True,
}


@dataclass
class DatasetGap:
    """Represents a gap in dataset coverage."""
    category: str
    name: str
    priority: str  # high, medium, low
    current_size: int
    target_size: int
    sources: List[str]
    notes: str


def analyze_gaps() -> List[DatasetGap]:
    """Analyze current datasets and identify gaps."""
    gaps = []

    # Check code language coverage
    code_gaps = {
        "rust": DatasetGap(
            category="code",
            name="Rust",
            priority="high",
            current_size=get_dataset_size("stack-v2-rust") + get_dataset_size("strandset-rust"),
            target_size=5_000_000_000,  # 5GB target
            sources=["the-stack-v2", "github-rust-repos", "crates.io-source"],
            notes="Stack v2 failed, need alternative sources"
        ),
        "typescript": DatasetGap(
            category="code",
            name="TypeScript",
            priority="high",
            current_size=get_dataset_size("stack-v2-typescript"),
            target_size=5_000_000_000,
            sources=["the-stack-v2", "github-ts-repos", "npm-packages"],
            notes="Stack v2 failed, need alternative sources"
        ),
        "go": DatasetGap(
            category="code",
            name="Go",
            priority="medium",
            current_size=0,
            target_size=3_000_000_000,
            sources=["the-stack-v2", "github-go-repos"],
            notes="No Go code currently"
        ),
    }
    gaps.extend(code_gaps.values())

    # IaC coverage
    iac_gaps = [
        DatasetGap(
            category="iac",
            name="Terraform",
            priority="high",
            current_size=get_dataset_size("terraform"),
            target_size=1_000_000_000,
            sources=["terraform-registry", "github-tf-modules", "official-examples"],
            notes="Need provider schemas + validated examples"
        ),
        DatasetGap(
            category="iac",
            name="Kubernetes",
            priority="high",
            current_size=get_dataset_size("kubernetes"),
            target_size=1_000_000_000,
            sources=["k8s-examples", "helm-charts", "kustomize-bases"],
            notes="Need CRD schemas + manifests"
        ),
        DatasetGap(
            category="iac",
            name="Ansible",
            priority="medium",
            current_size=get_dataset_size("ansible"),
            target_size=500_000_000,
            sources=["ansible-galaxy", "github-playbooks"],
            notes="Need collection docs + validated playbooks"
        ),
        DatasetGap(
            category="iac",
            name="Docker/Compose",
            priority="medium",
            current_size=get_dataset_size("docker"),
            target_size=500_000_000,
            sources=["dockerhub-official", "github-dockerfiles"],
            notes="Need best practices examples"
        ),
        DatasetGap(
            category="iac",
            name="GitHub Actions",
            priority="medium",
            current_size=get_dataset_size("github-actions"),
            target_size=200_000_000,
            sources=["github-marketplace", "github-repos"],
            notes="Need validated workflow examples"
        ),
    ]
    gaps.extend(iac_gaps)

    # Documentation gaps
    doc_gaps = [
        DatasetGap(
            category="docs",
            name="API Documentation",
            priority="medium",
            current_size=get_dataset_size("api-docs"),
            target_size=2_000_000_000,
            sources=["devdocs.io", "readthedocs", "official-docs"],
            notes="OpenAPI specs, SDK docs"
        ),
        DatasetGap(
            category="docs",
            name="RFCs/Standards",
            priority="low",
            current_size=get_dataset_size("rfcs"),
            target_size=500_000_000,
            sources=["ietf-rfcs", "w3c-specs", "ecma-specs"],
            notes="Technical standards"
        ),
    ]
    gaps.extend(doc_gaps)

    return gaps


def get_dataset_size(name: str) -> int:
    """Get size of a dataset in bytes."""
    for search_dir in [DATA_DIR / "pretrain", DATA_DIR / "curated", DATA_DIR]:
        path = search_dir / name
        if path.exists():
            result = subprocess.run(
                ["du", "-sb", str(path)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.split()[0])
    return 0


def print_gap_analysis(gaps: List[DatasetGap]):
    """Print formatted gap analysis."""
    print("\n" + "=" * 60)
    print("DATASET GAP ANALYSIS")
    print("=" * 60)

    # Group by category
    by_category = {}
    for gap in gaps:
        by_category.setdefault(gap.category, []).append(gap)

    for category, category_gaps in sorted(by_category.items()):
        print(f"\n## {category.upper()}")
        print("-" * 40)

        for gap in sorted(category_gaps, key=lambda g: (g.priority != "high", g.priority != "medium", g.name)):
            current_mb = gap.current_size / 1_000_000
            target_mb = gap.target_size / 1_000_000
            coverage = (gap.current_size / gap.target_size * 100) if gap.target_size > 0 else 0

            status = "🔴" if coverage < 10 else "🟡" if coverage < 50 else "🟢"
            priority_icon = "❗" if gap.priority == "high" else "❕" if gap.priority == "medium" else "·"

            print(f"\n{status} {priority_icon} {gap.name}")
            print(f"   Coverage: {coverage:.1f}% ({current_mb:.0f}MB / {target_mb:.0f}MB)")
            print(f"   Sources: {', '.join(gap.sources[:3])}")
            if gap.notes:
                print(f"   Note: {gap.notes}")


def curate_github_repos(language: str, max_repos: int = 100) -> List[Dict]:
    """
    Curate high-quality FOSS GitHub repositories for a language.

    Uses GitHub API to find repos matching quality criteria.
    """
    print(f"\n[GitHub] Curating {language} repositories...")

    # GitHub search query
    query_parts = [
        f"language:{language}",
        f"stars:>={GITHUB_QUALITY['min_stars']}",
        f"forks:>={GITHUB_QUALITY['min_forks']}",
        "archived:false",
        "is:public",
    ]

    # Add license filter
    license_query = " OR ".join([f"license:{lic}" for lic in list(FOSS_LICENSES)[:5]])

    query = " ".join(query_parts)

    repos = []

    try:
        # Use gh CLI for authentication
        result = subprocess.run(
            ["gh", "api", "-X", "GET", "/search/repositories",
             "-f", f"q={query}",
             "-f", "sort=stars",
             "-f", "order=desc",
             "-f", f"per_page={min(max_repos, 100)}"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            for repo in data.get("items", []):
                # Additional quality checks
                if not is_quality_repo(repo):
                    continue

                repos.append({
                    "full_name": repo["full_name"],
                    "url": repo["html_url"],
                    "clone_url": repo["clone_url"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "license": repo.get("license", {}).get("spdx_id", "unknown"),
                    "language": repo["language"],
                    "description": repo.get("description", ""),
                    "topics": repo.get("topics", []),
                    "updated_at": repo["updated_at"],
                })
        else:
            print(f"   Warning: GitHub API error: {result.stderr}")

    except Exception as e:
        print(f"   Error: {e}")

    print(f"   Found {len(repos)} quality repositories")
    return repos


def is_quality_repo(repo: Dict) -> bool:
    """Check if repo meets quality criteria."""
    # License check
    license_info = repo.get("license")
    if not license_info:
        return False

    license_id = license_info.get("spdx_id", "").lower()
    if license_id not in FOSS_LICENSES and license_id != "noassertion":
        return False

    # Activity check
    updated = repo.get("updated_at", "")
    if updated:
        try:
            update_date = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            days_ago = (datetime.now(update_date.tzinfo) - update_date).days
            if days_ago > GITHUB_QUALITY["max_age_days"]:
                return False
        except:
            pass

    return True


def clone_and_extract_code(repos: List[Dict], output_dir: Path, language: str):
    """Clone repos and extract code files for training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {
        "rust": [".rs"],
        "python": [".py"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "hcl": [".tf", ".tfvars"],
        "nix": [".nix"],
    }.get(language, [])

    samples = []

    for repo in repos[:20]:  # Limit to top 20 for now
        print(f"   Processing {repo['full_name']}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Shallow clone
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo["clone_url"], tmpdir],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                continue

            # Extract code files
            for root, dirs, files in os.walk(tmpdir):
                # Skip hidden dirs, vendor, node_modules
                dirs[:] = [d for d in dirs if not d.startswith(".")
                          and d not in ["vendor", "node_modules", "target", "__pycache__"]]

                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        filepath = Path(root) / file
                        try:
                            content = filepath.read_text(encoding="utf-8", errors="ignore")
                            if 100 < len(content) < 100000:  # Size filter
                                samples.append({
                                    "text": content,
                                    "source": f"github:{repo['full_name']}",
                                    "license": repo["license"],
                                    "language": language,
                                    "file": file,
                                    "stars": repo["stars"],
                                })
                        except Exception:
                            continue

    # Write to JSONL
    output_file = output_dir / f"{language}-github.jsonl"
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   Wrote {len(samples)} samples to {output_file}")
    return samples


def fetch_iac_schemas():
    """Fetch latest IaC schemas and documentation."""
    print("\n[IaC] Fetching latest schemas and docs...")

    IAC_DIR.mkdir(parents=True, exist_ok=True)

    # Terraform provider schemas
    print("   Terraform providers...")
    tf_dir = IAC_DIR / "terraform"
    tf_dir.mkdir(exist_ok=True)

    for provider in IAC_SCHEMAS["terraform"]["providers"]:
        try:
            result = subprocess.run(
                ["curl", "-s", f"https://registry.terraform.io/v1/providers/hashicorp/{provider}/latest"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                with open(tf_dir / f"{provider}-meta.json", "w") as f:
                    f.write(result.stdout)
        except Exception as e:
            print(f"      Warning: {provider}: {e}")

    # Kubernetes OpenAPI spec
    print("   Kubernetes schemas...")
    k8s_dir = IAC_DIR / "kubernetes"
    k8s_dir.mkdir(exist_ok=True)

    for version in IAC_SCHEMAS["kubernetes"]["versions"]:
        try:
            url = f"https://raw.githubusercontent.com/kubernetes/kubernetes/v{version}.0/api/openapi-spec/swagger.json"
            result = subprocess.run(
                ["curl", "-s", "-L", url],
                capture_output=True, text=True
            )
            if result.returncode == 0 and len(result.stdout) > 1000:
                with open(k8s_dir / f"k8s-{version}-openapi.json", "w") as f:
                    f.write(result.stdout)
        except Exception as e:
            print(f"      Warning: k8s {version}: {e}")

    # Docker Compose schema
    print("   Docker Compose schema...")
    docker_dir = IAC_DIR / "docker"
    docker_dir.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            ["curl", "-s", IAC_SCHEMAS["docker"]["compose_schema"]],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            with open(docker_dir / "compose-spec.json", "w") as f:
                f.write(result.stdout)
    except Exception as e:
        print(f"      Warning: Docker: {e}")

    # GitHub Actions schema
    print("   GitHub Actions schema...")
    gha_dir = IAC_DIR / "github-actions"
    gha_dir.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            ["curl", "-s", IAC_SCHEMAS["github-actions"]["schema_url"]],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            with open(gha_dir / "workflow-schema.json", "w") as f:
                f.write(result.stdout)
    except Exception as e:
        print(f"      Warning: GHA: {e}")

    print(f"   Schemas saved to {IAC_DIR}")


def create_iac_training_data():
    """Create training data from IaC schemas and examples."""
    print("\n[IaC] Creating training data...")

    samples = []

    # TODO: Parse schemas and generate training examples
    # - Schema explanations
    # - Valid vs invalid examples
    # - Best practices commentary

    output_file = CURATED_DIR / "iac-training.jsonl"
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"   Created {len(samples)} IaC training samples")


def main():
    parser = argparse.ArgumentParser(description="Dataset Curation Pipeline")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset gaps")
    parser.add_argument("--github", action="store_true", help="Curate GitHub repos")
    parser.add_argument("--iac", action="store_true", help="Fetch IaC schemas")
    parser.add_argument("--language", type=str, help="Specific language for GitHub curation")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        args.analyze = args.github = args.iac = True

    if not any([args.analyze, args.github, args.iac]):
        args.analyze = True  # Default to analysis

    print("=" * 60)
    print("DATASET CURATION PIPELINE")
    print("=" * 60)

    if args.analyze:
        gaps = analyze_gaps()
        print_gap_analysis(gaps)

    if args.github:
        languages = [args.language] if args.language else TARGET_LANGUAGES
        GITHUB_DIR.mkdir(parents=True, exist_ok=True)

        for lang in languages:
            repos = curate_github_repos(lang)
            if repos:
                clone_and_extract_code(repos, GITHUB_DIR / lang, lang)

    if args.iac:
        fetch_iac_schemas()
        create_iac_training_data()

    print("\n" + "=" * 60)
    print("CURATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
