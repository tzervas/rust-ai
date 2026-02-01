#!/usr/bin/env python3
"""
IaC Dataset Processor - Converts collected IaC files to JSONL training format.

Processes:
- Terraform (.tf) files
- Kubernetes (.yaml, .yml) manifests
- Ansible playbooks (.yaml, .yml)
- Dockerfiles
- GitHub Actions workflows (.yaml, .yml)

Output: JSONL files with format:
{
    "text": "<file content>",
    "meta": {
        "source": "terraform",
        "file_type": ".tf",
        "repo": "terraform-provider-aws",
        "path": "examples/vpc/main.tf"
    }
}
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Generator, Tuple
import argparse

# Configuration
IAC_DIR = Path("/data/datasets/tritter/iac")
REPOS_DIR = IAC_DIR / "repos"
OUTPUT_DIR = IAC_DIR / "training"

# File extensions by domain
DOMAIN_EXTENSIONS = {
    "terraform": [".tf", ".tfvars"],
    "kubernetes": [".yaml", ".yml"],  # Need to filter for k8s content
    "ansible": [".yaml", ".yml"],      # Need to filter for ansible content
    "docker": ["Dockerfile"],          # Also docker-compose.yml
    "github-actions": [".yaml", ".yml"]  # Under .github/workflows
}

# Size limits
MAX_FILE_SIZE = 100 * 1024  # 100KB max per file
MIN_FILE_SIZE = 50          # Skip very small files


def calculate_hash(content: str) -> str:
    """Calculate SHA256 hash for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def is_valid_terraform(content: str) -> bool:
    """Check if file content looks like valid Terraform."""
    tf_keywords = ['resource', 'variable', 'output', 'provider', 'module', 'data', 'locals', 'terraform']
    return any(kw in content for kw in tf_keywords)


def is_valid_kubernetes(content: str) -> bool:
    """Check if file content looks like valid Kubernetes manifest."""
    k8s_keywords = ['apiVersion:', 'kind:', 'metadata:', 'spec:']
    return sum(1 for kw in k8s_keywords if kw in content) >= 2


def is_valid_ansible(content: str) -> bool:
    """Check if file content looks like valid Ansible."""
    ansible_keywords = ['hosts:', 'tasks:', 'roles:', 'handlers:', 'playbook', '- name:']
    return any(kw in content for kw in ansible_keywords)


def is_valid_dockerfile(content: str) -> bool:
    """Check if file content looks like valid Dockerfile."""
    dockerfile_keywords = ['FROM', 'RUN', 'CMD', 'COPY', 'ADD', 'WORKDIR', 'ENV', 'EXPOSE']
    return any(content.strip().startswith(kw) or f'\n{kw}' in content for kw in dockerfile_keywords)


def is_valid_github_actions(content: str) -> bool:
    """Check if file content looks like valid GitHub Actions workflow."""
    actions_keywords = ['on:', 'jobs:', 'runs-on:', 'steps:', 'uses:']
    return sum(1 for kw in actions_keywords if kw in content) >= 2


def detect_domain(file_path: Path, content: str) -> str:
    """Detect the IaC domain from file path and content."""
    path_str = str(file_path).lower()

    # Path-based detection
    if '/terraform/' in path_str or file_path.suffix in ['.tf', '.tfvars']:
        return 'terraform'

    if '/.github/workflows/' in path_str:
        return 'github-actions'

    if '/docker/' in path_str or file_path.name.startswith('Dockerfile'):
        return 'docker'

    if '/ansible/' in path_str:
        return 'ansible'

    if '/kubernetes/' in path_str or '/k8s/' in path_str:
        return 'kubernetes'

    # Content-based detection for YAML files
    if file_path.suffix in ['.yaml', '.yml']:
        if is_valid_kubernetes(content):
            return 'kubernetes'
        if is_valid_ansible(content):
            return 'ansible'
        if is_valid_github_actions(content):
            return 'github-actions'

    return 'unknown'


def process_file(file_path: Path, base_dir: Path) -> Dict | None:
    """Process a single file and return JSONL record."""
    try:
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE or file_size < MIN_FILE_SIZE:
            return None

        # Read content
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return None

        # Skip binary or empty files
        if not content.strip():
            return None

        # Detect domain
        domain = detect_domain(file_path, content)
        if domain == 'unknown':
            return None

        # Validate content based on domain
        validators = {
            'terraform': is_valid_terraform,
            'kubernetes': is_valid_kubernetes,
            'ansible': is_valid_ansible,
            'docker': is_valid_dockerfile,
            'github-actions': is_valid_github_actions
        }

        if domain in validators and not validators[domain](content):
            return None

        # Get relative path for metadata
        try:
            relative_path = file_path.relative_to(base_dir)
        except ValueError:
            relative_path = file_path.name

        # Extract repo name from path
        parts = relative_path.parts
        repo = parts[1] if len(parts) > 1 else "unknown"

        return {
            "text": content,
            "meta": {
                "source": domain,
                "file_type": file_path.suffix if file_path.suffix else file_path.name,
                "repo": repo,
                "path": str(relative_path),
                "size": file_size,
                "hash": calculate_hash(content)
            }
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None


def collect_files(repos_dir: Path) -> Generator[Path, None, None]:
    """Recursively collect all IaC files from repos directory."""
    extensions = {'.tf', '.tfvars', '.yaml', '.yml'}

    for root, dirs, files in os.walk(repos_dir):
        # Skip hidden directories and common non-IaC directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'vendor', '__pycache__']]

        for file in files:
            file_path = Path(root) / file

            # Include by extension
            if file_path.suffix in extensions:
                yield file_path

            # Include Dockerfiles
            if file.startswith('Dockerfile'):
                yield file_path


def process_domain_batch(domain: str, files: list, repos_dir: Path, output_dir: Path) -> Tuple[str, int, int]:
    """Process a batch of files for a specific domain."""
    output_file = output_dir / f"{domain}.jsonl"
    seen_hashes = set()
    written = 0
    skipped = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in files:
            record = process_file(file_path, repos_dir)
            if record and record['meta']['source'] == domain:
                # Deduplication
                file_hash = record['meta']['hash']
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    written += 1
                else:
                    skipped += 1

    return domain, written, skipped


def main():
    parser = argparse.ArgumentParser(description='Process IaC files to JSONL training format')
    parser.add_argument('--repos-dir', type=Path, default=REPOS_DIR, help='Input repos directory')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='Output directory for JSONL files')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    repos_dir = args.repos_dir
    output_dir = args.output_dir

    if not repos_dir.exists():
        print(f"Error: Repos directory not found: {repos_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting files from: {repos_dir}")

    # Collect all files
    all_files = list(collect_files(repos_dir))
    print(f"Found {len(all_files)} potential IaC files")

    # Process files and group by domain
    domain_files = {
        'terraform': [],
        'kubernetes': [],
        'ansible': [],
        'docker': [],
        'github-actions': []
    }

    seen_hashes = set()
    stats = {'processed': 0, 'skipped': 0, 'duplicates': 0}

    print("Processing files...")

    for file_path in all_files:
        record = process_file(file_path, repos_dir)
        if record:
            domain = record['meta']['source']
            file_hash = record['meta']['hash']

            if file_hash in seen_hashes:
                stats['duplicates'] += 1
                continue

            seen_hashes.add(file_hash)

            if domain in domain_files:
                domain_files[domain].append(record)
                stats['processed'] += 1
        else:
            stats['skipped'] += 1

    print(f"Processed: {stats['processed']}, Skipped: {stats['skipped']}, Duplicates: {stats['duplicates']}")

    # Write domain-specific JSONL files
    print("\nWriting JSONL files...")

    for domain, records in domain_files.items():
        if not records:
            print(f"  {domain}: 0 records (skipped)")
            continue

        output_file = output_dir / f"{domain}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"  {domain}: {len(records)} records ({file_size:.2f} MB)")

    # Write combined file
    combined_file = output_dir / "iac_combined.jsonl"
    total_records = 0
    with open(combined_file, 'w', encoding='utf-8') as f:
        for domain, records in domain_files.items():
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_records += 1

    combined_size = combined_file.stat().st_size / (1024 * 1024)
    print(f"\nCombined: {total_records} total records ({combined_size:.2f} MB)")
    print(f"Output directory: {output_dir}")

    # Write metadata
    metadata = {
        "created": str(Path(__file__).stat().st_mtime),
        "source_dir": str(repos_dir),
        "stats": {
            "total_files_scanned": len(all_files),
            "processed": stats['processed'],
            "skipped": stats['skipped'],
            "duplicates": stats['duplicates']
        },
        "domains": {
            domain: len(records) for domain, records in domain_files.items()
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
