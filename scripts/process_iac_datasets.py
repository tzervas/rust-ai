#!/usr/bin/env python3
"""
Infrastructure as Code (IaC) Dataset Processing and Curation Script

Processes downloaded IaC datasets for training machine learning models.
Handles tokenization, deduplication, filtering, and split generation.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from datetime import datetime

import yaml


class IaCDatasetProcessor:
    """Process and curate Infrastructure as Code datasets."""

    def __init__(self, iac_dir: str, output_dir: str = None, verbose: bool = False):
        """
        Initialize the processor.

        Args:
            iac_dir: Root directory containing IaC datasets
            output_dir: Output directory for processed datasets
            verbose: Enable verbose logging
        """
        self.iac_dir = Path(iac_dir)
        self.datasets_dir = self.iac_dir / "datasets"
        self.output_dir = Path(output_dir) if output_dir else self.iac_dir / "processed"
        self.verbose = verbose

        # Statistics
        self.stats = {
            'terraform': {'count': 0, 'size': 0, 'errors': 0},
            'kubernetes': {'count': 0, 'size': 0, 'errors': 0},
            'ansible': {'count': 0, 'size': 0, 'errors': 0},
            'docker': {'count': 0, 'size': 0, 'errors': 0},
            'github_actions': {'count': 0, 'size': 0, 'errors': 0},
        }
        self.seen_hashes: Set[str] = set()
        self.domain_patterns: Dict[str, List[str]] = {
            'terraform': ['.tf'],
            'kubernetes': ['.yaml', '.yml'],
            'ansible': ['.yaml', '.yml'],
            'docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml', '.dockerignore'],
            'github_actions': ['.yaml', '.yml'],
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message with optional verbosity filter."""
        if level == "VERBOSE" and not self.verbose:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def get_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def is_duplicate(self, file_path: Path) -> bool:
        """Check if file is a duplicate based on content hash."""
        try:
            file_hash = self.get_file_hash(file_path)
            if file_hash in self.seen_hashes:
                return True
            self.seen_hashes.add(file_hash)
            return False
        except Exception as e:
            self.log(f"Error computing hash for {file_path}: {e}")
            return False

    def process_terraform_files(self, output_subdir: Path):
        """Process Terraform .tf files."""
        self.log("Processing Terraform files...")
        source_dir = self.datasets_dir / "terraform-files"
        output_dir = output_subdir / "terraform"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            self.log(f"Source directory not found: {source_dir}", "WARN")
            return

        combined_output = output_dir / "all-terraform.tf"
        with open(combined_output, 'w') as combined:
            combined.write("# Combined Terraform Configurations\n")
            combined.write(f"# Generated: {datetime.now().isoformat()}\n")
            combined.write("# Source: Terraform examples and modules\n\n")

            for tf_file in source_dir.rglob("*.tf"):
                if self.is_duplicate(tf_file):
                    self.log(f"Skipping duplicate: {tf_file}", "VERBOSE")
                    self.stats['terraform']['errors'] += 1
                    continue

                try:
                    with open(tf_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            combined.write(f"\n# File: {tf_file.relative_to(source_dir)}\n")
                            combined.write(content)
                            combined.write("\n")
                            self.stats['terraform']['count'] += 1
                            self.stats['terraform']['size'] += len(content.encode('utf-8'))
                except Exception as e:
                    self.log(f"Error processing {tf_file}: {e}", "WARN")
                    self.stats['terraform']['errors'] += 1

        self.log(f"Terraform processing complete: {self.stats['terraform']['count']} files")

    def process_kubernetes_manifests(self, output_subdir: Path):
        """Process Kubernetes YAML manifests."""
        self.log("Processing Kubernetes manifests...")
        source_dir = self.datasets_dir / "kubernetes-manifests"
        output_dir = output_subdir / "kubernetes"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            self.log(f"Source directory not found: {source_dir}", "WARN")
            return

        kind_counter = defaultdict(int)

        for yaml_file in source_dir.rglob("*.yaml"):
            if yaml_file.name == "metadata.json":
                continue

            if self.is_duplicate(yaml_file):
                self.log(f"Skipping duplicate: {yaml_file}", "VERBOSE")
                self.stats['kubernetes']['errors'] += 1
                continue

            try:
                with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Try to parse YAML to understand structure
                    try:
                        docs = list(yaml.safe_load_all(content))
                        for doc in docs:
                            if doc and isinstance(doc, dict):
                                kind = doc.get('kind', 'Unknown')
                                kind_counter[kind] += 1
                    except yaml.YAMLError:
                        pass

                    if content.strip():
                        self.stats['kubernetes']['count'] += 1
                        self.stats['kubernetes']['size'] += len(content.encode('utf-8'))
            except Exception as e:
                self.log(f"Error processing {yaml_file}: {e}", "WARN")
                self.stats['kubernetes']['errors'] += 1

        # Save kind statistics
        kind_stats_file = output_dir / "kubernetes-kinds.json"
        with open(kind_stats_file, 'w') as f:
            json.dump(dict(kind_counter), f, indent=2)

        self.log(f"Kubernetes processing complete: {self.stats['kubernetes']['count']} manifests")
        self.log(f"Kubernetes kinds found: {len(kind_counter)} types")

    def process_ansible_playbooks(self, output_subdir: Path):
        """Process Ansible YAML playbooks."""
        self.log("Processing Ansible playbooks...")
        source_dir = self.datasets_dir / "ansible-playbooks"
        output_dir = output_subdir / "ansible"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            self.log(f"Source directory not found: {source_dir}", "WARN")
            return

        combined_output = output_dir / "all-playbooks.yaml"
        with open(combined_output, 'w') as combined:
            combined.write("# Combined Ansible Playbooks\n")
            combined.write(f"# Generated: {datetime.now().isoformat()}\n")
            combined.write("# Note: This is a concatenation for reference; playbooks should run individually\n\n")

            for yaml_file in source_dir.rglob("*.yaml"):
                if yaml_file.name == "metadata.json":
                    continue

                if self.is_duplicate(yaml_file):
                    self.log(f"Skipping duplicate: {yaml_file}", "VERBOSE")
                    self.stats['ansible']['errors'] += 1
                    continue

                try:
                    with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            combined.write(f"\n# File: {yaml_file.relative_to(source_dir)}\n")
                            combined.write("---\n")
                            combined.write(content)
                            combined.write("\n")
                            self.stats['ansible']['count'] += 1
                            self.stats['ansible']['size'] += len(content.encode('utf-8'))
                except Exception as e:
                    self.log(f"Error processing {yaml_file}: {e}", "WARN")
                    self.stats['ansible']['errors'] += 1

        self.log(f"Ansible processing complete: {self.stats['ansible']['count']} playbooks")

    def process_docker_files(self, output_subdir: Path):
        """Process Dockerfiles and docker-compose files."""
        self.log("Processing Docker files...")
        source_dir = self.datasets_dir / "docker-files"
        output_dir = output_subdir / "docker"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            self.log(f"Source directory not found: {source_dir}", "WARN")
            return

        dockerfile_count = 0
        compose_count = 0

        # Process Dockerfiles
        for dockerfile in source_dir.rglob("Dockerfile*"):
            if self.is_duplicate(dockerfile):
                self.log(f"Skipping duplicate: {dockerfile}", "VERBOSE")
                self.stats['docker']['errors'] += 1
                continue

            try:
                with open(dockerfile, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        dockerfile_count += 1
                        self.stats['docker']['count'] += 1
                        self.stats['docker']['size'] += len(content.encode('utf-8'))
            except Exception as e:
                self.log(f"Error processing {dockerfile}: {e}", "WARN")
                self.stats['docker']['errors'] += 1

        # Process docker-compose files
        for compose_file in source_dir.rglob("docker-compose*"):
            if compose_file.suffix in ['.yml', '.yaml']:
                if self.is_duplicate(compose_file):
                    self.log(f"Skipping duplicate: {compose_file}", "VERBOSE")
                    self.stats['docker']['errors'] += 1
                    continue

                try:
                    with open(compose_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            compose_count += 1
                            self.stats['docker']['count'] += 1
                            self.stats['docker']['size'] += len(content.encode('utf-8'))
                except Exception as e:
                    self.log(f"Error processing {compose_file}: {e}", "WARN")
                    self.stats['docker']['errors'] += 1

        self.log(f"Docker processing complete: {dockerfile_count} Dockerfiles, {compose_count} compose files")

    def process_github_actions_workflows(self, output_subdir: Path):
        """Process GitHub Actions workflow files."""
        self.log("Processing GitHub Actions workflows...")
        source_dir = self.datasets_dir / "github-actions-workflows"
        output_dir = output_subdir / "github-actions"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            self.log(f"Source directory not found: {source_dir}", "WARN")
            return

        combined_output = output_dir / "all-workflows.yaml"
        with open(combined_output, 'w') as combined:
            combined.write("# Combined GitHub Actions Workflows\n")
            combined.write(f"# Generated: {datetime.now().isoformat()}\n")
            combined.write("# Note: Individual workflow files for reference\n\n")

            for yaml_file in source_dir.rglob("*.yaml"):
                if yaml_file.name == "metadata.json":
                    continue

                if self.is_duplicate(yaml_file):
                    self.log(f"Skipping duplicate: {yaml_file}", "VERBOSE")
                    self.stats['github_actions']['errors'] += 1
                    continue

                try:
                    with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            combined.write(f"\n# File: {yaml_file.relative_to(source_dir)}\n")
                            combined.write("# " + "="*70 + "\n")
                            combined.write(content)
                            combined.write("\n")
                            self.stats['github_actions']['count'] += 1
                            self.stats['github_actions']['size'] += len(content.encode('utf-8'))
                except Exception as e:
                    self.log(f"Error processing {yaml_file}: {e}", "WARN")
                    self.stats['github_actions']['errors'] += 1

        self.log(f"GitHub Actions processing complete: {self.stats['github_actions']['count']} workflows")

    def generate_statistics_report(self, output_subdir: Path):
        """Generate comprehensive statistics report."""
        self.log("Generating statistics report...")

        report_file = output_subdir / "statistics.json"

        # Calculate totals
        total_files = sum(s['count'] for s in self.stats.values())
        total_size = sum(s['size'] for s in self.stats.values())
        total_errors = sum(s['errors'] for s in self.stats.values())

        report = {
            'generated': datetime.now().isoformat(),
            'source_directory': str(self.iac_dir),
            'output_directory': str(output_subdir),
            'totals': {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_errors': total_errors,
                'unique_files': len(self.seen_hashes),
            },
            'by_domain': self.stats,
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"Statistics report saved: {report_file}")
        return report

    def process_all(self) -> Dict:
        """Process all IaC datasets."""
        self.log("="*70)
        self.log("Starting IaC Dataset Processing")
        self.log("="*70)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each domain
        self.process_terraform_files(self.output_dir)
        self.process_kubernetes_manifests(self.output_dir)
        self.process_ansible_playbooks(self.output_dir)
        self.process_docker_files(self.output_dir)
        self.process_github_actions_workflows(self.output_dir)

        # Generate report
        report = self.generate_statistics_report(self.output_dir)

        self.log("="*70)
        self.log("Processing Complete")
        self.log("="*70)

        return report

    def print_summary(self, report: Dict):
        """Print summary of processing results."""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"\nGenerated: {report['generated']}")
        print(f"Output Directory: {report['output_directory']}")

        print(f"\nTotal Files Processed: {report['totals']['total_files']:,}")
        print(f"Total Size: {report['totals']['total_size_mb']:.2f} MB")
        print(f"Unique Files: {report['totals']['unique_files']:,}")
        print(f"Errors: {report['totals']['total_errors']}")

        print(f"\nBy Domain:")
        for domain, stats in report['by_domain'].items():
            print(f"  {domain.replace('_', ' ').title()}:")
            print(f"    - Files: {stats['count']:,}")
            print(f"    - Size: {stats['size'] / (1024*1024):.2f} MB")
            if stats['errors'] > 0:
                print(f"    - Errors: {stats['errors']}")

        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process and curate Infrastructure as Code datasets'
    )
    parser.add_argument(
        '--iac-dir',
        default='/data/datasets/tritter/iac',
        help='Root directory of IaC datasets (default: /data/datasets/tritter/iac)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for processed datasets (default: iac_dir/processed)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    processor = IaCDatasetProcessor(
        iac_dir=args.iac_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    report = processor.process_all()
    processor.print_summary(report)


if __name__ == '__main__':
    main()
