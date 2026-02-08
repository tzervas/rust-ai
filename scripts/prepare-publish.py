#!/usr/bin/env python3
"""
prepare-publish.py - Convert workspace dependencies to explicit versions for publishing

Usage: ./scripts/prepare-publish.py <crate-name> [--dry-run] [--output PATH] [--validate]

This script implements Option C: Workspace Development + Pre-Publish Script
- Local development uses { workspace = true } for DRY dependency management
- Publishing uses explicit versions for standalone compatibility
- Strips path = "../..." dependencies (crates.io requirement)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

def extract_workspace_versions(workspace_toml: Path) -> Dict[str, str]:
    """Extract version mappings from workspace [workspace.dependencies] section."""
    versions = {}
    in_workspace_deps = False

    with open(workspace_toml) as f:
        for line in f:
            line = line.strip()

            # Detect [workspace.dependencies] section
            if line == '[workspace.dependencies]':
                in_workspace_deps = True
                continue

            # Exit section on next [header]
            if in_workspace_deps and line.startswith('[') and line != '[workspace.dependencies]':
                break

            # Parse dependency lines
            if in_workspace_deps and '=' in line:
                # Match: dep-name = "version" or dep-name = { version = "version", ... }
                match = re.match(r'^([a-zA-Z0-9_-]+)\s*=\s*(.+)$', line)
                if match:
                    dep_name = match.group(1)
                    dep_value = match.group(2)

                    # Extract version
                    # Format 1: "0.9.2"
                    version_match = re.search(r'^"([0-9.]+)"', dep_value)
                    if version_match:
                        versions[dep_name] = version_match.group(1)
                    else:
                        # Format 2: { version = "0.9.2", ... }
                        version_match = re.search(r'version\s*=\s*"([0-9.]+)"', dep_value)
                        if version_match:
                            versions[dep_name] = version_match.group(1)

    return versions

def convert_workspace_to_explicit(
    crate_toml: Path,
    workspace_versions: Dict[str, str],
    output_path: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Convert { workspace = true } references to explicit versions and strip path dependencies.

    Returns:
        Tuple of (replacements, warnings) counts
    """

    if output_path is None:
        output_path = crate_toml.with_suffix('.toml.publish')

    replacements = 0
    warnings = 0

    with open(crate_toml) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Match: dep-name = { workspace = true, ... }
            workspace_match = re.match(
                r'^(\s*)([a-zA-Z0-9_-]+)\s*=\s*\{\s*workspace\s*=\s*true\s*(.*?)\}(.*)$',
                line
            )

            if workspace_match:
                indent = workspace_match.group(1)
                dep_name = workspace_match.group(2)
                rest = workspace_match.group(3).strip()  # e.g., ", features = [...]"
                comment = workspace_match.group(4)  # trailing comment

                # Get version from workspace
                if dep_name in workspace_versions:
                    version = workspace_versions[dep_name]

                    # Build replacement
                    if rest:
                        # Has additional properties (e.g., features)
                        rest = rest.lstrip(', ')
                        line = f'{indent}{dep_name} = {{ version = "{version}", {rest} }}{comment}\n'
                    else:
                        # Simple case: just version
                        line = f'{indent}{dep_name} = "{version}"{comment}\n'

                    print(f'  ✓ {dep_name}: workspace → "{version}"', file=sys.stderr)
                    replacements += 1
                else:
                    print(f'  ⚠  {dep_name}: workspace version not found, keeping as-is', file=sys.stderr)
                    warnings += 1

            # Match: dep-name = { version = "...", path = "../...", ... }
            # CRITICAL: Strip path dependencies for crates.io publishing
            path_match = re.match(
                r'^(\s*)([a-zA-Z0-9_-]+)\s*=\s*\{([^}]+)\}(.*)$',
                line
            )

            if path_match and 'path' in line:
                indent = path_match.group(1)
                dep_name = path_match.group(2)
                dep_spec = path_match.group(3)
                comment = path_match.group(4)

                # Parse the dependency spec
                parts = {}
                for part in dep_spec.split(','):
                    part = part.strip()
                    if '=' in part:
                        key_match = re.match(r'([a-zA-Z0-9_-]+)\s*=\s*(.+)', part)
                        if key_match:
                            parts[key_match.group(1)] = key_match.group(2).strip()

                # If has path, strip it but keep version/features
                if 'path' in parts:
                    # Build new spec without path
                    new_parts = []
                    for key, value in parts.items():
                        if key != 'path':
                            new_parts.append(f'{key} = {value}')

                    if new_parts:
                        line = f'{indent}{dep_name} = {{ {", ".join(new_parts)} }}{comment}\n'
                        print(f'  ✓ {dep_name}: stripped path dependency', file=sys.stderr)
                        replacements += 1
                    else:
                        print(f'  ⚠  {dep_name}: path-only dependency, needs version!', file=sys.stderr)
                        warnings += 1

            outfile.write(line)

    return replacements, warnings

def main():
    parser = argparse.ArgumentParser(description='Prepare crate for publishing by converting workspace deps')
    parser.add_argument('crate_name', help='Name of the crate to prepare')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing')
    parser.add_argument('--output', type=Path, help='Output file path (default: Cargo.toml.publish)')
    parser.add_argument('--validate', action='store_true', help='Run cargo publish --dry-run after conversion')
    args = parser.parse_args()

    # Paths
    workspace_root = Path(__file__).parent.parent.resolve()
    workspace_toml = workspace_root / 'Cargo.toml'
    crate_dir = workspace_root / args.crate_name
    crate_toml = crate_dir / 'Cargo.toml'

    # Validate
    if not crate_dir.exists():
        print(f'Error: Crate directory not found: {crate_dir}', file=sys.stderr)
        sys.exit(1)

    if not crate_toml.exists():
        print(f'Error: Cargo.toml not found: {crate_toml}', file=sys.stderr)
        sys.exit(1)

    print(f'=== Preparing {args.crate_name} for publishing ===', file=sys.stderr)
    print(f'Workspace: {workspace_root}', file=sys.stderr)
    print(f'Crate: {crate_dir}', file=sys.stderr)
    print('', file=sys.stderr)

    # Extract workspace versions
    print('Extracting workspace dependency versions...', file=sys.stderr)
    workspace_versions = extract_workspace_versions(workspace_toml)
    print(f'Found {len(workspace_versions)} workspace dependencies', file=sys.stderr)
    print('', file=sys.stderr)

    # Convert
    print('Converting workspace references and stripping path dependencies...', file=sys.stderr)

    if args.dry_run:
        output_path = Path('/tmp/dry-run-Cargo.toml')
    elif args.output:
        output_path = args.output
    else:
        output_path = crate_toml.with_suffix('.toml.publish')

    replacements, warnings = convert_workspace_to_explicit(
        crate_toml,
        workspace_versions,
        output_path
    )

    print('', file=sys.stderr)
    print(f'✓ Processed {replacements} dependencies', file=sys.stderr)
    if warnings > 0:
        print(f'⚠  {warnings} warnings (check manually)', file=sys.stderr)

    print('', file=sys.stderr)
    print(f'Output: {output_path}', file=sys.stderr)

    # Optional validation
    if args.validate and not args.dry_run:
        print('', file=sys.stderr)
        print('Running cargo publish --dry-run...', file=sys.stderr)

        # Temporarily swap Cargo.toml
        backup_path = crate_toml.with_suffix('.toml.backup')
        crate_toml.rename(backup_path)
        output_path.rename(crate_toml)

        try:
            result = subprocess.run(
                ['cargo', 'publish', '--dry-run'],
                cwd=crate_dir,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print('✓ Dry-run passed! Ready to publish.', file=sys.stderr)
            else:
                print('✗ Dry-run failed:', file=sys.stderr)
                print(result.stderr, file=sys.stderr)

        finally:
            # Restore original
            crate_toml.rename(output_path)
            backup_path.rename(crate_toml)

    if args.dry_run:
        print('', file=sys.stderr)
        print('Dry-run mode - no file written to crate directory', file=sys.stderr)
        output_path.unlink()  # Clean up temp file

if __name__ == '__main__':
    main()
