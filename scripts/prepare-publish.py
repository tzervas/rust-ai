#!/usr/bin/env python3
"""
prepare-publish.py - Convert workspace dependencies to explicit versions for publishing

Usage: ./scripts/prepare-publish.py <crate-name> [--dry-run] [--output PATH]

This script implements Option C: Workspace Development + Pre-Publish Script
- Local development uses { workspace = true } for DRY dependency management
- Publishing uses explicit versions for standalone compatibility
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional

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
) -> int:
    """Convert { workspace = true } references to explicit versions."""

    if output_path is None:
        output_path = crate_toml.with_suffix('.toml.publish')

    replacements = 0
    warnings = 0

    with open(crate_toml) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            original_line = line

            # Match: dep-name = { workspace = true }
            # or: dep-name = { workspace = true, features = [...] }
            match = re.match(
                r'^(\s*)([a-zA-Z0-9_-]+)\s*=\s*\{\s*workspace\s*=\s*true\s*(.*?)\}(.*)$',
                line
            )

            if match:
                indent = match.group(1)
                dep_name = match.group(2)
                rest = match.group(3).strip()  # e.g., ", features = [...]"
                comment = match.group(4)  # trailing comment

                # Get version from workspace
                if dep_name in workspace_versions:
                    version = workspace_versions[dep_name]

                    # Build replacement
                    if rest:
                        # Has additional properties (e.g., features)
                        # Clean up comma if present
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

            outfile.write(line)

    return replacements, warnings

def main():
    parser = argparse.ArgumentParser(description='Prepare crate for publishing by converting workspace deps')
    parser.add_argument('crate_name', help='Name of the crate to prepare')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing')
    parser.add_argument('--output', type=Path, help='Output file path (default: Cargo.toml.publish)')
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
    print('Converting workspace references to explicit versions...', file=sys.stderr)

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
    print(f'✓ Converted {replacements} workspace dependencies', file=sys.stderr)
    if warnings > 0:
        print(f'⚠  {warnings} warnings (versions not found in workspace)', file=sys.stderr)

    print('', file=sys.stderr)
    print(f'Output: {output_path}', file=sys.stderr)

    if args.dry_run:
        print('', file=sys.stderr)
        print('Dry-run mode - no file written to crate directory', file=sys.stderr)
        output_path.unlink()  # Clean up temp file

if __name__ == '__main__':
    main()
