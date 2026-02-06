# License Policy

## Project License

All code in this workspace is licensed under the **MIT License** only.

## Rationale

- **Simplicity**: Single license reduces legal complexity
- **Permissive**: MIT allows maximum flexibility for users and contributors
- **Ecosystem Alignment**: Compatible with dual MIT/Apache Rust ecosystem

## Previous Dual Licensing

Earlier versions used "MIT OR Apache-2.0" dual licensing. As of January 2026, 
the project transitioned to MIT-only to:

1. Avoid Apache 2.0 patent clause complications
2. Simplify license compliance for downstream users
3. Align with project goals of maximum permissiveness

## Dependency Requirements

### Acceptable Licenses
- MIT
- MIT OR Apache-2.0 (used under MIT terms)
- BSD-2-Clause, BSD-3-Clause
- ISC
- Unlicense
- CC0

### File-Level Copyleft (Case-by-Case)
- MPL-2.0 (Mozilla Public License): Allowed for unmodified transitive dependencies
  - Example: `option-ext` via `dirs` → `cubecl`
  - Must not be directly modified without MPL compliance

### Prohibited Licenses (Viral Copyleft)
- GPL (v2, v3, any version)
- LGPL (v2.1, v3, any version)  
- AGPL (any version)
- SSPL, BSL (Server Side Public License, Business Source License)

## Compliance Process

Before adding a new direct dependency:

1. Check license: `cargo metadata --format-version 1 | jq '.packages[] | select(.name == "CRATE") | .license'`
2. Verify it's on the acceptable list
3. If uncertain, open an issue for review
4. Update NOTICE file if adding attribution-required dependency

## Audit Commands

```bash
# List all dependency licenses
cargo tree --workspace --format "{p} {l}" | sort -u

# Find potential viral licenses
cargo tree --workspace --format "{p} {l}" | grep -iE "GPL|LGPL|AGPL|SSPL"

# Check specific crate
cargo metadata --format-version 1 | jq -r '.packages[] | select(.name == "CRATE_NAME") | .license'
```

## Attribution

See [NOTICE](NOTICE) file for required third-party attributions.

## Questions

For licensing questions, open a GitHub issue with the `licensing` label.
