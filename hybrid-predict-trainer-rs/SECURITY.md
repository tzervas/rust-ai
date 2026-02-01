# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Email:** security@vectorweight.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 7 days
- **Resolution Target:** Within 30 days for critical issues

### What to Expect

1. We will acknowledge receipt of your report
2. We will investigate and validate the issue
3. We will work on a fix and coordinate disclosure timing
4. We will credit you (unless you prefer anonymity) in the security advisory

### Out of Scope

- Vulnerabilities in dependencies (report to upstream)
- Issues requiring physical access
- Social engineering attacks
- Denial of service attacks

## Security Measures

This crate implements several security measures:

- **No unsafe code** by default (`#![deny(unsafe_code)]` in workspace lints)
- **Dependency auditing** via `cargo-audit` in CI
- **License compliance** checking via `cargo-deny`
- **Fuzz testing** planned for input parsing

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities.
