# Review Tool Configuration
# This file is parsed structurally (no model needed).
# Values here override anything extracted from other docs.

## Tool Weights
| Tool | Weight |
|------|--------|
| check_security | 0.40 |
| check_quality | 0.22 |
| check_build_and_types | 0.18 |
| check_tests | 0.12 |
| check_config | 0.08 |

## Verdict Thresholds
- reject_threshold: 0.30
- request_changes_threshold: 0.62

## Domain Priorities
- security: 1.6
- config: 1.3
- quality: 1.0
- build: 0.9
- tests: 0.8

## Critical Paths
- src/auth/
- src/payments/
- src/billing/
- db/migrations/
- .github/workflows/
- Dockerfile
- docker-compose.yml
- infrastructure/

## Author Depth Overrides
- junior: 1.5
- mid: 1.0
- senior: 0.75
- lead: 0.6
- non_tech: 1.3

## Escalation Confidence Threshold
- escalation_confidence_threshold: 0.55

## Implemented External Tools
- semgrep
- ruff
- pylint
- radon
- pyyaml
- go
- javac
- cargo
- tsc

## Planned External Tools
- bandit
- gitleaks
- hadolint
- actionlint
- yamllint
- mypy

## Custom Rules
| Domain | Pattern | Severity | Message |
|--------|---------|----------|---------|
| security | src/auth/.*jwt\.decode\(.*verify_signature.*False | critical | Auth middleware must not disable JWT signature verification. |
| security | logger\..*password | critical | Auth and password flows must not log credentials, tokens, or sensitive PII. |
| quality | db/migrations/.*RunPython\( | warning | Data migrations must include a rollback or reverse_code plan. |
| config | infrastructure/.*resources: | warning | Infrastructure changes must preserve resource limits and health probes unless explicitly justified. |
| config | \.github/workflows/.*permissions:.*write | warning | GitHub Actions write permissions require least-privilege justification. |
| config | Dockerfile | warning | Dockerfile changes require pinned base images and an explicit non-root user. |
