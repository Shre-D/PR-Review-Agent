# Company Engineering Guidelines

## Security Standards

All code merged to main must meet these requirements:

### Authentication & Authorisation
- Every API endpoint must require authentication unless explicitly marked `@public`.
- JWT tokens must be validated with signature verification — never `decode(verify=False)`.
- Session tokens must never appear in logs, error messages, or response bodies.
- OAuth callback handlers must validate the `state` parameter to prevent CSRF.

### Data Handling
- No raw SQL string construction using user input. Use parameterised queries or an ORM.
- Passwords and secrets must never be hardcoded in source files or environment defaults.
- PII fields (email, phone, address) must be encrypted at rest using the company KMS.
- API responses must never return internal stack traces to clients.

### Secrets Management
- Use `settings.SECRET_KEY` for all secret access — never `os.environ["SECRET_KEY"]` directly.
- `.env` files must never be committed. `.env.example` with placeholder values is acceptable.
- GitHub Actions secrets must be accessed via `${{ secrets.NAME }}` — never hardcoded.

### Dependency Rules
- No new dependencies may be added without a security review if they have fewer than 1000 GitHub stars.
- All `pip install` in Dockerfiles must pin exact versions (`==x.y.z`), not ranges.
- `npm install` in CI must use `--frozen-lockfile` or `npm ci`.

## Code Quality Standards

### Python
- Functions must not exceed 40 lines. Classes must not exceed 200 lines.
- Cyclomatic complexity must not exceed 10 per function (enforced by radon).
- All public functions and classes must have docstrings.
- Use `ruff` for linting. `pylint` score must be >= 8.0.
- Type hints required on all new code. `mypy` must pass with no new errors.

### TypeScript / JavaScript
- `any` type is banned in new code. Use `unknown` or a proper type.
- `eslint` must pass with no new errors (`.eslintrc` at repo root governs rules).
- `tsc --noEmit` must pass with zero errors.

### Go
- `go vet` and `golangci-lint run` must pass.
- Error values must never be silently discarded (`_ = someFunc()`).
- Goroutines must always have a clear exit condition or context cancellation.

### Rust
- `cargo clippy -- -D warnings` must pass.
- `unwrap()` and `expect()` are banned in production code paths. Use `?` or match.
- Unsafe blocks require a comment explaining why they are necessary and safe.

### Java
- Null checks are required on all parameters accepted from external callers.
- `@SuppressWarnings` requires a comment justification.

## Testing Requirements

- All new features must have unit tests with >= 80% branch coverage on changed files.
- Database migrations must be reversible (both `up` and `down` migrations required).
- Integration tests must not depend on production data or external services (use fixtures).
- PRs touching the auth or payments domain must include at least one security-focused test.

## Infrastructure & Configuration

### Dockerfiles
- Images must not run as root. `USER nonroot` or equivalent is required.
- Base images must be pinned to a specific digest or version tag (not `:latest`).
- `EXPOSE` must only declare ports the service actually uses.
- Secrets must not be passed as `ARG` or `ENV` in the build stage.

### GitHub Actions
- Workflows must declare minimal `permissions`. Default is `contents: read`.
- `write` permissions require a comment justifying necessity.
- `pull-request: write` is banned except in dedicated triage workflows.
- Third-party actions must be pinned to a commit SHA, not a tag.

### YAML / Config
- All YAML files must pass `yamllint`.
- Environment variables that default to empty string for required config values are banned.
- Resource limits (`cpu`, `memory`) must be set on all Kubernetes pod specs.
