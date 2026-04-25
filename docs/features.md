# Critical Features & Protected Systems

## Tier 1: Cannot Break (Immediate incident if degraded)

### Payment Processing (`src/payments/`)
- Card charge, refund, and void flows via Stripe API
- Idempotency key enforcement (duplicate charge prevention)
- Webhook processing for async payment events
- **Any PR touching this path requires: check_security + check_tests + check_build_and_types**

### Authentication (`src/auth/`)
- JWT issuance, validation, and revocation
- OAuth2 flows (Google, GitHub SSO)
- Session management via Redis
- MFA verification
- **Any PR touching this path requires: check_security + check_quality**

### Billing Engine (`src/billing/`)
- Invoice generation and line-item calculation
- Subscription lifecycle (create, upgrade, downgrade, cancel)
- Proration calculations
- Tax computation
- **Monetary calculations must use integer cents — float arithmetic is banned here**

## Tier 2: High Impact (Degradation affects paying customers)

### Database Migrations (`db/migrations/`)
- Must always be reversible
- Must not lock tables for more than 100ms
- Data migrations must be batched (max 1000 rows per transaction)
- **Any migration PR must include a rollback plan in the PR description**

### API Rate Limiting (`src/middleware/rate_limit.py`)
- Removal or misconfiguration of rate limiting is a Tier 1 incident
- Changes require load test results proving the new config holds under 10× normal traffic

### Audit Logging (`src/audit/`)
- Audit log writes must never fail silently
- The audit log table is append-only — no `UPDATE` or `DELETE` is ever valid here

## Tier 3: Standard Features (Normal review process)

### Analytics Dashboard (`src/analytics/`)
- Read-only queries against a replica database
- Performance matters but correctness is not safety-critical

### Notification Service (`src/notifications/`)
- Email and webhook delivery
- Failures are retried but not critical-path

### Admin Panel (`src/admin/`)
- Internal tooling only — not customer-facing
- Still requires security review (admin endpoints are high-value attack targets)

## Feature Flags

The platform uses LaunchDarkly for feature flags. Rules:
- New features must ship behind a flag (`@feature_flag("feature_name")`)
- Flags must be cleaned up within 30 days of full rollout
- Flags controlling payment flows require a separate staging validation

## SLA Commitments

| Service | Uptime SLA | Max Latency (p99) |
|---------|------------|-------------------|
| Payment API | 99.95% | 2s |
| Auth API | 99.99% | 500ms |
| Analytics | 99.5% | 5s |
| Frontend | 99.9% | 3s (LCP) |

PRs that add synchronous external API calls to the critical path (payments, auth) must
include a timeout and circuit breaker — no unbounded waits.