# System Architecture

## Overview

This is a multi-tenant SaaS platform providing financial data analytics. The system processes
sensitive payment and billing data for enterprise customers. Security and correctness are
the highest priorities — availability is secondary.

## Frontend

- **Framework:** React 18 + TypeScript (strict mode)
- **State management:** Zustand for global state, React Query for server state
- **Build:** Vite, output bundled to `dist/`
- **Linting:** ESLint with `@typescript-eslint/recommended` + custom rules
- **Key constraint:** No `any` types. All API responses validated with Zod schemas.

## Backend

- **Framework:** FastAPI (Python 3.11+)
- **Database ORM:** SQLAlchemy 2.x (async) with Alembic migrations
- **Database:** PostgreSQL 15 (primary), Redis 7 (cache + session store)
- **Auth:** JWT (RS256) issued by the auth service. All endpoints require `Authorization: Bearer <token>`.
- **Key constraint:** All DB writes go through service layer — no direct ORM calls in route handlers.

## Services

```
client (React/TS)
    │  HTTPS
    ▼
API Gateway (nginx)
    │
    ├── /api/v1/auth     → auth-service (Python/FastAPI)
    ├── /api/v1/payments → payments-service (Python/FastAPI)
    ├── /api/v1/billing  → billing-service (Python/FastAPI)
    ├── /api/v1/analytics→ analytics-service (Python/FastAPI)
    └── /static          → frontend bundle (served by nginx)

Shared:
    PostgreSQL (one DB per service, no cross-service queries)
    Redis (shared cache, namespaced by service prefix)
    Celery + RabbitMQ (async tasks)
```

## Infrastructure

- **Containerisation:** Docker. All services have their own `Dockerfile`.
- **Orchestration:** Kubernetes (GKE). Manifests in `infrastructure/k8s/`.
- **CI/CD:** GitHub Actions. Workflows in `.github/workflows/`.
- **Secrets:** GCP Secret Manager. Never in environment variables or config files.
- **Monitoring:** Prometheus + Grafana. All services expose `/metrics`.

## Critical Domains

| Domain | Path | Risk Level |
|--------|------|------------|
| Authentication | `src/auth/` | Critical |
| Payments processing | `src/payments/` | Critical |
| Billing engine | `src/billing/` | Critical |
| Database migrations | `db/migrations/` | High |
| Infrastructure config | `infrastructure/` | High |
| CI/CD workflows | `.github/workflows/` | High |
| API route handlers | `src/api/` | Medium |
| Frontend components | `src/frontend/` | Low-Medium |

## Data Flow for Payment Processing

```
POST /api/v1/payments/charge
    → JWT validation (auth middleware)
    → Request validation (Pydantic model)
    → Idempotency key check (Redis)
    → Payment service layer
    → Payment provider API (Stripe)
    → Database write (payments table, encrypted)
    → Audit log (append-only table)
    → Response (sanitised, no card data)
```

## Key Invariants

1. Card numbers and CVVs are **never stored** — passed directly to Stripe and discarded.
2. Every DB write to `payments` or `billing` tables must be logged to the audit table in the same transaction.
3. All auth tokens expire in 15 minutes. Refresh tokens expire in 7 days.
4. Rate limiting: 100 req/min per user, 1000 req/min per tenant. Enforced at API Gateway.