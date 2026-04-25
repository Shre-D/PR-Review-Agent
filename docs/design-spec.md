# Design Specification

## API Design Patterns

### REST Conventions
- Resources are plural nouns: `/payments`, `/users`, `/invoices`
- Use HTTP verbs correctly: `GET` for reads (idempotent), `POST` for creation, `PUT` for full update, `PATCH` for partial update, `DELETE` for removal
- Return `201 Created` with `Location` header on resource creation
- Return `204 No Content` on successful `DELETE`
- Never return 200 for errors. Use 4xx/5xx correctly.

### Request/Response Format
- All request and response bodies are JSON
- Dates in ISO 8601 format (`2026-04-24T14:30:00Z`)
- Monetary values in integer cents (never floats)
- Pagination via `cursor` (not page/offset) for large collections
- Error responses must follow this schema:
  ```json
  {"error": {"code": "PAYMENT_FAILED", "message": "...", "request_id": "..."}}
  ```

### Versioning
- All routes prefixed with `/api/v1/`
- Breaking changes require a new version prefix (`/api/v2/`)
- Old versions must remain functional for 6 months after deprecation notice

## Python Code Patterns

### Service Layer Pattern
```python
# CORRECT: route calls service, service calls repository
@router.post("/payments")
async def create_payment(body: PaymentRequest, user: User = Depends(get_current_user)):
    return await payment_service.create(user.id, body)

# WRONG: direct ORM access in route handler
@router.post("/payments")
async def create_payment(body: PaymentRequest, db: Session = Depends(get_db)):
    payment = Payment(**body.dict())
    db.add(payment)  # ← never do this in a route handler
```

### Error Handling
```python
# CORRECT: typed exceptions bubble up to global handler
async def charge_card(amount: int, token: str) -> Payment:
    if amount <= 0:
        raise ValidationError("Amount must be positive")
    result = await stripe.charge(token, amount)
    if not result.success:
        raise PaymentError(result.error_code)
    return Payment.from_stripe(result)

# WRONG: silent failure or bare except
async def charge_card(amount, token):
    try:
        return await stripe.charge(token, amount)
    except Exception:
        return None  # ← caller has no idea what happened
```

### Async Patterns
- All database operations must use async SQLAlchemy (`AsyncSession`)
- CPU-bound work must be offloaded with `asyncio.to_thread()` or Celery
- Never call blocking I/O in an async function (`time.sleep`, `requests.get`, etc.)

## TypeScript Patterns

### API Client
```typescript
// CORRECT: typed response with Zod validation
const PaymentSchema = z.object({ id: z.string(), amount: z.number(), status: z.enum(["pending", "completed", "failed"]) });
type Payment = z.infer<typeof PaymentSchema>;

async function createPayment(body: CreatePaymentRequest): Promise<Payment> {
    const response = await apiClient.post("/payments", body);
    return PaymentSchema.parse(response.data);  // runtime validation
}

// WRONG: any type, no validation
async function createPayment(body: any): Promise<any> {
    const response = await fetch("/payments", { body: JSON.stringify(body) });
    return response.json();  // no validation, trust the server
}
```

### State Management
- Server state (API data) lives in React Query — never in Zustand
- UI state (modal open, selected tab) lives in component state or Zustand
- Never store derived data — compute it from source state

## Database Patterns

### Migrations
- Every migration must have a `downgrade` function that fully reverses the `upgrade`
- Column additions must have a default value or be nullable
- Column removals must be split into two PRs: first deprecate (keep column, stop writing), then remove
- Never modify existing migrations — always create a new one

### Query Patterns
- `SELECT *` is banned — always name columns explicitly
- `DISTINCT` on large tables requires a comment explaining why an index won't work
- N+1 queries must be eliminated with `joinedload` or batch fetching

## Security Patterns

### Input Validation
- All user input validated at the API boundary with Pydantic/Zod before touching business logic
- File uploads: validate MIME type from file content, not `Content-Type` header
- User-supplied IDs must be validated against the authenticated user's permissions (no IDOR)

### Logging
```python
# CORRECT: log event, not data
logger.info("Payment created", extra={"payment_id": payment.id, "user_id": user.id})

# WRONG: log sensitive data
logger.info(f"Payment created for {user.email} with card {card_number}")
```