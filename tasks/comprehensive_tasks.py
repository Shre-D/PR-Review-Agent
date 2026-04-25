"""13 comprehensive multi-file PR tasks for integration testing and loader benchmark.

These tasks are more realistic than the benchmark bank:
- Multi-file diffs (2-5 files per PR)
- Large refactors (50-150 diff lines)
- Cover all critical paths defined in docs/features.md
- Include author_level variance (junior/senior/non_tech)

Load with:
    from tasks.comprehensive_tasks import COMPREHENSIVE_TASKS
    for task in COMPREHENSIVE_TASKS:
        print(task.task_id, task.expected_verdict)

Or serialise to JSONL:
    python tasks/comprehensive_tasks.py --output tasks/comprehensive_tasks.jsonl
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pr_review_env.server.tasks import PRTask


COMPREHENSIVE_TASKS: list[PRTask] = [

    # ── 1. JWT validation removed in auth refactor ───────────────────────────
    PRTask(
        task_id="comp_auth_jwt_removed",
        pr_description="Refactor: simplify auth middleware by removing redundant token checks",
        diff_str="""\
diff --git a/src/auth/middleware.py b/src/auth/middleware.py
index aaa0001..bbb0002 100644
--- a/src/auth/middleware.py
+++ b/src/auth/middleware.py
@@ -12,16 +12,10 @@ import jwt
 class JWTMiddleware:
     def __init__(self, secret_key: str):
         self.secret_key = secret_key
-        self.verify_signature = True
-        self.verify_expiry = True

     async def __call__(self, request: Request, call_next):
         token = request.headers.get("Authorization", "").removeprefix("Bearer ")
         if not token:
             return JSONResponse({"error": "Unauthorized"}, status_code=401)
-        try:
-            payload = jwt.decode(token, self.secret_key, algorithms=["RS256"],
-                                 options={"verify_signature": True, "verify_exp": True})
-        except jwt.ExpiredSignatureError:
-            return JSONResponse({"error": "Token expired"}, status_code=401)
-        except jwt.InvalidTokenError:
-            return JSONResponse({"error": "Invalid token"}, status_code=401)
+        payload = jwt.decode(token, options={"verify_signature": False})
         request.state.user = payload
         return await call_next(request)

diff --git a/src/auth/models.py b/src/auth/models.py
index ccc0003..ddd0004 100644
--- a/src/auth/models.py
+++ b/src/auth/models.py
@@ -5,7 +5,6 @@ from pydantic import BaseModel
 class TokenPayload(BaseModel):
     user_id: str
     email: str
-    exp: int
     roles: list[str] = []

diff --git a/tests/test_auth.py b/tests/test_auth.py
index eee0005..fff0006 100644
--- a/tests/test_auth.py
+++ b/tests/test_auth.py
@@ -18,8 +18,3 @@ def test_valid_token_passes():
     response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
     assert response.status_code == 200

-def test_expired_token_rejected():
-    expired_token = create_token(expires_delta=timedelta(seconds=-1))
-    response = client.get("/protected", headers={"Authorization": f"Bearer {expired_token}"})
-    assert response.status_code == 401
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="reject",
        risk_domains=["security"],
        difficulty="hard",
        review_goal="JWT signature verification disabled — any token accepted including forged ones. Expiry check also removed.",
        author_level="junior",
        ownership_hint="auth",
    ),

    # ── 2. SQL injection via ORM bypass in payments ───────────────────────────
    PRTask(
        task_id="comp_payments_raw_sql",
        pr_description="Perf: replace ORM query in payment search with raw SQL for speed",
        diff_str="""\
diff --git a/src/payments/repository.py b/src/payments/repository.py
index aaa1001..bbb1002 100644
--- a/src/payments/repository.py
+++ b/src/payments/repository.py
@@ -8,12 +8,14 @@ from sqlalchemy.ext.asyncio import AsyncSession
 class PaymentRepository:
     def __init__(self, db: AsyncSession):
         self.db = db

-    async def search(self, user_id: str, query: str) -> list[Payment]:
-        return await self.db.execute(
-            select(Payment).where(Payment.user_id == user_id,
-                                  Payment.description.ilike(f"%{query}%"))
-        ).scalars().all()
+    async def search(self, user_id: str, query: str) -> list[Payment]:
+        sql = f"SELECT * FROM payments WHERE user_id = '{user_id}' AND description LIKE '%{query}%'"
+        result = await self.db.execute(text(sql))
+        return result.fetchall()

diff --git a/src/payments/service.py b/src/payments/service.py
index ccc1003..ddd1004 100644
--- a/src/payments/service.py
+++ b/src/payments/service.py
@@ -2,6 +2,7 @@ from src.payments.repository import PaymentRepository
+from sqlalchemy import text

 class PaymentService:
     async def search_payments(self, user_id: str, query: str):
+        # TODO: add input sanitisation later
         return await self.repo.search(user_id, query)
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="reject",
        risk_domains=["security"],
        difficulty="medium",
        review_goal="Raw SQL with f-string interpolation on user input — SQL injection in the payments critical path.",
        author_level="mid",
        ownership_hint="payments",
    ),

    # ── 3. Large Go service refactor — goroutine leak ─────────────────────────
    PRTask(
        task_id="comp_go_goroutine_leak",
        pr_description="Refactor: improve worker pool concurrency in analytics service",
        diff_str="""\
diff --git a/analytics/worker.go b/analytics/worker.go
index aaa2001..bbb2002 100644
--- a/analytics/worker.go
+++ b/analytics/worker.go
@@ -14,22 +14,24 @@ type WorkerPool struct {
 func NewWorkerPool(size int) *WorkerPool {
     return &WorkerPool{size: size, jobs: make(chan Job, 100)}
 }

-func (wp *WorkerPool) Start(ctx context.Context) {
+func (wp *WorkerPool) Start() {
     for i := 0; i < wp.size; i++ {
-        go func() {
-            for {
-                select {
-                case job, ok := <-wp.jobs:
-                    if !ok {
-                        return
-                    }
-                    job.Execute()
-                case <-ctx.Done():
-                    return
-                }
-            }
-        }()
+        go func() {
+            for job := range wp.jobs {
+                job.Execute()
+            }
+        }()
     }
 }

diff --git a/analytics/server.go b/analytics/server.go
index ccc2003..ddd2004 100644
--- a/analytics/server.go
+++ b/analytics/server.go
@@ -8,7 +8,7 @@ func StartServer(cfg Config) {
     pool := NewWorkerPool(cfg.Workers)
-    pool.Start(ctx)
+    pool.Start()
     defer pool.Stop()

diff --git a/analytics/worker_test.go b/analytics/worker_test.go
index eee2005..fff2006 100644
--- a/analytics/worker_test.go
+++ b/analytics/worker_test.go
@@ -22,12 +22,6 @@ func TestWorkerPool(t *testing.T) {
     pool.Submit(job)
     time.Sleep(100 * time.Millisecond)
     assert.Equal(t, 1, executed)
-
-    // Test graceful shutdown
-    cancel()
-    time.Sleep(50 * time.Millisecond)
-    // Workers should have exited
-    assert.Equal(t, 0, len(pool.jobs))
 }
""",
        primary_language="go",
        changed_file_types=["go"],
        repo_kind="backend_service",
        expected_verdict="request_changes",
        risk_domains=["quality", "build"],
        difficulty="hard",
        review_goal="Context removed from worker goroutines — they cannot be cancelled on shutdown, causing goroutine leak. Shutdown test deleted.",
        author_level="mid",
        ownership_hint="analytics",
    ),

    # ── 4. Multi-file TypeScript: `any` types + missing validation ────────────
    PRTask(
        task_id="comp_ts_any_types_api",
        pr_description="Feature: add bulk payment import endpoint and frontend form",
        diff_str="""\
diff --git a/src/api/bulk_import.ts b/src/api/bulk_import.ts
index aaa3001..bbb3002 100644
--- /dev/null
+++ b/src/api/bulk_import.ts
@@ -0,0 +1,28 @@
+import axios from "axios"
+
+export async function importPayments(file: File): Promise<any> {
+  const form = new FormData()
+  form.append("file", file)
+  const response = await axios.post("/api/v1/payments/bulk", form, {
+    headers: { "Content-Type": "multipart/form-data" },
+  })
+  return response.data  // no schema validation
+}
+
+export function parseImportResult(result: any): any {
+  return {
+    imported: result.imported_count,
+    failed: result.failed_rows,
+    errors: result.errors,
+  }
+}

diff --git a/src/components/BulkImport.tsx b/src/components/BulkImport.tsx
index ccc3003..ddd3004 100644
--- /dev/null
+++ b/src/components/BulkImport.tsx
@@ -0,0 +1,34 @@
+import React, { useState } from "react"
+import { importPayments, parseImportResult } from "../api/bulk_import"
+
+export function BulkImport() {
+  const [result, setResult] = useState<any>(null)
+  const [error, setError] = useState<any>(null)
+
+  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
+    const file = e.target.files?.[0]
+    if (!file) return
+    try {
+      const raw = await importPayments(file)
+      setResult(parseImportResult(raw))
+    } catch (err: any) {
+      setError(err.message)
+    }
+  }
+
+  return (
+    <div>
+      <input type="file" accept=".csv,.xlsx" onChange={handleUpload} />
+      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
+      {error && <div className="error">{error}</div>}
+    </div>
+  )
+}

diff --git a/backend/bulk_import.py b/backend/bulk_import.py
index eee3005..fff3006 100644
--- /dev/null
+++ b/backend/bulk_import.py
@@ -0,0 +1,22 @@
+from fastapi import APIRouter, UploadFile, File
+
+router = APIRouter()
+
+@router.post("/payments/bulk")
+async def bulk_import(file: UploadFile = File(...)):
+    content = await file.read()
+    # TODO: validate file type from content, not Content-Type header
+    rows = content.decode("utf-8").splitlines()
+    results = []
+    for row in rows:
+        parts = row.split(",")
+        amount = int(parts[2])  # no validation — IndexError if malformed
+        results.append({"amount": amount, "status": "queued"})
+    return {"imported_count": len(results), "failed_rows": 0, "errors": []}
""",
        primary_language="typescript",
        changed_file_types=["typescript", "python"],
        repo_kind="backend_service",
        expected_verdict="request_changes",
        risk_domains=["security", "quality"],
        difficulty="medium",
        review_goal="Multiple `any` types in TypeScript, no Zod validation, backend has no file type check or input validation.",
        author_level="junior",
        ownership_hint="frontend,payments",
    ),

    # ── 5. Dockerfile + GHA: secrets in build args + write permissions ────────
    PRTask(
        task_id="comp_infra_secrets_exposed",
        pr_description="CI: add production deploy workflow and update Dockerfile for cloud build",
        diff_str="""\
diff --git a/Dockerfile b/Dockerfile
index aaa4001..bbb4002 100644
--- a/Dockerfile
+++ b/Dockerfile
@@ -1,12 +1,16 @@
 FROM python:3.11-slim
+
+ARG DATABASE_URL
+ARG SECRET_KEY
+ARG STRIPE_SECRET_KEY
+
+ENV DATABASE_URL=${DATABASE_URL}
+ENV SECRET_KEY=${SECRET_KEY}
+ENV STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
+
 WORKDIR /app
 COPY requirements.txt .
 RUN pip install --no-cache-dir -r requirements.txt
 COPY . .
-USER nonroot
+RUN useradd -m appuser
 EXPOSE 8000
 CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

diff --git a/.github/workflows/deploy.yml b/.github/workflows/deploy.yml
index ccc4003..ddd4004 100644
--- /dev/null
+++ b/.github/workflows/deploy.yml
@@ -0,0 +1,34 @@
+name: Deploy to Production
+on:
+  push:
+    branches: [main]
+
+permissions:
+  contents: write
+  packages: write
+  deployments: write
+  id-token: write
+
+jobs:
+  deploy:
+    runs-on: ubuntu-latest
+    steps:
+      - uses: actions/checkout@v4
+      - name: Build image
+        run: |
+          docker build \
+            --build-arg DATABASE_URL=${{ secrets.DATABASE_URL }} \
+            --build-arg SECRET_KEY=${{ secrets.SECRET_KEY }} \
+            --build-arg STRIPE_SECRET_KEY=${{ secrets.STRIPE_SECRET_KEY }} \
+            -t myapp:latest .
+      - name: Push to registry
+        run: docker push myapp:latest
""",
        primary_language="python",
        changed_file_types=["dockerfile", "github_actions", "yaml"],
        repo_kind="infra",
        expected_verdict="reject",
        risk_domains=["security", "config"],
        difficulty="hard",
        review_goal="Secrets baked into Docker image via ARG/ENV — visible in image layers. USER directive removed. Overly broad GHA permissions.",
        author_level="non_tech",
        ownership_hint="infra",
    ),

    # ── 6. Rust: unwrap() on I/O across multiple modules ─────────────────────
    PRTask(
        task_id="comp_rust_unwrap_io_multi",
        pr_description="Refactor: split file processing into separate modules for clarity",
        diff_str="""\
diff --git a/src/reader.rs b/src/reader.rs
index aaa5001..bbb5002 100644
--- /dev/null
+++ b/src/reader.rs
@@ -0,0 +1,18 @@
+use std::fs;
+use std::path::Path;
+
+pub fn read_config(path: &Path) -> String {
+    fs::read_to_string(path).unwrap()
+}
+
+pub fn read_all_files(dir: &Path) -> Vec<String> {
+    fs::read_dir(dir).unwrap()
+        .map(|entry| {
+            let path = entry.unwrap().path();
+            fs::read_to_string(&path).unwrap()
+        })
+        .collect()
+}

diff --git a/src/writer.rs b/src/writer.rs
index ccc5003..ddd5004 100644
--- /dev/null
+++ b/src/writer.rs
@@ -0,0 +1,12 @@
+use std::fs;
+use std::path::Path;
+
+pub fn write_output(path: &Path, content: &str) {
+    fs::write(path, content).unwrap()
+}
+
+pub fn append_log(path: &Path, line: &str) {
+    let mut f = fs::OpenOptions::new().append(true).open(path).unwrap();
+    use std::io::Write;
+    writeln!(f, "{}", line).unwrap()
+}

diff --git a/src/main.rs b/src/main.rs
index eee5005..fff5006 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,6 +1,9 @@
+mod reader;
+mod writer;
+
 fn main() {
-    run().expect("fatal error");
+    let config = reader::read_config(std::path::Path::new("config.toml"));
+    println!("{}", config);
 }
""",
        primary_language="rust",
        changed_file_types=["rust"],
        repo_kind="library",
        expected_verdict="request_changes",
        risk_domains=["quality", "build"],
        difficulty="medium",
        review_goal="unwrap() on all file I/O — any missing file panics the process. Use Result<> and ? operator instead.",
        author_level="junior",
        ownership_hint="backend",
    ),

    # ── 7. Python: large Django migration — irreversible data change ──────────
    PRTask(
        task_id="comp_django_irreversible_migration",
        pr_description="Schema: merge first_name + last_name into full_name column for CRM integration",
        diff_str="""\
diff --git a/src/users/models.py b/src/users/models.py
index aaa6001..bbb6002 100644
--- a/src/users/models.py
+++ b/src/users/models.py
@@ -8,8 +8,7 @@ class User(models.Model):
     email = models.EmailField(unique=True)
-    first_name = models.CharField(max_length=100)
-    last_name = models.CharField(max_length=100)
+    full_name = models.CharField(max_length=200)
     created_at = models.DateTimeField(auto_now_add=True)

diff --git a/db/migrations/0042_merge_name_fields.py b/db/migrations/0042_merge_name_fields.py
index ccc6003..ddd6004 100644
--- /dev/null
+++ b/db/migrations/0042_merge_name_fields.py
@@ -0,0 +1,28 @@
+from django.db import migrations, models
+
+def merge_names(apps, schema_editor):
+    User = apps.get_model("users", "User")
+    for user in User.objects.all():
+        user.full_name = f"{user.first_name} {user.last_name}"
+        user.save()
+
+class Migration(migrations.Migration):
+    dependencies = [("users", "0041_add_crm_id")]
+
+    operations = [
+        migrations.AddField(
+            model_name="user",
+            name="full_name",
+            field=models.CharField(max_length=200, default=""),
+        ),
+        migrations.RunPython(merge_names),   # data migration: no reverse
+        migrations.RemoveField(model_name="user", name="first_name"),
+        migrations.RemoveField(model_name="user", name="last_name"),
+    ]

diff --git a/src/users/serializers.py b/src/users/serializers.py
index eee6005..fff6006 100644
--- a/src/users/serializers.py
+++ b/src/users/serializers.py
@@ -6,5 +6,4 @@ class UserSerializer(serializers.ModelSerializer):
     class Meta:
         model = User
-        fields = ["id", "email", "first_name", "last_name", "created_at"]
+        fields = ["id", "email", "full_name", "created_at"]
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="request_changes",
        risk_domains=["quality", "build"],
        difficulty="hard",
        review_goal="Irreversible migration — RunPython with no reverse function, and columns are deleted. Data loss if rollback needed.",
        author_level="mid",
        ownership_hint="backend",
    ),

    # ── 8. Java: null safety removed + unchecked exception swallowed ──────────
    PRTask(
        task_id="comp_java_null_exception",
        pr_description="Refactor: simplify payment processor to reduce boilerplate",
        diff_str="""\
diff --git a/src/main/java/payments/PaymentProcessor.java b/src/main/java/payments/PaymentProcessor.java
index aaa7001..bbb7002 100644
--- a/src/main/java/payments/PaymentProcessor.java
+++ b/src/main/java/payments/PaymentProcessor.java
@@ -18,24 +18,16 @@ public class PaymentProcessor {

-    public PaymentResult charge(String customerId, int amountCents, String currency) {
-        Objects.requireNonNull(customerId, "customerId must not be null");
-        Objects.requireNonNull(currency, "currency must not be null");
-        if (amountCents <= 0) {
-            throw new IllegalArgumentException("Amount must be positive, got: " + amountCents);
-        }
+    public PaymentResult charge(String customerId, int amountCents, String currency) {
         try {
             StripeCharge charge = stripe.charges().create(
                 customerId, amountCents, currency
             );
-            if (charge == null) {
-                throw new PaymentException("Stripe returned null charge");
-            }
             return PaymentResult.success(charge.getId());
-        } catch (StripeException e) {
-            log.error("Stripe charge failed for customer {}: {}", customerId, e.getMessage());
-            throw new PaymentException("Charge failed: " + e.getMessage(), e);
+        } catch (Exception e) {
+            return PaymentResult.failure("error");
         }
     }

diff --git a/src/test/java/payments/PaymentProcessorTest.java b/src/test/java/payments/PaymentProcessorTest.java
index ccc7003..ddd7004 100644
--- a/src/test/java/payments/PaymentProcessorTest.java
+++ b/src/test/java/payments/PaymentProcessorTest.java
@@ -30,10 +30,4 @@ class PaymentProcessorTest {
     void testChargeSuccess() {
         PaymentResult result = processor.charge("cus_123", 1000, "USD");
         assertTrue(result.isSuccess());
     }
-
-    void testNullCustomerIdThrows() {
-        assertThrows(NullPointerException.class, () ->
-            processor.charge(null, 1000, "USD"));
-    }
""",
        primary_language="java",
        changed_file_types=["java"],
        repo_kind="backend_service",
        expected_verdict="request_changes",
        risk_domains=["quality", "security"],
        difficulty="medium",
        review_goal="Null checks removed, bare Exception caught and swallowed with generic 'error' — Stripe failures now silent. Tests for null safety deleted.",
        author_level="junior",
        ownership_hint="payments",
    ),

    # ── 9. Go: nil pointer after interface change in middleware ───────────────
    PRTask(
        task_id="comp_go_nil_middleware",
        pr_description="Refactor: update request context to carry typed user struct instead of map",
        diff_str="""\
diff --git a/middleware/auth.go b/middleware/auth.go
index aaa8001..bbb8002 100644
--- a/middleware/auth.go
+++ b/middleware/auth.go
@@ -14,12 +14,10 @@ type contextKey string
 const userKey contextKey = "user"

 func AuthMiddleware(next http.Handler) http.Handler {
     return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
         token := r.Header.Get("Authorization")
         user, err := validateToken(token)
         if err != nil {
             http.Error(w, "Unauthorized", http.StatusUnauthorized)
             return
         }
-        ctx := context.WithValue(r.Context(), userKey, user)
+        ctx := context.WithValue(r.Context(), userKey, &user)
         next.ServeHTTP(w, r.WithContext(ctx))
     })
 }

diff --git a/handlers/payment.go b/handlers/payment.go
index ccc8003..ddd8004 100644
--- a/handlers/payment.go
+++ b/handlers/payment.go
@@ -8,9 +8,8 @@ func CreatePayment(w http.ResponseWriter, r *http.Request) {
-    user, ok := r.Context().Value(userKey).(User)
-    if !ok {
-        http.Error(w, "context error", 500)
-        return
-    }
+    user := r.Context().Value(userKey).(*User)
+    // user is guaranteed by middleware
     amount := r.FormValue("amount")
     processPayment(user.ID, amount)
 }
""",
        primary_language="go",
        changed_file_types=["go"],
        repo_kind="backend_service",
        expected_verdict="request_changes",
        risk_domains=["quality", "security"],
        difficulty="hard",
        review_goal="Type assertion without ok check — if context value is missing or wrong type, panics at runtime. Auth bypass could also trigger nil dereference.",
        author_level="mid",
        ownership_hint="backend",
    ),

    # ── 10. Python: logging change exposes PII ────────────────────────────────
    PRTask(
        task_id="comp_py_logging_pii",
        pr_description="Debug: improve logging in auth service to help diagnose login failures",
        diff_str="""\
diff --git a/src/auth/service.py b/src/auth/service.py
index aaa9001..bbb9002 100644
--- a/src/auth/service.py
+++ b/src/auth/service.py
@@ -22,18 +22,18 @@ class AuthService:
     async def login(self, email: str, password: str) -> TokenPair:
-        logger.info("Login attempt", extra={"email_hash": hash(email)})
+        logger.info(f"Login attempt for {email} with password {password}")
         user = await self.user_repo.find_by_email(email)
         if not user:
-            logger.warning("Login failed: user not found", extra={"email_hash": hash(email)})
+            logger.warning(f"User not found: {email}")
             raise AuthError("Invalid credentials")
         if not verify_password(password, user.hashed_password):
-            logger.warning("Login failed: bad password", extra={"user_id": user.id})
+            logger.warning(f"Bad password for {email}: supplied={password}, stored={user.hashed_password}")
             raise AuthError("Invalid credentials")
         token = self.create_token(user)
-        logger.info("Login success", extra={"user_id": user.id})
+        logger.info(f"Login success for {email}, token={token}")
         return token

diff --git a/src/auth/reset.py b/src/auth/reset.py
index ccc9003..ddd9004 100644
--- a/src/auth/reset.py
+++ b/src/auth/reset.py
@@ -14,7 +14,7 @@ async def request_password_reset(email: str):
     token = generate_reset_token()
     await send_reset_email(email, token)
-    logger.info("Reset token sent", extra={"email_hash": hash(email)})
+    logger.info(f"Password reset token for {email}: {token}")
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="reject",
        risk_domains=["security"],
        difficulty="medium",
        review_goal="Plaintext passwords, hashed passwords, auth tokens, and PII (emails) all logged. This is a critical credential exposure.",
        author_level="junior",
        ownership_hint="auth",
    ),

    # ── 11. GitHub Actions + YAML: k8s resource limits removed ───────────────
    PRTask(
        task_id="comp_k8s_resource_limits_removed",
        pr_description="Infra: update k8s deployment manifests for v2 rollout",
        diff_str="""\
diff --git a/infrastructure/k8s/payments-deployment.yaml b/infrastructure/k8s/payments-deployment.yaml
index aaaa001..bbbb002 100644
--- a/infrastructure/k8s/payments-deployment.yaml
+++ b/infrastructure/k8s/payments-deployment.yaml
@@ -24,16 +24,8 @@ spec:
         containers:
         - name: payments
           image: myapp/payments:latest
-          resources:
-            requests:
-              cpu: "250m"
-              memory: "512Mi"
-            limits:
-              cpu: "1000m"
-              memory: "1Gi"
           env:
           - name: DATABASE_URL
             valueFrom:
               secretKeyRef:
                 name: payments-secrets
                 key: database_url
-          livenessProbe:
-            httpGet:
-              path: /health
-              port: 8000
-            initialDelaySeconds: 10

diff --git a/.github/workflows/k8s-deploy.yml b/.github/workflows/k8s-deploy.yml
index cccc003..dddd004 100644
--- a/.github/workflows/k8s-deploy.yml
+++ b/.github/workflows/k8s-deploy.yml
@@ -4,6 +4,10 @@ on:
   push:
     branches: [main]

+permissions:
+  contents: write
+  id-token: write
+
 jobs:
   deploy:
     runs-on: ubuntu-latest
""",
        primary_language="python",
        changed_file_types=["yaml", "github_actions"],
        repo_kind="infra",
        expected_verdict="request_changes",
        risk_domains=["config", "security"],
        difficulty="medium",
        review_goal="Resource limits and liveness probe removed from payments pod — OOM kills + no health gate. GHA gains unnecessary write permissions.",
        author_level="non_tech",
        ownership_hint="infra",
    ),

    # ── 12. Python: rate limiting middleware removed ──────────────────────────
    PRTask(
        task_id="comp_py_rate_limit_removed",
        pr_description="Refactor: simplify API middleware stack, rate limiting moved to nginx",
        diff_str="""\
diff --git a/src/main.py b/src/main.py
index aaab001..bbab002 100644
--- a/src/main.py
+++ b/src/main.py
@@ -12,12 +12,8 @@ from src.middleware.auth import JWTMiddleware
-from src.middleware.rate_limit import RateLimitMiddleware
 from src.middleware.cors import CORSMiddleware

 app = FastAPI()

 app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS)
-app.add_middleware(RateLimitMiddleware,
-                   limit=100, window_seconds=60, by="user_id")
 app.add_middleware(JWTMiddleware, secret_key=settings.SECRET_KEY)

diff --git a/src/middleware/rate_limit.py b/src/middleware/rate_limit.py
deleted file mode 100644
index ccab003..0000000
--- a/src/middleware/rate_limit.py
+++ /dev/null
@@ -1,42 +0,0 @@
-import time
-import redis
-from starlette.middleware.base import BaseHTTPMiddleware
-
-class RateLimitMiddleware(BaseHTTPMiddleware):
-    def __init__(self, app, limit: int, window_seconds: int, by: str = "user_id"):
-        super().__init__(app)
-        self.limit = limit
-        self.window = window_seconds
-        self.by = by
-        self.redis = redis.Redis.from_url(settings.REDIS_URL)
-
-    async def dispatch(self, request, call_next):
-        key = f"ratelimit:{request.state.user.get(self.by, 'anon')}"
-        count = self.redis.incr(key)
-        if count == 1:
-            self.redis.expire(key, self.window)
-        if count > self.limit:
-            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
-        return await call_next(request)

diff --git a/nginx/nginx.conf b/nginx/nginx.conf
index ddab004..eeab005 100644
--- a/nginx/nginx.conf
+++ b/nginx/nginx.conf
@@ -18,4 +18,3 @@ server {
     location /api/ {
         proxy_pass http://app:8000;
-        # TODO: add limit_req here
     }
 }
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="reject",
        risk_domains=["security", "config"],
        difficulty="medium",
        review_goal="Rate limiting removed from app layer, nginx TODO comment shows it hasn't been added there either — API is now unprotected against abuse.",
        author_level="mid",
        ownership_hint="backend",
    ),

    # ── 13. Clean large refactor — should approve ─────────────────────────────
    PRTask(
        task_id="comp_clean_analytics_refactor",
        pr_description="Refactor: split monolithic analytics module into focused sub-modules",
        diff_str="""\
diff --git a/src/analytics/__init__.py b/src/analytics/__init__.py
index aaac001..bbac002 100644
--- a/src/analytics/__init__.py
+++ b/src/analytics/__init__.py
@@ -1,3 +1,6 @@
+from .aggregator import Aggregator
+from .reporter import Reporter
+from .exporter import Exporter

diff --git a/src/analytics/aggregator.py b/src/analytics/aggregator.py
index ccac003..ddac004 100644
--- /dev/null
+++ b/src/analytics/aggregator.py
@@ -0,0 +1,35 @@
+# Aggregates time-series data into bucketed summaries.
+from __future__ import annotations
+from datetime import datetime
+from typing import Sequence
+
+class Aggregator:
+    # Groups raw events into hourly/daily/monthly buckets.
+
+    def hourly(self, events: Sequence[dict]) -> dict:
+        buckets: dict[str, list] = {}
+        for event in events:
+            hour = datetime.fromisoformat(event["ts"]).strftime("%Y-%m-%dT%H:00")
+            buckets.setdefault(hour, []).append(event["value"])
+        return {k: sum(v) / len(v) for k, v in buckets.items()}

diff --git a/src/analytics/reporter.py b/src/analytics/reporter.py
index eeac005..ffac006 100644
--- /dev/null
+++ b/src/analytics/reporter.py
@@ -0,0 +1,28 @@
+# Generates structured reports from aggregated data.
+from __future__ import annotations
+from dataclasses import dataclass
+
+@dataclass
+class Report:
+    title: str
+    period: str
+    metrics: dict[str, float]
+
+class Reporter:
+    def build(self, title: str, period: str, data: dict) -> Report:
+        return Report(title=title, period=period, metrics=data)

diff --git a/tests/test_aggregator.py b/tests/test_aggregator.py
index ggac007..hhac008 100644
--- /dev/null
+++ b/tests/test_aggregator.py
@@ -0,0 +1,18 @@
+from src.analytics.aggregator import Aggregator
+
+def test_hourly_bucketing():
+    agg = Aggregator()
+    events = [
+        {"ts": "2026-04-24T14:05:00", "value": 10},
+        {"ts": "2026-04-24T14:30:00", "value": 20},
+        {"ts": "2026-04-24T15:00:00", "value": 5},
+    ]
+    result = agg.hourly(events)
+    assert result["2026-04-24T14:00"] == 15.0
+    assert result["2026-04-24T15:00"] == 5.0
""",
        primary_language="python",
        changed_file_types=["python"],
        repo_kind="backend_service",
        expected_verdict="approve",
        risk_domains=[],
        difficulty="easy",
        review_goal="Clean module split with docstrings, type hints, and tests. No security or quality issues — should approve efficiently.",
        author_level="senior",
        ownership_hint="analytics",
    ),
]


def to_jsonl(tasks: list[PRTask], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(t.to_dict()) for t in tasks]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tasks/comprehensive_tasks.jsonl")
    args = parser.parse_args()
    to_jsonl(COMPREHENSIVE_TASKS, args.output)
