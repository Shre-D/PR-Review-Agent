from __future__ import annotations

from ..compat import create_fastapi_app
from ..models import PRReviewAction, PRReviewObservation
from .pr_review_env import PRReviewEnv


app = create_fastapi_app(PRReviewEnv, PRReviewAction, PRReviewObservation)
