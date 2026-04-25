__all__ = ["PRReviewEnv"]


def __getattr__(name: str):
    if name == "PRReviewEnv":
        from .pr_review_env import PRReviewEnv

        return PRReviewEnv
    raise AttributeError(name)
