from envs.pr_review_env.models import (
    AuthorContext,
    CustomRule,
    Finding,
    PRReviewObservation,
    PRReviewVerdict,
    ReviewConfig,
)


def test_rich_review_models_instantiate():
    finding = Finding(
        tool_name="check_security",
        domain="security",
        severity="critical",
        message="Unsafe SQL construction",
    )
    verdict = PRReviewVerdict(
        verdict="reject",
        confidence=0.91,
        summary="Security issue found.",
        critical_findings=[finding],
        tools_used=["check_security"],
        aggregate_score=0.55,
    )
    config = ReviewConfig(
        custom_rules=[
            CustomRule(
                domain="security",
                pattern="execute\\(",
                severity="critical",
                message="Raw SQL execution",
            )
        ]
    )
    obs = PRReviewObservation(
        author_context=AuthorContext(level="junior"),
        critical_paths_touched=["src/auth/"],
        estimated_risk_level="high",
        final_verdict=verdict,
    )

    assert config.custom_rules[0].severity == "critical"
    assert obs.author_context.level == "junior"
    assert obs.final_verdict is not None
    assert obs.final_verdict.critical_findings[0].domain == "security"
