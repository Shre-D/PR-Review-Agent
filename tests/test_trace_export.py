from train.export_teacher_traces import build_prompt, export_teacher_traces


def test_export_teacher_traces_smoke():
    traces = export_teacher_traces(limit=2)
    assert len(traces) == 2
    assert all("trajectory" in item for item in traces)
    assert all("state_actions" in item for item in traces)
    assert all("predicted_verdict" in item for item in traces)
    assert any(
        state["phase"] == "terminal_ready"
        for trace in traces
        for state in trace["state_actions"]
    )


def test_prompt_contains_diff_context():
    trace = export_teacher_traces(limit=1)[0]
    prompt = trace["prompt"]
    assert "Language:" in prompt
    assert "Diff (truncated" in prompt
