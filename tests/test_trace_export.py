from train.export_teacher_traces import build_prompt, export_teacher_traces


def test_export_teacher_traces_smoke():
    traces = export_teacher_traces(limit=2)
    assert len(traces) == 2
    assert all("trajectory" in item for item in traces)
    assert all("predicted_verdict" in item for item in traces)


def test_prompt_contains_diff_context():
    trace = export_teacher_traces(limit=1)[0]
    prompt = trace["prompt"]
    assert "Primary language:" in prompt
    assert "Diff:" in prompt
