import json

from envs.pr_review_env.models import PRReviewObservation
from inference import is_ollama_model, ollama_model_name, run_ollama_step


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_model_prefix_helpers():
    assert is_ollama_model("ollama:qwen2.5:7b") is True
    assert is_ollama_model("Qwen/Qwen3-1.7B") is False
    assert ollama_model_name("ollama:qwen2.5:7b") == "qwen2.5:7b"


def test_run_ollama_step_posts_chat_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "message": {
                    "content": (
                        '{"tool_name":"submit_review","arguments":'
                        '{"verdict":"approve","confidence":0.8,"reasoning":"ok"}}'
                    )
                }
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    action = run_ollama_step(
        "qwen2.5:7b",
        PRReviewObservation(
            primary_language="python",
            changed_file_types=["python"],
            repo_kind="service",
            pr_description="Small cleanup",
            diff_str="+x = 1",
        ),
        host="http://localhost:11434",
    )

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 120
    assert captured["payload"]["model"] == "qwen2.5:7b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0
    assert action.tool_name == "submit_review"
    assert action.arguments["verdict"] == "approve"
