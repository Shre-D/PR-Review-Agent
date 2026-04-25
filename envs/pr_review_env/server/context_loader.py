from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..models import ReviewConfig


DEFAULT_CONFIG: dict[str, Any] = ReviewConfig().model_dump()
DEFAULT_CONFIG.update({
    "tool_weights": {},
    "enabled_tools": [],
    "planned_tools": [],
    "custom_rules": [],
    "extraction_method": "structural",
})


def _parse_numeric_list_item(line: str) -> tuple[str, float] | None:
    if not line.startswith("-") or ":" not in line:
        return None
    key, value = line.lstrip("- ").split(":", 1)
    try:
        return key.strip(), float(value.strip())
    except ValueError:
        return None


def _architecture_summary(docs_dir: Path) -> str:
    architecture = docs_dir / "architecture.md"
    if not architecture.exists():
        return ""

    paragraphs: list[str] = []
    current: list[str] = []
    in_overview = False
    for raw_line in architecture.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if in_overview and current:
                paragraphs.append(" ".join(current))
            if in_overview:
                break
            in_overview = line == "## Overview"
            current = []
            continue
        if not in_overview:
            continue
        if not line or line.startswith("#"):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("-") or line.startswith("|") or line.startswith("```"):
            continue
        current.append(line)
        if len(paragraphs) >= 2:
            break
    if current:
        paragraphs.append(" ".join(current))

    return " ".join(paragraphs[:2])


def extract_structural_config(docs_dir: str | Path) -> dict[str, Any]:
    docs_path = Path(docs_dir)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["architecture_summary"] = _architecture_summary(docs_path)

    review_tool = docs_path / "review-tool.md"
    if not review_tool.exists():
        return ReviewConfig.model_validate(config).model_dump()

    section: str | None = None
    for raw_line in review_tool.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("## Tool Weights"):
            section = "tool_weights"
            continue
        if line.startswith("## Verdict Thresholds"):
            section = "thresholds"
            continue
        if line.startswith("## Domain Priorities"):
            section = "domain_priorities"
            continue
        if line.startswith("## Critical Paths"):
            section = "critical_paths"
            continue
        if line.startswith("## Author Depth Overrides"):
            section = "author_depth"
            continue
        if line.startswith("## Escalation Confidence Threshold"):
            section = "escalation"
            continue
        if line.startswith("## Enabled External Tools") or line.startswith("## Implemented External Tools"):
            section = "enabled_tools"
            continue
        if line.startswith("## Planned External Tools"):
            section = "planned_tools"
            continue
        if line.startswith("##"):
            section = None
            continue

        if section == "tool_weights" and "|" in line and "check_" in line:
            parts = [part.strip() for part in line.split("|") if part.strip()]
            if len(parts) == 2:
                try:
                    config["tool_weights"][parts[0]] = float(parts[1])
                except ValueError:
                    pass
        elif section == "thresholds":
            parsed = _parse_numeric_list_item(line)
            if parsed:
                key, value = parsed
                if key in {"reject_threshold", "request_changes_threshold"}:
                    config[key] = value
        elif section == "domain_priorities":
            parsed = _parse_numeric_list_item(line)
            if parsed:
                key, value = parsed
                config["domain_priorities"][key] = value
        elif section == "critical_paths" and line.startswith("-"):
            config["critical_paths"].append(line.lstrip("- ").strip())
        elif section == "author_depth":
            parsed = _parse_numeric_list_item(line)
            if parsed:
                key, value = parsed
                config["author_depth"][key] = value
        elif section == "escalation":
            parsed = _parse_numeric_list_item(line)
            if parsed and parsed[0] == "escalation_confidence_threshold":
                config["escalation_confidence_threshold"] = parsed[1]
        elif section == "enabled_tools" and line.startswith("-"):
            config["enabled_tools"].append(line.lstrip("- ").strip())
        elif section == "planned_tools" and line.startswith("-"):
            config["planned_tools"].append(line.lstrip("- ").strip())

    return ReviewConfig.model_validate(config).model_dump()


def load_review_config(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReviewConfig.model_validate(payload).model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review_config.json from org documentation.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--output", default="review_config.json")
    parser.add_argument(
        "--extract-rules",
        action="store_true",
        help="Accepted for workflow compatibility; current implementation is structural only.",
    )
    args = parser.parse_args()

    config = extract_structural_config(args.docs_dir)
    if args.extract_rules:
        config["extraction_method"] = "structural"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "enabled_tools": len(config["enabled_tools"])}))


if __name__ == "__main__":
    main()
