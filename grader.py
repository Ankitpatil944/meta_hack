from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from models import ActionType, GraderResponse, HistoryEntry
from tasks import TaskSpec


PATCH_FILE_HEADER = re.compile(r"^\+\+\+\s+b/(.+)$")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def keyword_hit_count(content: str, keywords: List[str]) -> int:
    normalized = normalize_text(content)
    return sum(1 for keyword in keywords if keyword in normalized)


def bug_identification_matches(task: TaskSpec, content: str) -> bool:
    hits = keyword_hit_count(content, task.expected_bug_keywords)
    required_hits = 2 if task.difficulty == "easy" else 3
    return hits >= required_hits


def fix_explanation_matches(task: TaskSpec, content: str) -> bool:
    hits = keyword_hit_count(content, task.expected_fix_keywords)
    return hits >= 2


def _extract_patch_sections(patch_text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_file: Optional[str] = None
    current_lines: List[str] = []
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("+++ b/"):
            if current_file is not None:
                sections[current_file] = current_lines
            match = PATCH_FILE_HEADER.match(raw_line)
            current_file = match.group(1) if match else None
            current_lines = []
            continue
        if current_file is not None:
            current_lines.append(raw_line)
    if current_file is not None:
        sections[current_file] = current_lines
    return sections


def _apply_hunks(original_lines: List[str], hunk_lines: List[str]) -> List[str]:
    output: List[str] = []
    source_index = 0
    i = 0
    while i < len(hunk_lines):
        header = hunk_lines[i]
        match = HUNK_HEADER.match(header)
        if not match:
            raise ValueError(f"Invalid hunk header: {header}")
        start_old = int(match.group(1)) - 1
        output.extend(original_lines[source_index:start_old])
        source_index = start_old
        i += 1
        while i < len(hunk_lines) and not hunk_lines[i].startswith("@@"):
            line = hunk_lines[i]
            if not line:
                prefix, payload = " ", ""
            else:
                prefix, payload = line[0], line[1:]
            if prefix == " ":
                if source_index >= len(original_lines) or original_lines[source_index] != payload:
                    raise ValueError("Patch context does not match original file.")
                output.append(payload)
                source_index += 1
            elif prefix == "-":
                if source_index >= len(original_lines) or original_lines[source_index] != payload:
                    raise ValueError("Patch removal does not match original file.")
                source_index += 1
            elif prefix == "+":
                output.append(payload)
            elif prefix == "\\":
                pass
            else:
                raise ValueError(f"Unsupported patch line: {line}")
            i += 1
    output.extend(original_lines[source_index:])
    return output


def apply_unified_diff(files_before: Dict[str, str], patch_text: str) -> Dict[str, str]:
    if not patch_text.strip():
        raise ValueError("Empty patch.")
    patched = dict(files_before)
    sections = _extract_patch_sections(patch_text)
    if not sections:
        raise ValueError("No unified diff sections found.")
    for file_path, body_lines in sections.items():
        if file_path not in patched:
            raise ValueError(f"Patch targets unknown file: {file_path}")
        hunks: List[List[str]] = []
        current_hunk: List[str] = []
        for line in body_lines:
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = [line]
            elif current_hunk:
                current_hunk.append(line)
        if current_hunk:
            hunks.append(current_hunk)
        if not hunks:
            raise ValueError(f"No hunks found for file: {file_path}")
        updated_lines = patched[file_path].splitlines()
        for hunk in hunks:
            updated_lines = _apply_hunks(updated_lines, hunk)
        patched[file_path] = "\n".join(updated_lines) + "\n"
    return patched


def validate_fix_patch(task: TaskSpec, patch_text: str) -> tuple[bool, str]:
    try:
        patched_files = apply_unified_diff(task.files_before, patch_text)
    except Exception as exc:  # noqa: BLE001
        return False, f"Patch could not be applied: {exc}"

    workspace_path: Optional[Path] = None
    try:
        workspace_path = Path(tempfile.mkdtemp(prefix=f"{task.task_id}_"))
        for relative_path, content in patched_files.items():
            destination = workspace_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        init_file = workspace_path / "src" / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        if task.validator is None:
            return False, "Task validator is missing."
        return task.validator(workspace_path)
    finally:
        if workspace_path is not None and workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)


def evaluate_history(task: TaskSpec, history: List[HistoryEntry]) -> GraderResponse:
    bug_correct = False
    fix_correct = False
    decision_correct = False
    clarification_useful = False
    score = 0.0
    components: Dict[str, float] = {}
    efficiency_penalty = 0.0
    repeated_action_counts: Dict[str, int] = {}

    for index, entry in enumerate(history, start=1):
        repeated_action_counts[entry.action_type] = repeated_action_counts.get(entry.action_type, 0) + 1
        action_type = ActionType(entry.action_type)
        content = entry.content or ""

        if action_type == ActionType.IDENTIFY_BUG:
            if not bug_correct and bug_identification_matches(task, content):
                bug_correct = True
                score += 0.5
                components["correct_bug_identification"] = 0.5
            elif not bug_identification_matches(task, content):
                score -= 0.2
                components["false_positive_bug_identification"] = components.get(
                    "false_positive_bug_identification", 0.0
                ) - 0.2

        elif action_type == ActionType.SUGGEST_FIX:
            patch_valid, _ = validate_fix_patch(task, content)
            if not fix_correct and patch_valid:
                fix_correct = True
                score += 0.3
                components["validated_fix"] = 0.3
            elif not patch_valid and not fix_explanation_matches(task, content):
                score -= 0.2
                components["invalid_fix"] = components.get("invalid_fix", 0.0) - 0.2

        elif action_type == ActionType.ASK_FOR_CLARIFICATION:
            helpful = task.uncertainty_level == "high"
            if helpful:
                clarification_useful = True
                score += 0.05
                components["useful_clarification"] = components.get("useful_clarification", 0.0) + 0.05
            else:
                score -= 0.05
                components["unnecessary_clarification"] = components.get(
                    "unnecessary_clarification", 0.0
                ) - 0.05

        elif action_type == ActionType.ADD_COMMENT:
            if bug_identification_matches(task, content) or keyword_hit_count(content, task.expected_fix_keywords) >= 1:
                score += 0.05
                components["useful_comment"] = components.get("useful_comment", 0.0) + 0.05

        elif action_type == ActionType.APPROVE_PR:
            if task.bug_present and not fix_correct:
                score -= 0.5
                components["approved_buggy_pr"] = -0.5
            else:
                decision_correct = True

        elif action_type == ActionType.REQUEST_CHANGES:
            if task.bug_present:
                decision_correct = True

        score -= 0.05 * index
        efficiency_penalty -= 0.05 * index
        if repeated_action_counts[entry.action_type] > 1 and action_type not in {
            ActionType.ADD_COMMENT,
            ActionType.ASK_FOR_CLARIFICATION,
        }:
            score -= 0.05
            efficiency_penalty -= 0.05

    if bug_correct and fix_correct and decision_correct and task.bug_present:
        score += 1.0
        components["fully_correct_review"] = 1.0

    score = max(0.0, min(1.0, round(score, 4)))
    summary = "Agent partially completed the review."
    if score >= 0.95:
        summary = "Agent completed a correct end-to-end review."
    elif score == 0.0:
        summary = "Agent failed to perform a correct review."

    return GraderResponse(
        task_id=task.task_id,
        score=score,
        components=components,
        bug_identification_correct=bug_correct,
        fix_correct=fix_correct,
        decision_correct=decision_correct,
        efficiency_penalty=efficiency_penalty,
        clarification_useful=clarification_useful,
        summary=summary,
    )
