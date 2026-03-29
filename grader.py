from __future__ import annotations

import re
import types
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
    normalized = normalize_text(content)
    if task.diagnosis_concepts:
        matched_concepts = 0
        for concept_group in task.diagnosis_concepts:
            if any(normalize_text(keyword) in normalized for keyword in concept_group):
                matched_concepts += 1
        required_concepts = 2 if task.difficulty == "easy" else min(3, len(task.diagnosis_concepts))
        return matched_concepts >= required_concepts

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


def _load_module_from_source(module_name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(module_name)
    exec(compile(source, module_name, "exec"), module.__dict__)
    return module


def _validate_patched_files(task: TaskSpec, patched_files: Dict[str, str]) -> tuple[bool, str]:
    if task.task_id == "easy_keyword_preview":
        module = _load_module_from_source("task_easy_keyword_preview", patched_files["src/keyword_preview.py"])
        build_keyword_preview = getattr(module, "build_keyword_preview")
        if build_keyword_preview([" Billing ", "Payments", " Outage "]) != "billing, payments, outage":
            return False, "Preview should include all cleaned tokens in order."
        if build_keyword_preview(["   ", "Status"]) != "status":
            return False, "Preview should skip empty entries but preserve remaining labels."
        if build_keyword_preview([]) != "":
            return False, "Empty input should return an empty preview."
        return True, "All keyword preview tests passed."

    if task.task_id == "medium_job_retry":
        module = _load_module_from_source("task_medium_job_retry", patched_files["src/job_retry.py"])
        collect_retryable_jobs = getattr(module, "collect_retryable_jobs")
        jobs = [
            {"job_id": "alpha", "status": "failed", "attempts": 1, "max_attempts": 3},
            {"job_id": "bravo", "status": "failed", "attempts": 3, "max_attempts": 3},
            {"job_id": "charlie", "status": "running", "attempts": 1, "max_attempts": 2},
            {"job_id": "delta", "status": "failed", "attempts": 0, "max_attempts": 1},
        ]
        if collect_retryable_jobs(jobs) != ["alpha", "delta"]:
            return False, "Only failed jobs with attempts still below max_attempts should be retried."
        if collect_retryable_jobs([{"job_id": "echo", "status": "failed", "attempts": 4, "max_attempts": 4}]) != []:
            return False, "Jobs at the retry limit must not be retried again."
        return True, "All retry selection tests passed."

    if task.task_id == "hard_feature_flags":
        module = _load_module_from_source("task_hard_feature_flags", patched_files["src/feature_flags.py"])
        merge_feature_flags = getattr(module, "merge_feature_flags")
        current = {"beta_dashboard": True, "dark_mode": True, "priority_support": False}
        incoming = {"beta_dashboard": False, "dark_mode": None, "priority_support": True}
        expected = {"beta_dashboard": False, "dark_mode": True, "priority_support": True}
        if merge_feature_flags(current, incoming) != expected:
            return False, "Explicit false must disable a flag, while None means keep the existing value."
        if merge_feature_flags({"reports": False}, {"reports": None}) != {"reports": False}:
            return False, "None updates should not modify existing values."
        return True, "All feature flag merge tests passed."

    if task.task_id == "frontier_discount_rollup":
        pricing = _load_module_from_source("task_frontier_pricing", patched_files["src/pricing_rollup.py"])
        summary = _load_module_from_source("task_frontier_summary", patched_files["src/summary_builder.py"])
        summarize_checkout = getattr(pricing, "summarize_checkout")
        build_summary_line = getattr(summary, "build_summary_line")
        carts = [
            {
                "items": [{"sku": "starter", "price_cents": 5000, "quantity": 1}],
                "coupon": {"type": "flat", "amount_cents": 7000},
                "shipping_cents": 1200,
            },
            {
                "items": [{"sku": "pro", "price_cents": 2500, "quantity": 2}],
                "coupon": {"type": "flat", "amount_cents": 1000},
                "shipping_cents": 500,
            },
        ]
        result = [summarize_checkout(cart) for cart in carts]
        expected = [
            {"subtotal_cents": 5000, "discount_cents": 5000, "total_cents": 1200},
            {"subtotal_cents": 5000, "discount_cents": 1000, "total_cents": 4500},
        ]
        if result != expected:
            return False, "Discounts must be capped at the subtotal so totals never go negative before shipping."
        if build_summary_line(result[0]) != "subtotal=5000 discount=5000 total=1200":
            return False, "Summary output must reflect the capped discount and resulting total."
        return True, "All capped-discount pricing tests passed."

    if task.task_id == "hard_billing_suspension":
        module = _load_module_from_source("task_hard_billing_suspension", patched_files["src/billing_suspension.py"])
        should_suspend_account = getattr(module, "should_suspend_account")
        boundary_account = {
            "invoice_status": "unpaid",
            "overdue_days": 14,
            "grace_days": 14,
            "active_payment_plan": False,
        }
        if should_suspend_account(boundary_account) is not False:
            return False, "Accounts exactly at the grace boundary should not be suspended yet."
        plan_account = {
            "invoice_status": "unpaid",
            "overdue_days": 21,
            "grace_days": 14,
            "active_payment_plan": True,
        }
        if should_suspend_account(plan_account) is not False:
            return False, "Accounts with an active payment plan must not be suspended."
        overdue_account = {
            "invoice_status": "unpaid",
            "overdue_days": 21,
            "grace_days": 14,
            "active_payment_plan": False,
        }
        if should_suspend_account(overdue_account) is not True:
            return False, "Accounts past grace with no payment plan should be suspended."
        paid_account = {
            "invoice_status": "paid",
            "overdue_days": 40,
            "grace_days": 14,
            "active_payment_plan": False,
        }
        if should_suspend_account(paid_account) is not False:
            return False, "Paid accounts must never be suspended."
        return True, "All billing suspension tests passed."

    if task.task_id == "medium_receipt_format_cleanup":
        module = _load_module_from_source("task_medium_receipt_format", patched_files["src/receipt_format.py"])
        build_receipt_line = getattr(module, "build_receipt_line")
        if build_receipt_line("starter", 2500, 2) != "starter x2 - $50.00":
            return False, "Receipt formatting should stay unchanged for standard line items."
        if build_receipt_line("pro", 1250, 1) != "pro x1 - $12.50":
            return False, "Receipt formatting should preserve quantity and currency formatting."
        return True, "All receipt formatting tests passed."

    if task.task_id == "frontier_incident_rollout":
        module = _load_module_from_source("task_frontier_incident_rollout", patched_files["src/incident_rollout.py"])
        should_page_incident = getattr(module, "should_page_incident")
        boundary_incident = {
            "severity": 4,
            "page_threshold": 4,
            "minutes_open": 5,
            "customer_tier": "standard",
            "customer_impacting": False,
        }
        if should_page_incident(boundary_incident) is not False:
            return False, "Severity equal to the threshold should not page without customer impact."
        impacting_incident = {
            "severity": 2,
            "page_threshold": 4,
            "minutes_open": 10,
            "customer_tier": "standard",
            "customer_impacting": True,
        }
        if should_page_incident(impacting_incident) is not True:
            return False, "Customer-impacting incidents must still page even below the severity threshold."
        severe_incident = {
            "severity": 5,
            "page_threshold": 4,
            "minutes_open": 3,
            "customer_tier": "standard",
            "customer_impacting": False,
        }
        if should_page_incident(severe_incident) is not True:
            return False, "Incidents above the page threshold should page."
        return True, "All incident rollout tests passed."

    return False, "Task validator is missing."


def validate_fix_patch(task: TaskSpec, patch_text: str) -> tuple[bool, str]:
    try:
        patched_files = apply_unified_diff(task.files_before, patch_text)
    except Exception as exc:  # noqa: BLE001
        return False, f"Patch could not be applied: {exc}"
    try:
        return _validate_patched_files(task, patched_files)
    except Exception as exc:  # noqa: BLE001
        return False, f"Patched code could not be evaluated: {exc}"


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
            has_diff = "diff --git" in content
            if not fix_correct and patch_valid:
                fix_correct = True
                score += 0.3
                components["validated_fix"] = 0.3
            elif not patch_valid and not has_diff and fix_explanation_matches(task, content):
                score += 0.1
                components["partial_fix_reasoning"] = components.get("partial_fix_reasoning", 0.0) + 0.1
            elif not patch_valid and not fix_explanation_matches(task, content):
                score -= 0.1
                components["invalid_fix"] = components.get("invalid_fix", 0.0) - 0.1

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
                if not task.bug_present:
                    score += 0.8
                    components["correct_approval"] = 0.8

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
    elif decision_correct and not task.bug_present:
        score += 0.6
        components["fully_correct_review"] = 0.6

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
