from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
MAX_STEPS = 6
TEMPERATURE = 0.0


SYSTEM_PROMPT = """You are reviewing a pull request in a code review environment.
Reply with a single JSON object: {"action_type": "...", "content": "..."}.
Valid action_type values are identify_bug, suggest_fix, add_comment, ask_for_clarification, request_changes, approve_pr.
Use suggest_fix only when you can provide a unified diff patch beginning with diff --git.
Favor request_changes for buggy code. Do not include markdown fences.
"""


def choose_heuristic_action(observation: Dict[str, Any], step_index: int) -> Dict[str, str]:
    task_id = observation["task_id"]
    if step_index == 0:
        if task_id == "easy_keyword_preview":
            return {
                "action_type": "identify_bug",
                "content": "The preview builder only returns the first cleaned keyword, so the ticket card drops the remaining normalized terms instead of joining all of them.",
            }
        if task_id == "medium_job_retry":
            return {
                "action_type": "identify_bug",
                "content": "The retry guard is inverted. Failed jobs whose attempts are already at max_attempts are being re-queued, while jobs with retry budget left are skipped.",
            }
        if task_id == "frontier_discount_rollup":
            return {
                "action_type": "ask_for_clarification",
                "content": "Should a flat coupon be capped at the subtotal before shipping is applied, or can it push the merchandise total negative?",
            }
        return {
            "action_type": "ask_for_clarification",
            "content": "Can you confirm whether null means no change while explicit false should disable the feature flag?",
        }
    if step_index == 1:
        if task_id == "easy_keyword_preview":
            patch = """diff --git a/src/keyword_preview.py b/src/keyword_preview.py
--- a/src/keyword_preview.py
+++ b/src/keyword_preview.py
@@ -2,6 +2,4 @@
     cleaned = [token.strip().lower() for token in tokens if token and token.strip()]
     if not cleaned:
         return ""
-    if cleaned:
-        return cleaned[0]
-    return ""
+    return ", ".join(cleaned)
"""
            return {"action_type": "suggest_fix", "content": patch}
        if task_id == "medium_job_retry":
            patch = """diff --git a/src/job_retry.py b/src/job_retry.py
--- a/src/job_retry.py
+++ b/src/job_retry.py
@@ -3,6 +3,6 @@
     for job in jobs:
         if job["status"] != "failed":
             continue
-        if job["attempts"] >= job["max_attempts"]:
-            retryable.append(job["job_id"])
-            continue
+        if job["attempts"] >= job["max_attempts"]:
+            continue
+        retryable.append(job["job_id"])
     return retryable
"""
            return {"action_type": "suggest_fix", "content": patch}
        if task_id == "frontier_discount_rollup":
            return {
                "action_type": "identify_bug",
                "content": "The flat coupon amount is no longer capped at the subtotal, so oversized discounts create negative merchandise totals and invalid receipt summaries on the edge case with shipping.",
            }
        return {
            "action_type": "identify_bug",
            "content": "The merge logic uses a truthy check, which ignores explicit false updates. That breaks the contract where None means no change but false means disable the flag.",
        }
    if step_index == 2 and task_id == "frontier_discount_rollup":
        patch = """diff --git a/src/pricing_rollup.py b/src/pricing_rollup.py
--- a/src/pricing_rollup.py
+++ b/src/pricing_rollup.py
@@ -2,7 +2,7 @@
     subtotal_cents = sum(item["price_cents"] * item["quantity"] for item in cart["items"])
     coupon = cart.get("coupon")
     discount_cents = 0
     if coupon and coupon["type"] == "flat":
-        discount_cents = coupon["amount_cents"]
+        discount_cents = min(coupon["amount_cents"], subtotal_cents)
     total_cents = subtotal_cents - discount_cents + cart.get("shipping_cents", 0)
     return {
"""
        return {"action_type": "suggest_fix", "content": patch}
    if step_index == 2 and task_id == "hard_feature_flags":
        patch = """diff --git a/src/feature_flags.py b/src/feature_flags.py
--- a/src/feature_flags.py
+++ b/src/feature_flags.py
@@ -1,5 +1,6 @@
 def merge_feature_flags(current: dict[str, bool], incoming: dict[str, Optional[bool]]) -> dict[str, bool]:
     merged = current.copy()
     for flag_name, value in incoming.items():
-        if value:
-            merged[flag_name] = value
+        if value is None:
+            continue
+        merged[flag_name] = value
     return merged
"""
        return {"action_type": "suggest_fix", "content": patch}
    return {
        "action_type": "request_changes",
        "content": "The pull request introduces a bug and should not be approved until the fix is applied.",
    }


def call_model(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, str]:
    user_prompt = json.dumps(
        {
            "task_id": observation["task_id"],
            "difficulty": observation["difficulty"],
            "pr_title": observation["pr_title"],
            "commit_message": observation["commit_message"],
            "changed_files": observation["changed_files"],
            "code_diff": observation["code_diff"],
            "test_results": observation["test_results"],
            "issue_description": observation.get("issue_description"),
            "discussion_context": observation.get("discussion_context", []),
            "remaining_steps": observation["remaining_steps"],
        },
        indent=2,
    )
    response = client.responses.create(
        model=DEFAULT_MODEL_NAME,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.output_text.strip()
    payload = json.loads(text)
    return {"action_type": payload["action_type"], "content": payload.get("content", "")}


def run_episode(http_client: httpx.Client, task_id: str, use_model: bool, llm_client: Optional[OpenAI]) -> Dict[str, Any]:
    reset_response = http_client.post("/reset", json={"task_id": task_id})
    reset_response.raise_for_status()
    observation = reset_response.json()["observation"]
    trajectory: List[Dict[str, Any]] = []

    for step_index in range(MAX_STEPS):
        if use_model:
            if llm_client is None:
                raise RuntimeError("OpenAI client is required when use_model=True.")
            action = call_model(llm_client, observation)
        else:
            action = choose_heuristic_action(observation, step_index)
        step_response = http_client.post("/step", json=action)
        step_response.raise_for_status()
        payload = step_response.json()
        trajectory.append({"action": action, "reward": payload["reward"]["value"]})
        observation = payload["observation"]
        if payload["done"]:
            break

    grader_response = http_client.post("/grader", json={})
    grader_response.raise_for_status()
    grade = grader_response.json()
    return {
        "task_id": task_id,
        "score": grade["score"],
        "components": grade["components"],
        "trajectory": trajectory,
    }


def run_baseline(base_url: str = DEFAULT_BASE_URL, use_model: bool = True) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    llm_client: Optional[OpenAI] = None
    if use_model:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set for model-driven baseline runs.")
        llm_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("API_BASE_URL"),
        )

    with httpx.Client(base_url=base_url, timeout=30.0) as http_client:
        tasks_response = http_client.get("/tasks")
        tasks_response.raise_for_status()
        tasks = tasks_response.json()
        results = [run_episode(http_client, task["task_id"], use_model, llm_client) for task in tasks]

    average_score = round(sum(result["score"] for result in results) / len(results), 4)
    return {"results": results, "average_score": average_score}


def main() -> None:
    base_url = os.getenv("ENV_BASE_URL", DEFAULT_BASE_URL)
    result = run_baseline(base_url=base_url, use_model=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
