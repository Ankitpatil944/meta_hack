from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import CodeReviewEnv
from grader import validate_fix_patch
from models import Action, ActionType
from tasks import get_task


def test_easy_task_success_path() -> None:
    env = CodeReviewEnv()
    env.reset("easy_keyword_preview")
    env.step(
        Action(
            action_type=ActionType.IDENTIFY_BUG,
            content="The preview only returns the first keyword instead of joining all cleaned values.",
        )
    )
    env.step(
        Action(
            action_type=ActionType.SUGGEST_FIX,
            content="""diff --git a/src/keyword_preview.py b/src/keyword_preview.py
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
""",
        )
    )
    _, reward, done, _ = env.step(
        Action(
            action_type=ActionType.REQUEST_CHANGES,
            content="This introduces a regression in keyword previews.",
        )
    )
    assert done is True
    assert env.state().bug_identified_correctly is True
    assert env.state().fix_suggested_correctly is True
    assert reward.components["fully_correct_review"] > 0
    assert reward.cumulative_reward >= 1.4
    assert env.state().reasoning_trace


def test_efficiency_penalty_grows_with_steps() -> None:
    env = CodeReviewEnv()
    env.reset("easy_keyword_preview")
    _, first_reward, _, _ = env.step(
        Action(action_type=ActionType.ADD_COMMENT, content="Checking the preview behavior.")
    )
    _, second_reward, _, _ = env.step(
        Action(action_type=ActionType.ADD_COMMENT, content="Still tracing the preview builder.")
    )
    assert first_reward.components["efficiency_penalty"] == -0.05
    assert second_reward.components["efficiency_penalty"] == -0.1


def test_public_state_hides_internal_bug_truth() -> None:
    env = CodeReviewEnv()
    env.reset("medium_job_retry")
    public_state = env.public_state().model_dump()
    assert "bug_present" not in public_state
    assert "bug_type" not in public_state
    assert "bug_identified_correctly" not in public_state
    assert "fix_suggested_correctly" not in public_state


def test_reset_cycles_task_variants() -> None:
    env = CodeReviewEnv()
    first = env.reset("medium_job_retry")
    second = env.reset("medium_job_retry")
    assert first.task_id == second.task_id == "medium_job_retry"
    assert first.variant_id != second.variant_id
    assert first.code_diff != second.code_diff


def test_incorrect_fix_penalized() -> None:
    env = CodeReviewEnv()
    env.reset("easy_keyword_preview")
    env.step(Action(action_type=ActionType.IDENTIFY_BUG, content="The PR changes keyword preview behavior incorrectly."))
    _, reward, done, _ = env.step(
        Action(
            action_type=ActionType.SUGGEST_FIX,
            content="""diff --git a/src/keyword_preview.py b/src/keyword_preview.py
--- a/src/keyword_preview.py
+++ b/src/keyword_preview.py
@@ -2,6 +2,6 @@
     cleaned = [token.strip().lower() for token in tokens if token and token.strip()]
     if not cleaned:
         return ""
-    if cleaned:
-        return cleaned[0]
+    if cleaned:
+        return cleaned[-1]
     return ""
""",
        )
    )
    assert done is False
    assert reward.value < 0
    assert reward.components["invalid_fix"] < 0


def test_clarification_flow() -> None:
    env = CodeReviewEnv()
    env.reset("hard_feature_flags")
    obs, reward, done, _ = env.step(
        Action(
            action_type=ActionType.ASK_FOR_CLARIFICATION,
            content="Is False a valid override?",
        )
    )
    assert done is False
    assert reward.components["useful_clarification"] > 0
    assert "explicit false must disable the flag" in " ".join(obs.discussion_context).lower()


def test_wrong_approval_penalty() -> None:
    env = CodeReviewEnv()
    env.reset("easy_keyword_preview")
    _, reward, done, info = env.step(
        Action(
            action_type=ActionType.APPROVE_PR,
            content="Looks good to merge.",
        )
    )
    assert done is True
    assert reward.value < 0
    assert reward.components["approved_buggy_pr"] < 0
    assert info["final_decision"] == ActionType.APPROVE_PR.value


def test_hard_task_edge_case_behavior() -> None:
    env = CodeReviewEnv()
    obs = env.reset("hard_feature_flags")
    assert obs.uncertainty_level == "high"
    assert obs.review_hint is not None
    assert "false is a valid override" in obs.review_hint.lower()
    combined_context = f"{obs.test_results}\n{obs.issue_description or ''}"
    assert "false" in combined_context.lower()
    assert "explicit" in combined_context.lower() or "omitted" in combined_context.lower()


def test_equivalent_hard_fix_is_behaviorally_accepted() -> None:
    task = get_task("hard_feature_flags")
    patch = """diff --git a/src/feature_flags.py b/src/feature_flags.py
--- a/src/feature_flags.py
+++ b/src/feature_flags.py
@@ -4,5 +4,5 @@
 def merge_feature_flags(current: dict[str, bool], incoming: dict[str, Optional[bool]]) -> dict[str, bool]:
     merged = current.copy()
     for flag_name, value in incoming.items():
-        if value:
+        if value is not None:
             merged[flag_name] = value
     return merged
"""
    valid, message = validate_fix_patch(task, patch)
    assert valid is True, message


def test_equivalent_medium_fix_is_behaviorally_accepted() -> None:
    task = get_task("medium_job_retry")
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
+        if job["attempts"] < job["max_attempts"]:
+            retryable.append(job["job_id"])
+            continue
     return retryable
"""
    valid, message = validate_fix_patch(task, patch)
    assert valid is True, message


def test_hard_billing_suspension_requires_contract_complete_fix() -> None:
    task = get_task("hard_billing_suspension")
    boundary_only_patch = """diff --git a/src/billing_suspension.py b/src/billing_suspension.py
--- a/src/billing_suspension.py
+++ b/src/billing_suspension.py
@@ -1,4 +1,4 @@
 def should_suspend_account(account: dict) -> bool:
     if account["invoice_status"] != "unpaid":
         return False
-    return account["overdue_days"] >= account["grace_days"]
+    return account["overdue_days"] > account["grace_days"]
"""
    valid, _ = validate_fix_patch(task, boundary_only_patch)
    assert valid is False


def test_partial_fix_reasoning_gets_small_positive_signal() -> None:
    env = CodeReviewEnv()
    env.reset("medium_job_retry")
    _, reward, done, _ = env.step(
        Action(
            action_type=ActionType.SUGGEST_FIX,
            content="The fix is to retry only failed jobs with retry budget left and skip exhausted jobs.",
        )
    )
    assert done is False
    assert reward.components["partial_fix_reasoning"] > 0
    assert reward.value > 0
