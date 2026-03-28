from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Tuple


Validator = Callable[[Path], Tuple[bool, str]]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    pr_title: str
    summary: str
    commit_message: str
    changed_files: List[str]
    code_diff: str
    test_results: str
    issue_description: Optional[str]
    bug_present: bool
    bug_type: str
    uncertainty_level: str
    files_before: Dict[str, str]
    expected_bug_keywords: List[str]
    expected_fix_keywords: List[str]
    clarification_hints: Dict[str, str] = field(default_factory=dict)
    validator: Optional[Validator] = None


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_easy(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_easy_keyword_preview", workspace / "src" / "keyword_preview.py")
    build_keyword_preview = getattr(module, "build_keyword_preview")
    if build_keyword_preview([" Billing ", "Payments", " Outage "]) != "billing, payments, outage":
        return False, "Preview should include all cleaned tokens in order."
    if build_keyword_preview(["   ", "Status"]) != "status":
        return False, "Preview should skip empty entries but preserve remaining labels."
    if build_keyword_preview([]) != "":
        return False, "Empty input should return an empty preview."
    return True, "All keyword preview tests passed."


def _validate_medium(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_medium_job_retry", workspace / "src" / "job_retry.py")
    collect_retryable_jobs = getattr(module, "collect_retryable_jobs")
    jobs = [
        {"job_id": "alpha", "status": "failed", "attempts": 1, "max_attempts": 3},
        {"job_id": "bravo", "status": "failed", "attempts": 3, "max_attempts": 3},
        {"job_id": "charlie", "status": "running", "attempts": 1, "max_attempts": 2},
        {"job_id": "delta", "status": "failed", "attempts": 0, "max_attempts": 1},
    ]
    result = collect_retryable_jobs(jobs)
    if result != ["alpha", "delta"]:
        return False, "Only failed jobs with attempts still below max_attempts should be retried."
    second_run = collect_retryable_jobs(
        [{"job_id": "echo", "status": "failed", "attempts": 4, "max_attempts": 4}]
    )
    if second_run != []:
        return False, "Jobs at the retry limit must not be retried again."
    return True, "All retry selection tests passed."


def _validate_hard(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_hard_feature_flags", workspace / "src" / "feature_flags.py")
    merge_feature_flags = getattr(module, "merge_feature_flags")
    current = {"beta_dashboard": True, "dark_mode": True, "priority_support": False}
    incoming = {"beta_dashboard": False, "dark_mode": None, "priority_support": True}
    result = merge_feature_flags(current, incoming)
    expected = {"beta_dashboard": False, "dark_mode": True, "priority_support": True}
    if result != expected:
        return False, "Explicit false must disable a flag, while None means keep the existing value."
    no_change = merge_feature_flags({"reports": False}, {"reports": None})
    if no_change != {"reports": False}:
        return False, "None updates should not modify existing values."
    return True, "All feature flag merge tests passed."


def _validate_frontier(workspace: Path) -> Tuple[bool, str]:
    pricing = _load_module("task_frontier_pricing", workspace / "src" / "pricing_rollup.py")
    summarizer = _load_module("task_frontier_summary", workspace / "src" / "summary_builder.py")
    summarize_checkout = getattr(pricing, "summarize_checkout")
    build_summary_line = getattr(summarizer, "build_summary_line")

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

    summary = build_summary_line(result[0])
    if summary != "subtotal=5000 discount=5000 total=1200":
        return False, "Summary output must reflect the capped discount and resulting total."

    return True, "All capped-discount pricing tests passed."


TASKS: Dict[str, TaskSpec] = {
    "easy_keyword_preview": TaskSpec(
        task_id="easy_keyword_preview",
        difficulty="easy",
        pr_title="Speed up keyword previews on ticket cards",
        summary="Reviewer must catch that the preview only returns the first cleaned keyword instead of the full summary.",
        commit_message="perf: short-circuit keyword preview generation for non-empty token lists",
        changed_files=["src/keyword_preview.py", "tests/test_keyword_preview.py"],
        code_diff="""diff --git a/src/keyword_preview.py b/src/keyword_preview.py
index 9f03a61..cbf7d2d 100644
--- a/src/keyword_preview.py
+++ b/src/keyword_preview.py
@@ -1,9 +1,12 @@
 def build_keyword_preview(tokens: list[str]) -> str:
     cleaned = [token.strip().lower() for token in tokens if token and token.strip()]
     if not cleaned:
         return ""
-    return ", ".join(cleaned)
+    if cleaned:
+        return cleaned[0]
+    return ""

diff --git a/tests/test_keyword_preview.py b/tests/test_keyword_preview.py
index a1f31bb..8f0d9f1 100644
--- a/tests/test_keyword_preview.py
+++ b/tests/test_keyword_preview.py
@@ -5,6 +5,6 @@ def test_preview_normalizes_all_keywords():
-    assert build_keyword_preview([" Billing ", "Payments", " Outage "]) == "billing, payments, outage"
+    assert build_keyword_preview([" Billing ", "Payments", " Outage "]) == "billing, payments, outage"
""",
        test_results="""FAILED tests/test_keyword_preview.py::test_preview_normalizes_all_keywords
E   AssertionError: assert 'billing' == 'billing, payments, outage'
E     - billing, payments, outage
E     + billing
""",
        issue_description="The keyword preview on support ticket cards should show every normalized token, not just the first one.",
        bug_present=True,
        bug_type="partial_list_processing",
        uncertainty_level="low",
        files_before={
            "src/keyword_preview.py": """def build_keyword_preview(tokens: list[str]) -> str:
    cleaned = [token.strip().lower() for token in tokens if token and token.strip()]
    if not cleaned:
        return ""
    if cleaned:
        return cleaned[0]
    return ""
""",
            "tests/test_keyword_preview.py": """from src.keyword_preview import build_keyword_preview


def test_preview_normalizes_all_keywords() -> None:
    assert build_keyword_preview([" Billing ", "Payments", " Outage "]) == "billing, payments, outage"


def test_preview_skips_empty_values() -> None:
    assert build_keyword_preview(["   ", "Status"]) == "status"
""",
        },
        expected_bug_keywords=["first", "only", "keyword", "preview", "all", "join"],
        expected_fix_keywords=["join", "cleaned", "comma", "preview"],
        clarification_hints={
            "preview": "The card should display the entire normalized preview string, not just the first token.",
            "default": "This ticket card preview is meant to summarize all searchable terms.",
        },
        validator=_validate_easy,
    ),
    "medium_job_retry": TaskSpec(
        task_id="medium_job_retry",
        difficulty="medium",
        pr_title="Refactor retry queue selection for failed background jobs",
        summary="Reviewer must catch that the retry filter was inverted and now selects exhausted jobs instead of retryable ones.",
        commit_message="cleanup: simplify failed-job selection before queueing retries",
        changed_files=["src/job_retry.py", "tests/test_job_retry.py"],
        code_diff="""diff --git a/src/job_retry.py b/src/job_retry.py
index 411d26f..8bd4b5f 100644
--- a/src/job_retry.py
+++ b/src/job_retry.py
@@ -1,11 +1,11 @@
 def collect_retryable_jobs(jobs: list[dict]) -> list[str]:
     retryable: list[str] = []
     for job in jobs:
         if job["status"] != "failed":
             continue
-        if job["attempts"] >= job["max_attempts"]:
-            continue
-        retryable.append(job["job_id"])
+        if job["attempts"] >= job["max_attempts"]:
+            retryable.append(job["job_id"])
+            continue
     return retryable

diff --git a/tests/test_job_retry.py b/tests/test_job_retry.py
index 24bfb18..a3f7be5 100644
--- a/tests/test_job_retry.py
+++ b/tests/test_job_retry.py
@@ -9,4 +9,4 @@ def test_retry_selector_only_returns_failed_jobs_with_budget():
-    assert collect_retryable_jobs(jobs) == ["alpha", "delta"]
+    assert collect_retryable_jobs(jobs) == ["alpha", "delta"]
""",
        test_results="""FAILED tests/test_job_retry.py::test_retry_selector_only_returns_failed_jobs_with_budget
E   AssertionError: assert ['bravo'] == ['alpha', 'delta']
E     At index 0 diff: 'bravo' != 'alpha'
E     Right contains one more item: 'delta'
""",
        issue_description="Only failed jobs that still have retry budget should be re-queued.",
        bug_present=True,
        bug_type="inverted_retry_guard",
        uncertainty_level="medium",
        files_before={
            "src/job_retry.py": """def collect_retryable_jobs(jobs: list[dict]) -> list[str]:
    retryable: list[str] = []
    for job in jobs:
        if job["status"] != "failed":
            continue
        if job["attempts"] >= job["max_attempts"]:
            retryable.append(job["job_id"])
            continue
    return retryable
""",
            "tests/test_job_retry.py": """from src.job_retry import collect_retryable_jobs


def test_retry_selector_only_returns_failed_jobs_with_budget() -> None:
    jobs = [
        {"job_id": "alpha", "status": "failed", "attempts": 1, "max_attempts": 3},
        {"job_id": "bravo", "status": "failed", "attempts": 3, "max_attempts": 3},
        {"job_id": "charlie", "status": "running", "attempts": 1, "max_attempts": 2},
        {"job_id": "delta", "status": "failed", "attempts": 0, "max_attempts": 1},
    ]
    assert collect_retryable_jobs(jobs) == ["alpha", "delta"]
""",
        },
        expected_bug_keywords=["retry", "attempt", "max_attempts", "inverted", "exhausted", "failed"],
        expected_fix_keywords=["continue", "less than", "retry budget", "skip exhausted"],
        clarification_hints={
            "budget": "A job should be retried only while attempts remain strictly below max_attempts.",
            "default": "The queue should exclude jobs that have already consumed their retry budget.",
        },
        validator=_validate_medium,
    ),
    "hard_feature_flags": TaskSpec(
        task_id="hard_feature_flags",
        difficulty="hard",
        pr_title="Unify feature-flag merge path for partial account updates",
        summary="Reviewer must reason about explicit false vs None semantics and may ask for clarification before deciding.",
        commit_message="refactor: reuse truthy merge logic for partial feature flag updates",
        changed_files=["src/feature_flags.py", "tests/test_feature_flags.py", "docs/account_patch_contract.md"],
        code_diff="""diff --git a/src/feature_flags.py b/src/feature_flags.py
index c2930f0..14dc714 100644
--- a/src/feature_flags.py
+++ b/src/feature_flags.py
@@ -1,8 +1,8 @@
def merge_feature_flags(current: dict[str, bool], incoming: dict[str, Optional[bool]]) -> dict[str, bool]:
     merged = current.copy()
     for flag_name, value in incoming.items():
-        if value is None:
-            continue
-        merged[flag_name] = value
+        if value:
+            merged[flag_name] = value
     return merged

diff --git a/docs/account_patch_contract.md b/docs/account_patch_contract.md
index 1a1f91f..c7b5224 100644
--- a/docs/account_patch_contract.md
+++ b/docs/account_patch_contract.md
@@ -1,5 +1,5 @@
 PATCH semantics:
-null = field omitted / keep current value
+null = field omitted / keep current value
 false = explicitly disable the flag
 true = explicitly enable the flag
""",
        test_results="""FAILED tests/test_feature_flags.py::test_merge_applies_explicit_false_updates
E   AssertionError: assert {'beta_dashboard': True, 'dark_mode': True, 'priority_support': True} == {'beta_dashboard': False, 'dark_mode': True, 'priority_support': True}
E     Differing items:
E     {'beta_dashboard': True} != {'beta_dashboard': False}
""",
        issue_description="The account patch contract distinguishes omitted values from explicit disables.",
        bug_present=True,
        bug_type="false_vs_none_merge_semantics",
        uncertainty_level="high",
        files_before={
            "src/feature_flags.py": """from typing import Optional


def merge_feature_flags(current: dict[str, bool], incoming: dict[str, Optional[bool]]) -> dict[str, bool]:
    merged = current.copy()
    for flag_name, value in incoming.items():
        if value:
            merged[flag_name] = value
    return merged
""",
            "tests/test_feature_flags.py": """from src.feature_flags import merge_feature_flags


def test_merge_applies_explicit_false_updates() -> None:
    current = {"beta_dashboard": True, "dark_mode": True, "priority_support": False}
    incoming = {"beta_dashboard": False, "dark_mode": None, "priority_support": True}
    assert merge_feature_flags(current, incoming) == {
        "beta_dashboard": False,
        "dark_mode": True,
        "priority_support": True,
    }
""",
            "docs/account_patch_contract.md": """PATCH semantics:
null = field omitted / keep current value
false = explicitly disable the flag
true = explicitly enable the flag
""",
        },
        expected_bug_keywords=["false", "none", "truthy", "flag", "explicit", "disable"],
        expected_fix_keywords=["is none", "continue", "explicit false", "merged"],
        clarification_hints={
            "false": "For this endpoint, null means no change. Explicit false must disable the flag.",
            "none": "For this endpoint, null means no change. Explicit false must disable the flag.",
            "default": "Account PATCH uses tri-state semantics: true enables, false disables, null leaves the existing value unchanged.",
        },
        validator=_validate_hard,
    ),
    "frontier_discount_rollup": TaskSpec(
        task_id="frontier_discount_rollup",
        difficulty="hard",
        pr_title="Reuse flat-discount helper across checkout totals and receipt summary",
        summary="Reviewer must catch a subtle multi-file pricing bug where oversized flat coupons create impossible negative subtotals that only surface in an edge-case receipt path.",
        commit_message="refactor: consolidate flat discount handling for checkout summary generation",
        changed_files=[
            "src/pricing_rollup.py",
            "src/summary_builder.py",
            "tests/test_pricing_rollup.py",
            "docs/checkout_contract.md",
        ],
        code_diff="""diff --git a/src/pricing_rollup.py b/src/pricing_rollup.py
index 4a8f123..bc99211 100644
--- a/src/pricing_rollup.py
+++ b/src/pricing_rollup.py
@@ -1,13 +1,14 @@
 def summarize_checkout(cart: dict) -> dict[str, int]:
     subtotal_cents = sum(item["price_cents"] * item["quantity"] for item in cart["items"])
     coupon = cart.get("coupon")
     discount_cents = 0
     if coupon and coupon["type"] == "flat":
-        discount_cents = min(coupon["amount_cents"], subtotal_cents)
+        discount_cents = coupon["amount_cents"]
     total_cents = subtotal_cents - discount_cents + cart.get("shipping_cents", 0)
     return {
         "subtotal_cents": subtotal_cents,
         "discount_cents": discount_cents,
         "total_cents": total_cents,
     }

diff --git a/src/summary_builder.py b/src/summary_builder.py
index 07c12ef..8ca1c90 100644
--- a/src/summary_builder.py
+++ b/src/summary_builder.py
@@ -1,4 +1,4 @@
 def build_summary_line(summary: dict[str, int]) -> str:
-    return f"subtotal={summary['subtotal_cents']} discount={summary['discount_cents']} total={summary['total_cents']}"
+    return f"subtotal={summary['subtotal_cents']} discount={summary['discount_cents']} total={summary['total_cents']}"

diff --git a/docs/checkout_contract.md b/docs/checkout_contract.md
index 0ccba27..9938d21 100644
--- a/docs/checkout_contract.md
+++ b/docs/checkout_contract.md
@@ -1,4 +1,4 @@
 Flat coupon rules:
 - flat coupons reduce subtotal but cannot exceed it
 - shipping is added after discounts
 - receipts should never show negative merchandise totals
""",
        test_results="""FAILED tests/test_pricing_rollup.py::test_flat_coupon_is_capped_before_shipping
E   AssertionError: assert {'subtotal_cents': 5000, 'discount_cents': 7000, 'total_cents': -800} == {'subtotal_cents': 5000, 'discount_cents': 5000, 'total_cents': 1200}
E     Differing items:
E     {'discount_cents': 7000} != {'discount_cents': 5000}
E     {'total_cents': -800} != {'total_cents': 1200}
""",
        issue_description="Receipts should never show negative merchandise totals even when a flat coupon is larger than the cart subtotal.",
        bug_present=True,
        bug_type="uncapped_flat_discount_rollup",
        uncertainty_level="high",
        files_before={
            "src/pricing_rollup.py": """def summarize_checkout(cart: dict) -> dict[str, int]:
    subtotal_cents = sum(item["price_cents"] * item["quantity"] for item in cart["items"])
    coupon = cart.get("coupon")
    discount_cents = 0
    if coupon and coupon["type"] == "flat":
        discount_cents = coupon["amount_cents"]
    total_cents = subtotal_cents - discount_cents + cart.get("shipping_cents", 0)
    return {
        "subtotal_cents": subtotal_cents,
        "discount_cents": discount_cents,
        "total_cents": total_cents,
    }
""",
            "src/summary_builder.py": """def build_summary_line(summary: dict[str, int]) -> str:
    return f"subtotal={summary['subtotal_cents']} discount={summary['discount_cents']} total={summary['total_cents']}"
""",
            "tests/test_pricing_rollup.py": """from src.pricing_rollup import summarize_checkout
from src.summary_builder import build_summary_line


def test_flat_coupon_is_capped_before_shipping() -> None:
    cart = {
        "items": [{"sku": "starter", "price_cents": 5000, "quantity": 1}],
        "coupon": {"type": "flat", "amount_cents": 7000},
        "shipping_cents": 1200,
    }
    summary = summarize_checkout(cart)
    assert summary == {
        "subtotal_cents": 5000,
        "discount_cents": 5000,
        "total_cents": 1200,
    }
    assert build_summary_line(summary) == "subtotal=5000 discount=5000 total=1200"
""",
            "docs/checkout_contract.md": """Flat coupon rules:
- flat coupons reduce subtotal but cannot exceed it
- shipping is added after discounts
- receipts should never show negative merchandise totals
""",
        },
        expected_bug_keywords=["flat", "coupon", "cap", "subtotal", "negative", "receipt", "shipping"],
        expected_fix_keywords=["min", "subtotal", "cap", "discount", "shipping"],
        clarification_hints={
            "coupon": "Flat coupons are capped at the merchandise subtotal. Shipping is applied after discounts.",
            "shipping": "Shipping is added after discounts, so an oversized coupon should zero out merchandise but not create a negative total.",
            "default": "Checkout totals must honor the contract in docs/checkout_contract.md: flat coupons cannot exceed subtotal.",
        },
        validator=_validate_frontier,
    ),
}


def get_task(task_id: str) -> TaskSpec:
    return TASKS[task_id]


def list_tasks() -> List[TaskSpec]:
    return [TASKS[key] for key in sorted(TASKS)]
