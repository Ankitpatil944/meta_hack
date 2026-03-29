from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field, replace
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
    review_hint: str
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
    diagnosis_concepts: List[List[str]] = field(default_factory=list)
    variant_id: str = "base"
    clarification_hints: Dict[str, str] = field(default_factory=dict)
    validator: Optional[Validator] = None


@dataclass(frozen=True)
class TaskVariantSpec:
    variant_id: str
    pr_title: Optional[str] = None
    summary: Optional[str] = None
    review_hint: Optional[str] = None
    commit_message: Optional[str] = None
    changed_files: Optional[List[str]] = None
    code_diff_prefix: str = ""
    code_diff_suffix: str = ""
    test_results_prefix: str = ""
    test_results_suffix: str = ""
    issue_description: Optional[str] = None
    clarification_hints: Optional[Dict[str, str]] = None
    randomization_tokens: Dict[str, List[str]] = field(default_factory=dict)


def _replace_tokens(value: str, randomization_tokens: Dict[str, List[str]], seed_index: int) -> str:
    output = value
    for token, options in randomization_tokens.items():
        if not options:
            continue
        replacement = options[seed_index % len(options)]
        output = output.replace(f"{{{token}}}", replacement)
    return output


def _materialize_variant(base: TaskSpec, variant: TaskVariantSpec, seed_index: int = 0) -> TaskSpec:
    changed_files = list(base.changed_files)
    if variant.changed_files is not None:
        changed_files = variant.changed_files

    randomization_tokens = variant.randomization_tokens
    if randomization_tokens:
        changed_files = [_replace_tokens(path, randomization_tokens, seed_index) for path in changed_files]

    clarification_hints = dict(base.clarification_hints)
    if variant.clarification_hints:
        clarification_hints.update(variant.clarification_hints)
    if randomization_tokens:
        clarification_hints = {
            _replace_tokens(key, randomization_tokens, seed_index): _replace_tokens(value, randomization_tokens, seed_index)
            for key, value in clarification_hints.items()
        }

    code_diff_parts = [part for part in [variant.code_diff_prefix, base.code_diff, variant.code_diff_suffix] if part]
    test_result_parts = [
        part for part in [variant.test_results_prefix, base.test_results, variant.test_results_suffix] if part
    ]

    code_diff = "\n\n".join(code_diff_parts)
    test_results = "\n".join(test_result_parts)
    pr_title = variant.pr_title or base.pr_title
    summary = variant.summary or base.summary
    review_hint = variant.review_hint or base.review_hint
    commit_message = variant.commit_message or base.commit_message
    issue_description = variant.issue_description if variant.issue_description is not None else base.issue_description

    if randomization_tokens:
        pr_title = _replace_tokens(pr_title, randomization_tokens, seed_index)
        summary = _replace_tokens(summary, randomization_tokens, seed_index)
        review_hint = _replace_tokens(review_hint, randomization_tokens, seed_index)
        commit_message = _replace_tokens(commit_message, randomization_tokens, seed_index)
        code_diff = _replace_tokens(code_diff, randomization_tokens, seed_index)
        test_results = _replace_tokens(test_results, randomization_tokens, seed_index)
        if issue_description is not None:
            issue_description = _replace_tokens(issue_description, randomization_tokens, seed_index)

    return replace(
        base,
        variant_id=variant.variant_id,
        pr_title=pr_title,
        summary=summary,
        review_hint=review_hint,
        commit_message=commit_message,
        changed_files=changed_files,
        code_diff=code_diff,
        test_results=test_results,
        issue_description=issue_description,
        clarification_hints=clarification_hints,
    )


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


def _validate_suspension(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_hard_billing_suspension", workspace / "src" / "billing_suspension.py")
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


def _validate_approval(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_medium_receipt_format", workspace / "src" / "receipt_format.py")
    build_receipt_line = getattr(module, "build_receipt_line")

    line = build_receipt_line("starter", 2500, 2)
    if line != "starter x2 - $50.00":
        return False, "Receipt formatting should stay unchanged for standard line items."

    discounted = build_receipt_line("pro", 1250, 1)
    if discounted != "pro x1 - $12.50":
        return False, "Receipt formatting should preserve quantity and currency formatting."

    return True, "All receipt formatting tests passed."


def _validate_incident_rollout(workspace: Path) -> Tuple[bool, str]:
    module = _load_module("task_frontier_incident_rollout", workspace / "src" / "incident_rollout.py")
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


TASKS: Dict[str, TaskSpec] = {
    "easy_keyword_preview": TaskSpec(
        task_id="easy_keyword_preview",
        difficulty="easy",
        pr_title="Speed up keyword previews on ticket cards",
        summary="Reviewer must catch that the preview only returns the first cleaned keyword instead of the full summary.",
        review_hint="Important: preserve all normalized tokens in the preview string, not just the first token.",
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
        diagnosis_concepts=[["first", "only"], ["preview", "keyword"], ["join", "all", "cleaned"]],
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
        review_hint="Important: only failed jobs with remaining retry attempts should be returned.",
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
        diagnosis_concepts=[
            ["retry", "attempt", "max_attempts"],
            ["failed", "retryable", "re-queued", "returned"],
            ["remaining", "less than", "budget", "exhausted", "skip"],
        ],
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
        review_hint="Important: False is a valid override and must not be ignored. None means keep the current value.",
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
        diagnosis_concepts=[
            ["false", "none", "null"],
            ["truthy", "ignore", "explicit"],
            ["disable", "preserve", "current", "flag"],
        ],
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
        review_hint="Important: flat discounts must never exceed subtotal, and shipping is applied after discounts.",
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
        diagnosis_concepts=[
            ["flat", "coupon", "discount"],
            ["subtotal", "cap", "capped"],
            ["negative", "receipt", "shipping", "total"],
        ],
        clarification_hints={
            "coupon": "Flat coupons are capped at the merchandise subtotal. Shipping is applied after discounts.",
            "shipping": "Shipping is added after discounts, so an oversized coupon should zero out merchandise but not create a negative total.",
            "default": "Checkout totals must honor the contract in docs/checkout_contract.md: flat coupons cannot exceed subtotal.",
        },
        validator=_validate_frontier,
    ),
    "hard_billing_suspension": TaskSpec(
        task_id="hard_billing_suspension",
        difficulty="hard",
        pr_title="Simplify overdue-account suspension checks in billing control loop",
        summary=(
            "Reviewer must catch that the refactor changed both the grace-period boundary and the payment-plan exception. "
            "A boundary-only fix looks plausible from the failing test, but it still violates the billing contract."
        ),
        review_hint="Important: accounts exactly at the grace boundary should not suspend, and active payment plans override suspension.",
        commit_message="cleanup: collapse billing suspension predicate into a single return statement",
        changed_files=[
            "src/billing_suspension.py",
            "tests/test_billing_suspension.py",
            "docs/billing_suspension_contract.md",
        ],
        code_diff="""diff --git a/src/billing_suspension.py b/src/billing_suspension.py
index 87ac120..2af1cc1 100644
--- a/src/billing_suspension.py
+++ b/src/billing_suspension.py
@@ -1,5 +1,5 @@
 def should_suspend_account(account: dict) -> bool:
     if account["invoice_status"] != "unpaid":
         return False
-    return account["overdue_days"] > account["grace_days"] and not account.get("active_payment_plan", False)
+    return account["overdue_days"] >= account["grace_days"]

diff --git a/docs/billing_suspension_contract.md b/docs/billing_suspension_contract.md
index d4ac993..d4ac993 100644
--- a/docs/billing_suspension_contract.md
+++ b/docs/billing_suspension_contract.md
@@ -1,4 +1,4 @@
 Suspension contract:
 - suspend only when unpaid invoices are past the grace window
 - accounts exactly at the grace boundary remain active until the next day
 - active payment plans pause automated suspension
""",
        test_results="""FAILED tests/test_billing_suspension.py::test_account_at_grace_boundary_stays_active
E   AssertionError: assert True is False
E     Boundary accounts should remain active until they are past the grace window.
""",
        issue_description="Billing automation should not suspend accounts that are merely at the grace boundary, and payment plans still pause suspension.",
        bug_present=True,
        bug_type="billing_suspension_contract_regression",
        uncertainty_level="high",
        files_before={
            "src/billing_suspension.py": """def should_suspend_account(account: dict) -> bool:
    if account["invoice_status"] != "unpaid":
        return False
    return account["overdue_days"] >= account["grace_days"]
""",
            "tests/test_billing_suspension.py": """from src.billing_suspension import should_suspend_account


def test_account_at_grace_boundary_stays_active() -> None:
    account = {
        "invoice_status": "unpaid",
        "overdue_days": 14,
        "grace_days": 14,
        "active_payment_plan": False,
    }
    assert should_suspend_account(account) is False
""",
            "docs/billing_suspension_contract.md": """Suspension contract:
- suspend only when unpaid invoices are past the grace window
- accounts exactly at the grace boundary remain active until the next day
- active payment plans pause automated suspension
""",
        },
        expected_bug_keywords=[
            "boundary",
            "grace",
            "payment plan",
            "suspend",
            "unpaid",
            "past",
        ],
        expected_fix_keywords=["grace", "payment plan", "active_payment_plan", "overdue", "strictly"],
        diagnosis_concepts=[
            ["grace", "boundary", "past"],
            ["payment plan", "active_payment_plan"],
            ["suspend", "overdue", "unpaid"],
        ],
        clarification_hints={
            "payment plan": "Active payment plans pause automated suspension, even when the invoice is overdue.",
            "grace": "The account should suspend only after it is past the grace window, not when overdue_days equals grace_days.",
            "default": "Billing suspension requires both conditions: past grace and no active payment plan.",
        },
        validator=_validate_suspension,
    ),
    "medium_receipt_format_cleanup": TaskSpec(
        task_id="medium_receipt_format_cleanup",
        difficulty="medium",
        pr_title="Refactor receipt line formatting helper for readability",
        summary=(
            "Reviewer must recognize that the PR is safe. The diff is a formatting cleanup with no behavior change, "
            "and the correct trajectory is to approve the PR rather than invent a bug."
        ),
        review_hint="Important: this PR should preserve existing receipt formatting behavior; do not reject harmless cleanups.",
        commit_message="refactor: simplify receipt line formatting without changing output",
        changed_files=["src/receipt_format.py", "tests/test_receipt_format.py"],
        code_diff="""diff --git a/src/receipt_format.py b/src/receipt_format.py
index 441aa32..553bc71 100644
--- a/src/receipt_format.py
+++ b/src/receipt_format.py
@@ -1,3 +1,4 @@
 def build_receipt_line(name: str, price_cents: int, quantity: int) -> str:
     total_cents = price_cents * quantity
-    return f"{name} x{quantity} - ${total_cents / 100:.2f}"
+    total_dollars = total_cents / 100
+    return f"{name} x{quantity} - ${total_dollars:.2f}"
""",
        test_results="""PASSED tests/test_receipt_format.py::test_receipt_line_uses_quantity_and_currency_formatting
PASSED tests/test_receipt_format.py::test_receipt_line_handles_single_quantity
""",
        issue_description="The refactor should not change receipt formatting behavior.",
        bug_present=False,
        bug_type="no_regression",
        uncertainty_level="low",
        files_before={
            "src/receipt_format.py": """def build_receipt_line(name: str, price_cents: int, quantity: int) -> str:
    total_cents = price_cents * quantity
    total_dollars = total_cents / 100
    return f"{name} x{quantity} - ${total_dollars:.2f}"
""",
            "tests/test_receipt_format.py": """from src.receipt_format import build_receipt_line


def test_receipt_line_uses_quantity_and_currency_formatting() -> None:
    assert build_receipt_line("starter", 2500, 2) == "starter x2 - $50.00"


def test_receipt_line_handles_single_quantity() -> None:
    assert build_receipt_line("pro", 1250, 1) == "pro x1 - $12.50"
""",
        },
        expected_bug_keywords=["regression", "bug", "format", "wrong", "broken", "currency"],
        expected_fix_keywords=["preserve", "format", "currency", "quantity"],
        diagnosis_concepts=[],
        clarification_hints={
            "default": "This refactor should preserve the exact same receipt output. There is no known regression here.",
        },
        validator=_validate_approval,
    ),
    "frontier_incident_rollout": TaskSpec(
        task_id="frontier_incident_rollout",
        difficulty="hard",
        pr_title="Unify incident paging thresholds with rollout-safe defaults",
        summary=(
            "Reviewer must catch a mixed-signal incident-management regression. The visible failing test points to the threshold boundary, "
            "but the rollout also drops the customer-impacting override. A boundary-only fix is still wrong."
        ),
        review_hint="Important: incidents should page only above the threshold unless they are customer-impacting, which pages regardless.",
        commit_message="refactor: collapse incident paging logic and tidy nearby rollout notes",
        changed_files=[
            "src/incident_rollout.py",
            "src/rollout_notes.py",
            "docs/incident_paging_contract.md",
            "tests/test_incident_rollout.py",
        ],
        code_diff="""diff --git a/src/incident_rollout.py b/src/incident_rollout.py
index 55ad221..8bc9911 100644
--- a/src/incident_rollout.py
+++ b/src/incident_rollout.py
@@ -1,7 +1,7 @@
 def should_page_incident(incident: dict) -> bool:
     threshold = incident.get("page_threshold", 4)
-    if incident.get("customer_impacting", False):
-        return True
-    return incident["severity"] > threshold
+    if incident.get("customer_tier") == "vip":
+        return True
+    return incident["severity"] >= threshold

diff --git a/src/rollout_notes.py b/src/rollout_notes.py
index 88ba110..88ba223 100644
--- a/src/rollout_notes.py
+++ b/src/rollout_notes.py
@@ -1,3 +1,3 @@
 def render_rollout_note(flag_name: str) -> str:
-    return f"rollout-note:{flag_name}"
+    return f"rollout:{flag_name}"

diff --git a/docs/incident_paging_contract.md b/docs/incident_paging_contract.md
index 0daaa10..0daaa10 100644
--- a/docs/incident_paging_contract.md
+++ b/docs/incident_paging_contract.md
@@ -1,4 +1,4 @@
 Paging contract:
 - page when severity is strictly above the configured threshold
 - customer-impacting incidents always page immediately
 - cosmetic rollout note changes must not affect paging behavior
""",
        test_results="""FAILED tests/test_incident_rollout.py::test_threshold_boundary_does_not_page
E   AssertionError: assert True is False
E     Incidents exactly at the threshold should remain non-paging until severity is above the threshold.
""",
        issue_description="Rollout-safe paging should still preserve the customer-impacting override and the strict-above-threshold rule.",
        bug_present=True,
        bug_type="incident_paging_contract_regression",
        uncertainty_level="high",
        files_before={
            "src/incident_rollout.py": """def should_page_incident(incident: dict) -> bool:
    threshold = incident.get("page_threshold", 4)
    if incident.get("customer_tier") == "vip":
        return True
    return incident["severity"] >= threshold
""",
            "src/rollout_notes.py": """def render_rollout_note(flag_name: str) -> str:
    return f"rollout:{flag_name}"
""",
            "docs/incident_paging_contract.md": """Paging contract:
- page when severity is strictly above the configured threshold
- customer-impacting incidents always page immediately
- cosmetic rollout note changes must not affect paging behavior
""",
            "tests/test_incident_rollout.py": """from src.incident_rollout import should_page_incident


def test_threshold_boundary_does_not_page() -> None:
    incident = {
        "severity": 4,
        "page_threshold": 4,
        "minutes_open": 5,
        "customer_tier": "standard",
        "customer_impacting": False,
    }
    assert should_page_incident(incident) is False
""",
        },
        expected_bug_keywords=["threshold", "customer-impacting", "page", "severity", "strictly above", "override"],
        expected_fix_keywords=["customer-impacting", "threshold", ">", "override", "page"],
        diagnosis_concepts=[
            ["threshold", "strictly above", "boundary", "severity"],
            ["customer-impacting", "impacting", "override"],
            ["page", "paging", "incident"],
        ],
        clarification_hints={
            "customer": "Customer-impacting incidents must still page immediately, regardless of the severity threshold.",
            "threshold": "The threshold is strict: incidents at the threshold should not page unless another override applies.",
            "default": "The rollout keeps two rules: strict-above-threshold paging and customer-impacting override paging.",
        },
        validator=_validate_incident_rollout,
    ),
}


TASK_VARIANTS: Dict[str, List[TaskVariantSpec]] = {
    "easy_keyword_preview": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"ticket_surface": ["ticket card", "queue card", "support card"]},
        ),
        TaskVariantSpec(
            variant_id="ticket-card-noise",
            pr_title="Reduce {ticket_surface} preview allocations during keyword normalization",
            commit_message="perf: trim {ticket_surface} preview formatting path and leave the search index untouched",
            changed_files=[
                "src/keyword_preview.py",
                "src/search_index.py",
                "tests/test_keyword_preview.py",
            ],
            code_diff_suffix="""diff --git a/src/search_index.py b/src/search_index.py
index 18bb112..3c00491 100644
--- a/src/search_index.py
+++ b/src/search_index.py
@@ -8,3 +8,4 @@ def normalize_search_terms(values: list[str]) -> list[str]:
     return [value.strip().lower() for value in values if value and value.strip()]

+# no behavior change in this helper
""",
            test_results_prefix="NOTE: Search indexing still behaves correctly; the regression is isolated to {ticket_surface} preview rendering.",
            issue_description="Support {ticket_surface}s should still render a full normalized preview even after the allocation cleanup.",
            clarification_hints={
                "search": "The search index helper is unchanged. The regression is in the preview string returned to the {ticket_surface} UI."
            },
            randomization_tokens={"ticket_surface": ["ticket card", "queue card", "support card"]},
        ),
    ],
    "medium_job_retry": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"queue_surface": ["retry queue", "worker queue", "job queue"]},
        ),
        TaskVariantSpec(
            variant_id="queue-worker-noise",
            pr_title="Clean up {queue_surface} filtering and worker telemetry formatting",
            commit_message="cleanup: simplify {queue_surface} eligibility checks and reorder worker metrics output",
            changed_files=[
                "src/job_retry.py",
                "src/worker_metrics.py",
                "tests/test_job_retry.py",
            ],
            code_diff_suffix="""diff --git a/src/worker_metrics.py b/src/worker_metrics.py
index 4dfac31..7a91c12 100644
--- a/src/worker_metrics.py
+++ b/src/worker_metrics.py
@@ -1,4 +1,4 @@
 def render_worker_metrics(active_workers: int, queued_jobs: int) -> str:
-    return f"workers={active_workers} queued={queued_jobs}"
+    return f"queued={queued_jobs} workers={active_workers}"
""",
            test_results_suffix="Hint: Only failed jobs with remaining retry attempts should be returned.",
            issue_description="Retry selection should ignore telemetry refactors and only re-queue failed jobs that still have budget left in the {queue_surface}.",
            randomization_tokens={"queue_surface": ["retry queue", "worker queue", "job queue"]},
        ),
    ],
    "hard_feature_flags": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"account_surface": ["account PATCH", "profile PATCH", "tenant PATCH"]},
        ),
        TaskVariantSpec(
            variant_id="contract-doc-noise",
            pr_title="Consolidate {account_surface} flag merge and profile audit formatting",
            commit_message="refactor: reuse {account_surface} flag merge helper while touching nearby audit docs",
            changed_files=[
                "src/feature_flags.py",
                "docs/account_patch_contract.md",
                "docs/account_audit_log.md",
                "tests/test_feature_flags.py",
            ],
            code_diff_suffix="""diff --git a/docs/account_audit_log.md b/docs/account_audit_log.md
index 42aa551..cb18821 100644
--- a/docs/account_audit_log.md
+++ b/docs/account_audit_log.md
@@ -1,3 +1,3 @@
 Audit log semantics:
-record actor and account id
+record account id and actor
 - do not modify flag state here
""",
            test_results_suffix="Hint: Treat False as an explicit value, not as a missing value.",
            issue_description="The {account_surface} contract is tri-state: true enables, false disables, and null preserves the current flag.",
            randomization_tokens={"account_surface": ["account PATCH", "profile PATCH", "tenant PATCH"]},
        ),
    ],
    "hard_billing_suspension": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"collection_surface": ["collections", "billing controls", "dunning"]},
        ),
        TaskVariantSpec(
            variant_id="collections-note-noise",
            pr_title="Refactor {collection_surface} suspension guard and adjacent dunning note formatting",
            commit_message="cleanup: collapse suspension checks while touching {collection_surface} note templates",
            changed_files=[
                "src/billing_suspension.py",
                "docs/billing_suspension_contract.md",
                "templates/dunning_note.txt",
                "tests/test_billing_suspension.py",
            ],
            code_diff_suffix="""diff --git a/templates/dunning_note.txt b/templates/dunning_note.txt
index aa72c11..ea44cc2 100644
--- a/templates/dunning_note.txt
+++ b/templates/dunning_note.txt
@@ -1,2 +1,2 @@
-Please update your billing details.
+Please review your billing details.
 Payment plans remain visible in the customer portal.
""",
            test_results_suffix="Hint: Payment plans still pause suspension after the refactor.",
            issue_description="{collection_surface} messaging changed nearby, but automated suspension still needs both the past-grace rule and the payment-plan exception.",
            randomization_tokens={"collection_surface": ["collections", "billing controls", "dunning"]},
        ),
    ],
    "medium_receipt_format_cleanup": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"receipt_surface": ["receipt", "checkout receipt", "invoice receipt"]},
        ),
        TaskVariantSpec(
            variant_id="cosmetic-template-noise",
            pr_title="Tidy {receipt_surface} formatting helper and adjacent template spacing",
            commit_message="cleanup: reduce inline arithmetic and normalize {receipt_surface} template spacing",
            changed_files=[
                "src/receipt_format.py",
                "templates/receipt_footer.txt",
                "tests/test_receipt_format.py",
            ],
            code_diff_suffix="""diff --git a/templates/receipt_footer.txt b/templates/receipt_footer.txt
index 27aa110..62bc991 100644
--- a/templates/receipt_footer.txt
+++ b/templates/receipt_footer.txt
@@ -1,2 +1,2 @@
-thanks for your order
+thanks for your order
 visit again soon
""",
            issue_description="The formatting cleanup should remain behavior-preserving across {receipt_surface} output.",
            randomization_tokens={"receipt_surface": ["receipt", "checkout receipt", "invoice receipt"]},
        ),
    ],
    "frontier_incident_rollout": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"incident_surface": ["incident", "incident page", "on-call page"]},
        ),
        TaskVariantSpec(
            variant_id="rollout-note-noise",
            pr_title="Unify {incident_surface} thresholds and tidy rollout annotations",
            commit_message="refactor: collapse {incident_surface} paging checks while touching rollout note formatting",
            test_results_suffix="Hint: customer-impacting incidents still page immediately even if a nearby rollout note changed.",
            issue_description="The {incident_surface} rollout should preserve both the strict threshold rule and the customer-impacting override.",
            clarification_hints={
                "rollout": "The rollout note formatting change is cosmetic. The real logic still depends on threshold and customer impact."
            },
            randomization_tokens={"incident_surface": ["incident", "incident page", "on-call page"]},
        ),
    ],
    "frontier_discount_rollup": [
        TaskVariantSpec(
            variant_id="base",
            randomization_tokens={"receipt_surface": ["receipt", "checkout summary", "order summary"]},
        ),
        TaskVariantSpec(
            variant_id="receipt-template-noise",
            pr_title="Share flat coupon logic across checkout totals and {receipt_surface} templates",
            commit_message="refactor: align {receipt_surface} rendering helper with checkout pricing rollup",
            changed_files=[
                "src/pricing_rollup.py",
                "src/summary_builder.py",
                "templates/receipt_header.txt",
                "tests/test_pricing_rollup.py",
                "docs/checkout_contract.md",
            ],
            code_diff_suffix="""diff --git a/templates/receipt_header.txt b/templates/receipt_header.txt
index 12ab331..88cce19 100644
--- a/templates/receipt_header.txt
+++ b/templates/receipt_header.txt
@@ -1,2 +1,2 @@
-thanks for shopping with us
+thanks for shopping with us online
 keep this template cosmetic only
""",
            test_results_prefix="Observed only on oversized flat-coupon carts; standard percentage discounts still pass.",
            issue_description="The {receipt_surface} path should never show negative merchandise totals, even when shipping is present and the flat coupon exceeds subtotal.",
            randomization_tokens={"receipt_surface": ["receipt", "checkout summary", "order summary"]},
        ),
    ],
}


def get_task(task_id: str, variant_index: int = 0) -> TaskSpec:
    base_task = TASKS[task_id]
    variants = TASK_VARIANTS.get(task_id, [TaskVariantSpec(variant_id="base")])
    variant = variants[variant_index % len(variants)]
    return _materialize_variant(base_task, variant, seed_index=variant_index // len(variants))


def get_task_variant_count(task_id: str) -> int:
    return len(TASK_VARIANTS.get(task_id, [TaskVariantSpec(variant_id="base")]))


def list_tasks() -> List[TaskSpec]:
    return [TASKS[key] for key in sorted(TASKS)]
