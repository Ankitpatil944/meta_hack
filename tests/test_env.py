from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import CodeReviewEnv
from models import Action, ActionType


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
