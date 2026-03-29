from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from grader import bug_identification_matches, evaluate_history, fix_explanation_matches, validate_fix_patch
from models import (
    Action,
    ActionType,
    EnvironmentState,
    HistoryEntry,
    Observation,
    PublicEnvironmentState,
    Reward,
)
from tasks import TaskSpec, get_task, get_task_variant_count, list_tasks


class CodeReviewEnv:
    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps
        self._task_order = [task.task_id for task in list_tasks()]
        self._task_index = 0
        self._task_variant_index = {task_id: 0 for task_id in self._task_order}
        self._task: Optional[TaskSpec] = None
        self._state = EnvironmentState(max_steps=max_steps)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is None:
            task_id = self._task_order[self._task_index]
            self._task_index = (self._task_index + 1) % len(self._task_order)
        variant_index = self._task_variant_index.get(task_id, 0)
        self._task = get_task(task_id, variant_index=variant_index)
        self._task_variant_index[task_id] = variant_index + 1
        self._state = EnvironmentState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task.task_id,
            variant_id=self._task.variant_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self.max_steps,
            bug_present=self._task.bug_present,
            bug_type=self._task.bug_type,
            uncertainty_level=self._task.uncertainty_level,
            history=[],
            discussion_context=[],
            action_log=[],
            mistakes=[],
            reasoning_trace=[],
            cumulative_reward=0.0,
            done=False,
        )
        return self._build_observation(0.0, {})

    def state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def public_state(self) -> PublicEnvironmentState:
        return PublicEnvironmentState(
            episode_id=self._state.episode_id,
            task_id=self._state.task_id,
            variant_id=self._state.variant_id,
            difficulty=self._state.difficulty,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            clarification_requested=self._state.clarification_requested,
            maintainer_replied=self._state.maintainer_replied,
            history=self._state.history,
            discussion_context=self._state.discussion_context,
            action_log=self._state.action_log,
            mistakes=self._state.mistakes,
            reasoning_trace=self._state.reasoning_trace,
            cumulative_reward=self._state.cumulative_reward,
            done=self._state.done,
        )

    def current_task(self) -> TaskSpec:
        if self._task is None:
            return get_task(self._task_order[0])
        return self._task

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._task is None:
            self.reset()
        assert self._task is not None

        if self._state.done:
            reward = Reward(
                value=0.0,
                components={},
                reason="Episode already finished. Reset before taking more actions.",
                cumulative_reward=self._state.cumulative_reward,
            )
            return self._build_observation(0.0, {}), reward, True, {"warning": "episode_done"}

        self._state.step_count += 1
        reward_components: Dict[str, float] = {}
        reward_value = 0.0
        reason_parts: List[str] = []
        mistakes: List[str] = []
        reasoning_note = f"step {self._state.step_count}: {action.action_type.value}"

        if action.action_type == ActionType.IDENTIFY_BUG:
            if not self._state.bug_identified_correctly and bug_identification_matches(self._task, action.content):
                self._state.bug_identified_correctly = True
                reward_components["correct_bug_identification"] = 0.5
                reward_value += 0.5
                reason_parts.append("correctly identified the bug")
            elif not bug_identification_matches(self._task, action.content):
                reward_components["false_positive_bug_identification"] = -0.2
                reward_value -= 0.2
                reason_parts.append("identified the wrong bug")
                mistakes.append("incorrect bug diagnosis")

        elif action.action_type == ActionType.SUGGEST_FIX:
            patch_valid, validation_message = validate_fix_patch(self._task, action.content)
            has_diff = "diff --git" in action.content
            if patch_valid and not self._state.fix_suggested_correctly:
                self._state.fix_suggested_correctly = True
                reward_components["validated_fix"] = 0.3
                reward_value += 0.3
                reason_parts.append("suggested a fix that passes deterministic tests")
            elif not patch_valid and not has_diff and fix_explanation_matches(self._task, action.content):
                reward_components["partial_fix_reasoning"] = 0.1
                reward_value += 0.1
                reason_parts.append("described the right fix direction but did not provide an executable patch")
                mistakes.append("incomplete fix suggestion")
            elif not patch_valid:
                reward_components["invalid_fix"] = -0.1
                reward_value -= 0.1
                reason_parts.append(validation_message)
                mistakes.append("invalid or incomplete fix suggestion")

        elif action.action_type == ActionType.ADD_COMMENT:
            if bug_identification_matches(self._task, action.content):
                reward_components["useful_comment"] = 0.05
                reward_value += 0.05
                reason_parts.append("added a useful review comment")
            else:
                reason_parts.append("added a neutral review comment")

        elif action.action_type == ActionType.ASK_FOR_CLARIFICATION:
            self._state.clarification_requested = True
            self._state.maintainer_replied = True
            reply = self._maintainer_reply(action.content)
            self._state.discussion_context.append(reply)
            if self._task.uncertainty_level == "high":
                reward_components["useful_clarification"] = 0.05
                reward_value += 0.05
                reason_parts.append("asked for clarification on an ambiguous task")
            else:
                reward_components["unnecessary_clarification"] = -0.05
                reward_value -= 0.05
                reason_parts.append("asked for clarification even though the evidence was already sufficient")
                mistakes.append("unnecessary clarification request")

        elif action.action_type == ActionType.APPROVE_PR:
            self._state.final_decision = ActionType.APPROVE_PR.value
            self._state.done = True
            if self._task.bug_present and not self._state.fix_suggested_correctly:
                reward_components["approved_buggy_pr"] = -0.5
                reward_value -= 0.5
                reason_parts.append("approved a buggy pull request")
                mistakes.append("approved buggy code")
            elif not self._task.bug_present:
                reward_components["correct_approval"] = 0.8
                reward_value += 0.8
                reason_parts.append("correctly approved a behavior-preserving pull request")
            else:
                reason_parts.append("approved the pull request")

        elif action.action_type == ActionType.REQUEST_CHANGES:
            self._state.final_decision = ActionType.REQUEST_CHANGES.value
            self._state.done = True
            reason_parts.append("requested changes on the pull request")

        step_penalty = round(0.05 * self._state.step_count, 4)
        reward_components["efficiency_penalty"] = reward_components.get("efficiency_penalty", 0.0) - step_penalty
        reward_value -= step_penalty
        reason_parts.append(f"incurred efficiency penalty for step {self._state.step_count}")

        history_entry = HistoryEntry(
            step=self._state.step_count,
            actor="reviewer",
            action_type=action.action_type.value,
            content=action.content,
            reward_delta=reward_value,
        )
        self._state.history.append(history_entry)
        self._state.action_log.append(f"{action.action_type.value}: {action.content.strip()[:120]}")
        self._state.mistakes.extend(mistakes)
        if reason_parts:
            reasoning_note = f"{reasoning_note} -> {'; '.join(reason_parts)}"
        self._state.reasoning_trace.append(reasoning_note)
        if action.action_type == ActionType.ASK_FOR_CLARIFICATION and self._state.discussion_context:
            maintainer_entry = HistoryEntry(
                step=self._state.step_count,
                actor="maintainer",
                action_type="maintainer_reply",
                content=self._state.discussion_context[-1],
                reward_delta=0.0,
            )
            self._state.history.append(maintainer_entry)

        if self._state.done and self._state.final_decision == ActionType.REQUEST_CHANGES.value:
            if self._state.bug_identified_correctly and self._state.fix_suggested_correctly:
                reward_components["fully_correct_review"] = reward_components.get("fully_correct_review", 0.0) + 1.0
                reward_value += 1.0
                reason_parts.append("completed a fully correct review")
        elif self._state.done and self._state.final_decision == ActionType.APPROVE_PR.value:
            if not self._task.bug_present:
                reward_components["fully_correct_review"] = reward_components.get("fully_correct_review", 0.0) + 0.6
                reward_value += 0.6
                reason_parts.append("completed a correct approval review")

        if self._state.step_count >= self.max_steps:
            self._state.done = True
            if not self._state.final_decision:
                self._state.final_decision = "max_steps_reached"
                reason_parts.append("reached the step limit")

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward_value, 4)
        reward = Reward(
            value=round(reward_value, 4),
            components=reward_components,
            reason="; ".join(reason_parts) if reason_parts else "no significant change",
            cumulative_reward=self._state.cumulative_reward,
        )
        observation = self._build_observation(reward.value, reward_components)
        info = {
            "task_id": self._task.task_id,
            "bug_identified_correctly": self._state.bug_identified_correctly,
            "fix_suggested_correctly": self._state.fix_suggested_correctly,
            "final_decision": self._state.final_decision,
        }
        return observation, reward, self._state.done, info

    def grade_current_episode(self):
        if self._task is None:
            raise RuntimeError("Environment has not been reset.")
        history = [entry for entry in self._state.history if entry.actor == "reviewer"]
        return evaluate_history(self._task, history)

    def task_summaries(self) -> List[Dict[str, str]]:
        summaries: List[Dict[str, str]] = []
        for task in list_tasks():
            summaries.append(
                {
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "pr_title": task.pr_title,
                    "summary": task.summary,
                }
            )
        return summaries

    def _build_observation(self, reward_value: float, reward_breakdown: Dict[str, float]) -> Observation:
        task = self.current_task()
        return Observation(
            task_id=task.task_id,
            variant_id=task.variant_id,
            difficulty=task.difficulty,
            uncertainty_level=task.uncertainty_level,
            review_hint=task.review_hint,
            pr_title=task.pr_title,
            commit_message=task.commit_message,
            changed_files=task.changed_files,
            code_diff=task.code_diff,
            test_results=task.test_results,
            issue_description=task.issue_description,
            history=self._state.history,
            discussion_context=self._state.discussion_context,
            reasoning_trace=self._state.reasoning_trace,
            remaining_steps=max(self.max_steps - self._state.step_count, 0),
            done=self._state.done,
            reward=reward_value,
            reward_breakdown=reward_breakdown,
        )

    def _maintainer_reply(self, question: str) -> str:
        normalized_question = question.lower()
        for key, reply in self.current_task().clarification_hints.items():
            if key != "default" and key in normalized_question:
                return reply
        return self.current_task().clarification_hints.get(
            "default",
            "Please review the diff and failing tests; they contain the needed context.",
        )
