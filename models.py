from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    IDENTIFY_BUG = "identify_bug"
    SUGGEST_FIX = "suggest_fix"
    APPROVE_PR = "approve_pr"
    REQUEST_CHANGES = "request_changes"
    ADD_COMMENT = "add_comment"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"


class Action(BaseModel):
    action_type: ActionType
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    reason: str
    cumulative_reward: float


class HistoryEntry(BaseModel):
    step: int
    actor: str
    action_type: str
    content: str
    reward_delta: float = 0.0


class Observation(BaseModel):
    task_id: str
    variant_id: str = "base"
    difficulty: str
    uncertainty_level: str
    review_hint: Optional[str] = None
    pr_title: str
    commit_message: str
    changed_files: List[str]
    code_diff: str
    test_results: str
    issue_description: Optional[str] = None
    history: List[HistoryEntry] = Field(default_factory=list)
    discussion_context: List[str] = Field(default_factory=list)
    reasoning_trace: List[str] = Field(default_factory=list)
    remaining_steps: int
    done: bool = False
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    variant_id: Optional[str] = None
    difficulty: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    bug_present: bool = True
    bug_type: Optional[str] = None
    uncertainty_level: str = "low"
    bug_identified_correctly: bool = False
    fix_suggested_correctly: bool = False
    clarification_requested: bool = False
    maintainer_replied: bool = False
    final_decision: Optional[str] = None
    history: List[HistoryEntry] = Field(default_factory=list)
    discussion_context: List[str] = Field(default_factory=list)
    action_log: List[str] = Field(default_factory=list)
    mistakes: List[str] = Field(default_factory=list)
    reasoning_trace: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


class PublicEnvironmentState(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    variant_id: Optional[str] = None
    difficulty: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    clarification_requested: bool = False
    maintainer_replied: bool = False
    history: List[HistoryEntry] = Field(default_factory=list)
    discussion_context: List[str] = Field(default_factory=list)
    action_log: List[str] = Field(default_factory=list)
    mistakes: List[str] = Field(default_factory=list)
    reasoning_trace: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ResetResponse(BaseModel):
    observation: Observation


class GraderRequest(BaseModel):
    task_id: Optional[str] = None
    history: Optional[List[HistoryEntry]] = None


class GraderResponse(BaseModel):
    task_id: str
    score: float
    components: Dict[str, float]
    bug_identification_correct: bool
    fix_correct: bool
    decision_correct: bool
    efficiency_penalty: float
    clarification_useful: bool
    summary: str


class TaskSummary(BaseModel):
    task_id: str
    difficulty: str
    pr_title: str
    summary: str
