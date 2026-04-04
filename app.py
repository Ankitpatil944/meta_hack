from __future__ import annotations

from threading import Lock
from typing import Dict, List

from fastapi import FastAPI, Header, HTTPException

from env import CodeReviewEnv
from inference import choose_heuristic_action
from models import (
    Action,
    GraderRequest,
    GraderResponse,
    PublicEnvironmentState,
    ResetRequest,
    ResetResponse,
    StepResponse,
    TaskSummary,
)
from tasks import get_task


app = FastAPI(
    title="AI Code Review & Bug Reasoning Environment",
    description="OpenEnv-style environment for code review, bug reasoning, and patch validation.",
    version="0.1.0",
)
DEFAULT_SESSION_ID = "default"
ENVIRONMENTS: Dict[str, CodeReviewEnv] = {}
ENVIRONMENTS_LOCK = Lock()


def _resolve_session_id(session_id: str | None) -> str:
    return (session_id or DEFAULT_SESSION_ID).strip() or DEFAULT_SESSION_ID


def _get_env(session_id: str | None) -> CodeReviewEnv:
    resolved_session_id = _resolve_session_id(session_id)
    with ENVIRONMENTS_LOCK:
        env = ENVIRONMENTS.get(resolved_session_id)
        if env is None:
            env = CodeReviewEnv(max_steps=6)
            ENVIRONMENTS[resolved_session_id] = env
        return env


def _task_summaries() -> List[Dict[str, str]]:
    return _get_env(DEFAULT_SESSION_ID).task_summaries()


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "AI Code Review & Bug Reasoning Environment"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest, x_session_id: str | None = Header(default=None)) -> ResetResponse:
    env = _get_env(x_session_id)
    if request.task_id is not None and request.task_id not in {task["task_id"] for task in _task_summaries()}:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {request.task_id}")
    return ResetResponse(observation=env.reset(request.task_id))


@app.post("/step", response_model=StepResponse)
def step(action: Action, x_session_id: str | None = Header(default=None)) -> StepResponse:
    env = _get_env(x_session_id)
    observation, reward, done, info = env.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=PublicEnvironmentState)
def state(x_session_id: str | None = Header(default=None)) -> PublicEnvironmentState:
    env = _get_env(x_session_id)
    return env.public_state()


@app.get("/tasks", response_model=List[TaskSummary])
def tasks() -> List[TaskSummary]:
    return [TaskSummary(**task) for task in _task_summaries()]


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest, x_session_id: str | None = Header(default=None)) -> GraderResponse:
    env = _get_env(x_session_id)
    if request.history is not None:
        if not request.task_id:
            raise HTTPException(status_code=400, detail="task_id is required when grading an explicit history.")
        current_reviewer_history = [entry for entry in env.state().history if entry.actor == "reviewer"]
        if env.state().task_id == request.task_id and request.history == current_reviewer_history:
            return env.grade_current_episode()
        from grader import evaluate_history

        return evaluate_history(get_task(request.task_id), request.history or [])
    if env.state().task_id is None:
        raise HTTPException(status_code=400, detail="Environment must be reset before grading the active episode.")
    return env.grade_current_episode()


@app.get("/baseline")
def baseline() -> Dict:
    env = CodeReviewEnv(max_steps=6)
    results = []
    for task in env.task_summaries():
        observation = env.reset(task["task_id"])
        trajectory = []
        for step_index in range(env.max_steps):
            action_payload = choose_heuristic_action(observation.model_dump(), step_index)
            observation, reward, done, _ = env.step(Action(**action_payload))
            trajectory.append({"action": action_payload, "reward": reward.value})
            if done:
                break
        grade = env.grade_current_episode()
        results.append(
            {
                "task_id": task["task_id"],
                "score": grade.score,
                "components": grade.components,
                "trajectory": trajectory,
            }
        )
    average_score = round(sum(item["score"] for item in results) / len(results), 4)
    return {"results": results, "average_score": average_score}
