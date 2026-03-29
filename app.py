from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, HTTPException

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
ENV = CodeReviewEnv(max_steps=6)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "AI Code Review & Bug Reasoning Environment"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    if request.task_id is not None and request.task_id not in {task["task_id"] for task in ENV.task_summaries()}:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {request.task_id}")
    return ResetResponse(observation=ENV.reset(request.task_id))


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    observation, reward, done, info = ENV.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=PublicEnvironmentState)
def state() -> PublicEnvironmentState:
    return ENV.public_state()


@app.get("/tasks", response_model=List[TaskSummary])
def tasks() -> List[TaskSummary]:
    return [TaskSummary(**task) for task in ENV.task_summaries()]


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest) -> GraderResponse:
    if request.history is not None:
        if not request.task_id:
            raise HTTPException(status_code=400, detail="task_id is required when grading an explicit history.")
        current_reviewer_history = [entry for entry in ENV.state().history if entry.actor == "reviewer"]
        if ENV.state().task_id == request.task_id and request.history == current_reviewer_history:
            return ENV.grade_current_episode()
        from grader import evaluate_history

        return evaluate_history(get_task(request.task_id), request.history or [])
    if ENV.state().task_id is None:
        raise HTTPException(status_code=400, detail="Environment must be reset before grading the active episode.")
    return ENV.grade_current_episode()


@app.get("/baseline")
def baseline() -> Dict:
    results = []
    for task in ENV.task_summaries():
        observation = ENV.reset(task["task_id"])
        trajectory = []
        for step_index in range(ENV.max_steps):
            action_payload = choose_heuristic_action(observation.model_dump(), step_index)
            observation, reward, done, _ = ENV.step(Action(**action_payload))
            trajectory.append({"action": action_payload, "reward": reward.value})
            if done:
                break
        grade = ENV.grade_current_episode()
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
