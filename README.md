# AI Code Review & Bug Reasoning Environment

This project implements a real-world OpenEnv-style environment for pull request review. An agent reads realistic PR metadata, failing tests, and a code diff, then takes review actions over a trajectory: identify the bug, suggest a patch, comment on the PR, ask for clarification, and decide whether to approve or request changes.

The environment is deterministic and designed for agent training and evaluation. It includes dense reward shaping, four curated tasks with increasing difficulty, a maintainer reply simulator for ambiguous reviews, action/mistake logging, and deterministic graders that validate proposed fixes by applying patches and running task-local tests.

## Why this is novel

- interactive PR review instead of one-shot classification
- patch-based validation instead of answer-string matching
- deterministic grading with task-local tests
- clarification and maintainer replies for ambiguous reviews
- frontier-style hard tasks with multi-file diffs and misleading edge-case failures

## Why this environment is useful

Most code review benchmarks are single-turn classification tasks. This environment is closer to an actual software engineering workflow:

- the agent must inspect a realistic PR diff and test output
- the agent can ask for clarification when the contract is ambiguous
- the agent is rewarded for partial progress, not just final success
- fix proposals are validated with deterministic local tests
- bad review behavior, such as approving buggy code or taking too many steps, is penalized
- the environment trains and evaluates code-review agents on reasoning, not just final labels

## Project structure

```text
.
├── app.py
├── env.py
├── grader.py
├── inference.py
├── models.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── tasks.py
└── Dockerfile
```

## Observation space

Each `reset()` or `step()` returns an `Observation` with:

- `task_id`: stable task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `pr_title`: realistic pull request title
- `commit_message`: commit message for the PR
- `changed_files`: files touched by the PR
- `code_diff`: unified diff under review
- `test_results`: failing test output or contract evidence
- `issue_description`: optional hint
- `history`: prior reviewer and maintainer actions
- `discussion_context`: maintainer responses to clarification requests
- `reasoning_trace`: serialized path of the environment's recorded review progression
- `remaining_steps`: steps left before the episode ends
- `done`: whether the episode is over
- `reward`: latest scalar reward
- `reward_breakdown`: latest reward component map

## Action space

The `Action` model supports:

- `identify_bug`
- `suggest_fix`
- `add_comment`
- `approve_pr`
- `request_changes`
- `ask_for_clarification`

`suggest_fix` expects a unified diff patch. The grader attempts to apply the patch to the buggy task snapshot and then runs deterministic task-local tests to validate the fix.

## Hidden environment state

`state()` returns the typed internal state used to score the review:

- episode metadata
- hidden bug flags such as `bug_present` and `bug_type`
- correctness trackers for diagnosis and fix quality
- clarification and maintainer reply markers
- action log, mistakes, and final reasoning trace for debugging/demo use
- step counts, cumulative reward, and final decision

## Reward function

The environment uses dense rewards:

- `+0.5` for first correct bug identification
- `+0.3` for first validated correct fix suggestion
- `-0.5` for approving buggy code
- `-0.2` for false positives or invalid fixes
- `+1.0` for a fully correct review trajectory
- `-0.05 * step_count` on every action to reward efficient review trajectories

Ambiguous tasks also support a small positive signal for useful clarification requests.

## Tasks

### Easy: `easy_keyword_preview`

The PR regresses a ticket keyword preview builder so it returns only the first cleaned token. The agent should notice the mismatch between the diff and the failing preview test, propose a patch that joins all cleaned terms, and reject the PR.

### Medium: `medium_job_retry`

The PR inverts the retry guard for failed background jobs. The agent must infer from the failing test that the queue now selects exhausted jobs while skipping jobs with remaining retry budget.

### Hard: `hard_feature_flags`

The PR changes partial feature-flag merges to use truthiness. The agent must reason about the difference between `None` and explicit `False`, and may ask for clarification before suggesting a patch and requesting changes.

### Frontier Hard: `frontier_discount_rollup`

This task spans checkout calculation and receipt rendering. Most carts look fine, but an oversized flat coupon creates a negative merchandise total on an edge case with shipping. The agent must combine multi-file diff context, a contract doc, and misleadingly narrow test evidence to propose the right capped-discount fix.

## API

The FastAPI server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

### Example usage

Reset to a specific task:

```bash
curl -X POST http://127.0.0.1:7860/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"task_id\": \"medium_job_retry\"}"
```

Take a review step:

```bash
curl -X POST http://127.0.0.1:7860/step ^
  -H "Content-Type: application/json" ^
  -d "{\"action_type\": \"identify_bug\", \"content\": \"The retry budget check is inverted and exhausted jobs are being re-queued.\"}"
```

Inspect internal state:

```bash
curl http://127.0.0.1:7860/state
```

Grade the current episode:

```bash
curl -X POST http://127.0.0.1:7860/grader -H "Content-Type: application/json" -d "{}"
```

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run the baseline:

```bash
set OPENAI_API_KEY=your_key
set MODEL_NAME=gpt-4.1-mini
python inference.py
```

If you want a local deterministic smoke test without a model key, hit `/baseline`. That route uses the bundled heuristic trajectory generator and returns per-task scores.

Run the demo walkthrough:

```bash
python demo_run.py
```

This prints each review step, reward changes, the final score, and the recorded reasoning path.

## Docker

Build and run:

```bash
docker build -t ai-code-review-env .
docker run -p 7860:7860 ai-code-review-env
```

## Hugging Face Spaces

Create a Docker Space, tag it with `openenv`, and push this repository. The `Dockerfile` starts the FastAPI app on port `7860`, which is compatible with Docker Spaces.

Environment variables for the external baseline runner:

- `OPENAI_API_KEY`
- `API_BASE_URL` when routing through a compatible OpenAI-style endpoint
- `MODEL_NAME`
- `HF_TOKEN` if you want to reuse the same token in hosted runs

## OpenEnv validation

Install the framework:

```bash
pip install openenv-core
```

Then validate your manifest and environment wiring:

```bash
openenv validate
```

The exact validation command may evolve with OpenEnv, so verify against the version you install. The environment contract in this repo is centered on the required `reset`, `step`, and `state` API plus typed models and Docker packaging.

## Logging and demos

The environment records:

- actions taken
- reviewer mistakes
- reasoning trace across the episode

These are returned by `state()` and surfaced in `demo_run.py` for debugging and demos.

## Baseline expectations

The bundled heuristic baseline exposed by `/baseline` should achieve high deterministic scores across all four tasks because it uses task-specific review trajectories. The model-driven `inference.py` run is intended for reproducible benchmark scoring once `OPENAI_API_KEY` and `MODEL_NAME` are configured.
