# AI Code Review & Bug Reasoning Environment

This project implements a real-world OpenEnv-style environment for pull request review. An agent reads realistic PR metadata, failing tests, and a code diff, then takes review actions over a trajectory: identify the bug, suggest a patch, comment on the PR, ask for clarification, and decide whether to approve or request changes.

The environment is deterministic and designed for agent training and evaluation. It includes dense reward shaping, seven curated tasks spanning easy, medium, and hard review episodes, a maintainer reply simulator for ambiguous reviews, action/mistake logging, and deterministic graders that validate proposed fixes against task-local contract checks.

Each task is implemented as a task family with observable variants. On reset, the environment cycles through different PR metadata, diff noise, and failing-test context for the same underlying bug, and several families also rotate visible labels such as UI surface names or rollout terminology. This reduces answer-key memorization while preserving deterministic grading.

## Why this is novel

- interactive PR review instead of one-shot classification
- patch-based validation instead of answer-string matching
- deterministic grading with task-local tests
- clarification and maintainer replies for ambiguous reviews
- frontier-style hard tasks with multi-file diffs and misleading edge-case failures
- task-family variants that rotate visible PR context and distractor diffs at reset time

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
|-- app.py
|-- env.py
|-- grader.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- README.md
|-- requirements.txt
|-- tasks.py
`-- Dockerfile
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

`state()` returns a typed public runtime state for the active episode:

- episode metadata
- clarification and maintainer reply markers
- action log, mistakes, and final reasoning trace for debugging/demo use
- step counts and cumulative reward

Hidden scoring fields such as bug truth and correctness trackers stay internal so agents cannot read oracle signals through the public API.

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

### Medium: `medium_receipt_format_cleanup`

This task is intentionally safe. The PR simplifies receipt line formatting without changing behavior, and the tests remain green. A good reviewer should approve the PR rather than hallucinate a regression. This breaks the benchmark pattern where every task requires rejection.

### Hard: `hard_feature_flags`

The PR changes partial feature-flag merges to use truthiness. The agent must reason about the difference between `None` and explicit `False`, and may ask for clarification before suggesting a patch and requesting changes.

### Hard: `hard_billing_suspension`

This task models billing automation. The visible failing test points at the grace-period boundary, so an obvious fix is to change `>=` back to `>`. That is necessary but not sufficient: the billing contract also says active payment plans pause suspension. The validator checks both conditions, so only a contract-correct fix passes.

### Frontier Hard: `frontier_incident_rollout`

This task is intentionally mixed-signal. The PR touches incident paging logic and a nearby rollout-note helper. The visible failing test points to the severity-threshold boundary, but the rollout also drops the customer-impacting override and replaces it with an irrelevant VIP shortcut. A threshold-only fix is still wrong, so the reviewer has to separate the cosmetic diff from the real contract regression.

### Frontier Hard: `frontier_discount_rollup`

This task spans checkout calculation and receipt rendering. Most carts look fine, but an oversized flat coupon creates a negative merchandise total on an edge case with shipping. The agent must combine multi-file diff context, a contract doc, and misleadingly narrow test evidence to propose the right capped-discount fix.

## Baseline scores

Deterministic local baseline (`GET /baseline`, heuristic trajectory generator):

- `easy_keyword_preview`: `1.0`
- `medium_job_retry`: `1.0`
- `medium_receipt_format_cleanup`: `1.0`
- `hard_feature_flags`: `1.0`
- `hard_billing_suspension`: `1.0`
- `frontier_incident_rollout`: `1.0`
- `frontier_discount_rollup`: `1.0`
- Average: `1.0`

Canonical submission runner:

- `inference.py` is the required root-level submission script.
- It uses the OpenAI client and reads `OPENAI_API_KEY` or `HF_TOKEN`, plus `API_BASE_URL`, `MODEL_NAME`, and optional `ENV_BASE_URL`.
- The deterministic `/baseline` route is for local smoke testing; `inference.py` is for external reproduction.

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
set ENV_BASE_URL=http://127.0.0.1:7860
python inference.py
```

If you are routing model calls through an OpenAI-compatible provider, also set:

```bash
set API_BASE_URL=https://your-model-endpoint/v1
```

If you want a local deterministic smoke test without a model key, hit `/baseline`. That route uses the bundled heuristic trajectory generator and returns per-task scores, but `inference.py` is the canonical submission baseline script.

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
- `API_BASE_URL` for the OpenAI-compatible model endpoint
- `MODEL_NAME`
- `HF_TOKEN` if you want to reuse the same token in hosted runs
- `ENV_BASE_URL` for the environment server URL when not using the default `http://127.0.0.1:7860`

## OpenEnv validation

The Docker image installs `openenv-core`, so local and container validation use the same dependency set.

Install the framework locally if needed:

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

## Why equivalent fixes are accepted

The grader is behavior-based, not patch-string-based. Submitted unified diffs are applied to the task snapshot and then validated by task-specific checks. That means logically equivalent fixes are accepted as long as they restore the intended behavior.

Examples:

- `hard_feature_flags`: both `if value is not None:` and other equivalent tri-state-safe rewrites are accepted if explicit `False` updates work and `None` preserves the current value.
- `medium_job_retry`: both “skip exhausted jobs” and “append only when attempts remain” styles are accepted if the resulting function returns only failed jobs with retry budget left.
- `frontier_incident_rollout`: fixes are accepted only if they restore both the strict-above-threshold rule and the customer-impacting override, even if the visible failing test only highlights the threshold boundary.

## Failure cases

The grader also rejects plausible but incomplete or harmful actions.

- `easy_keyword_preview`: returning the last cleaned token instead of joining all cleaned tokens is rejected as an invalid fix.
- `easy_keyword_preview`: approving the PR immediately is penalized because the code is still buggy.
- `hard_billing_suspension`: changing the boundary from `>=` to `>` without restoring the active payment plan exception is still rejected, even though it fixes the visible failing test.
- `frontier_incident_rollout`: changing only the threshold boundary is still rejected because customer-impacting incidents must also retain their immediate paging override.
- `medium_receipt_format_cleanup`: inventing a bug and requesting changes is the wrong decision; the correct behavior is to approve the safe refactor.

## Limitations

- The tasks are still compact PR snapshots rather than large repository-scale reviews.
- Task families rotate visible context, distractor diffs, and several surface labels, but they are not yet fully randomized at the identifier and literal level.
- The baseline agent includes fallback and recovery logic for robustness; it is a strong reference policy, not a claim that an unconstrained frontier model will solve every task without scaffolding.
- Diagnosis scoring still uses task-specific keyword cues before fix validation, even though final fix acceptance is behavior-based.

## Baseline expectations

The bundled heuristic baseline exposed by `/baseline` currently produces:

- `easy_keyword_preview`: `1.0`
- `medium_job_retry`: `1.0`
- `medium_receipt_format_cleanup`: `1.0`
- `hard_feature_flags`: `1.0`
- `hard_billing_suspension`: `1.0`
- `frontier_incident_rollout`: `1.0`
- `frontier_discount_rollup`: `1.0`
- average score: `1.0`

The official benchmark entrypoint is root-level `inference.py`, which uses the OpenAI client and the required submission env vars.
