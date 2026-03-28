from __future__ import annotations

from env import CodeReviewEnv
from inference import choose_heuristic_action
from models import Action


def main() -> None:
    env = CodeReviewEnv()
    observation = env.reset("frontier_discount_rollup")
    print(f"Task: {observation.task_id}")
    print(f"PR: {observation.pr_title}")

    for step_index in range(env.max_steps):
        action_payload = choose_heuristic_action(observation.model_dump(), step_index)
        print(f"Step {step_index + 1}: {action_payload['action_type']} -> {action_payload['content'][:100]}")
        observation, reward, done, _ = env.step(Action(**action_payload))
        print(f"Reward: {reward.value:.2f} | Reason: {reward.reason}")
        if done:
            break

    grade = env.grade_current_episode()
    print(f"Final score: {grade.score:.2f}")
    print(f"Components: {grade.components}")
    print(f"Reasoning path: {env.state().reasoning_trace}")
    print(f"Mistakes: {env.state().mistakes}")


if __name__ == "__main__":
    main()
