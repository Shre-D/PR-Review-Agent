from __future__ import annotations

from ..compat import EnvClient, StepResult
from ..models import PRReviewAction, PRReviewObservation, PRReviewState


class PRReviewEnvClient(EnvClient[PRReviewAction, PRReviewObservation, PRReviewState]):

    def _step_payload(self, action: PRReviewAction) -> dict:
        """Serialize action for the /step POST body."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[PRReviewObservation]:
        """Parse /reset and /step responses.
        Handles both:
          - envelope: {"observation": {...}, "reward": float|None, "done": bool}
          - flat: the whole payload is the observation dict
        Sets observation.reward and observation.done from top-level fields.
        """
        if "observation" in payload:
            observation_payload = payload["observation"]
            reward = payload.get("reward")
            done = payload.get("done", False)
        else:
            observation_payload = payload
            reward = payload.get("reward")
            done = payload.get("done", False)
        observation = PRReviewObservation.model_validate(observation_payload)
        if reward is not None:
            observation.reward = reward
        observation.done = done
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> PRReviewState:
        return PRReviewState.model_validate(payload)
