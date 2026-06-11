"""Stage 2 — high-test-time-compute reflection via LLM-as-a-Judge.

For each gated episode the judge (any :class:`rlm.clients.base_lm.BaseLM`)
is asked to reflect — what went wrong, what required multiple tries, what
was hard to predict — and to distill the lesson into concrete SFT
examples (STaR-style: failures are corrected with hindsight, successes
after struggle are compressed into the direct solution).

Test-time compute is spent two ways:

* ``n_samples`` independent reflections with majority voting on the
  learn/skip verdict (self-consistency), and
* an optional verification pass per distilled example — the
  generation-verification gap is the only supervision we have, so only
  verified examples earn a gradient (see ``docs/learning_signal.md``).
"""

from __future__ import annotations

import json
import re

from rlm.clients.base_lm import BaseLM
from rlm.sleep.config import JudgeConfig
from rlm.sleep.types import Episode, JudgeOutput, TrainingExample

REFLECT_PROMPT = """\
You are a senior engineer reviewing one day's agent trajectories to decide \
what is worth learning from. Reflect carefully on the episode below.

Look for: actions that were wrong, actions that needed multiple tries, \
outcomes that were hard to predict in advance, and reusable lessons.

Respond with ONLY a JSON object:
{{
  "verdict": "learn" or "skip",
  "confidence": <float in [0,1]>,
  "lesson": "<one-sentence reusable lesson, empty if skip>",
  "examples": [
    {{"prompt": "<a realistic task prompt exercising the lesson>",
      "response": "<the ideal assistant response, applying the lesson directly>"}}
  ]
}}

Rules: at most {max_examples} examples; "skip" if the episode is routine \
("nothing new here"); responses must be correct and direct — show the fixed \
approach, not the original mistake.

EPISODE:
{transcript}
"""

VERIFY_PROMPT = """\
You are verifying a candidate training example before it is used to train a model.

LESSON: {lesson}
PROMPT: {prompt}
RESPONSE: {response}

Is the response (a) a correct, helpful answer to the prompt and (b) consistent \
with the lesson? Answer with exactly one word: YES or NO.
"""

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LM response."""
    match = JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"No JSON object in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


class ReflectionJudge:
    def __init__(self, lm: BaseLM, config: JudgeConfig):
        self.lm = lm
        self.config = config

    def reflect(self, episode: Episode) -> JudgeOutput:
        """Run ``n_samples`` reflections over one episode and aggregate."""
        cfg = self.config
        prompt = REFLECT_PROMPT.format(
            max_examples=cfg.max_examples_per_episode,
            transcript=episode.transcript(max_chars=cfg.max_transcript_chars),
        )
        raw_responses = [self.lm.completion(prompt) for _ in range(cfg.n_samples)]
        parsed = [extract_json(r) for r in raw_responses]

        learn_votes = [p for p in parsed if p.get("verdict") == "learn"]
        if len(learn_votes) * 2 <= len(parsed):
            return JudgeOutput(
                episode_id=episode.episode_id,
                verdict="skip",
                confidence=1.0 - len(learn_votes) / len(parsed),
                lesson="",
                raw_responses=raw_responses,
            )

        best = max(learn_votes, key=lambda p: float(p.get("confidence", 0.0)))
        confidence = float(best.get("confidence", 0.0))
        lesson = str(best.get("lesson", ""))
        if confidence < cfg.min_confidence:
            return JudgeOutput(
                episode_id=episode.episode_id,
                verdict="skip",
                confidence=confidence,
                lesson=lesson,
                raw_responses=raw_responses,
            )

        examples = []
        for ex in list(best.get("examples", []))[: cfg.max_examples_per_episode]:
            example = TrainingExample(
                prompt=str(ex["prompt"]),
                response=str(ex["response"]),
                lesson=lesson,
                source_episode_id=episode.episode_id,
            )
            example.verified = self.verify(example) if cfg.verify_examples else True
            examples.append(example)

        return JudgeOutput(
            episode_id=episode.episode_id,
            verdict="learn",
            confidence=confidence,
            lesson=lesson,
            examples=examples,
            raw_responses=raw_responses,
        )

    def verify(self, example: TrainingExample) -> bool:
        """The verification gate: YES/NO check on one distilled example."""
        response = self.lm.completion(
            VERIFY_PROMPT.format(
                lesson=example.lesson, prompt=example.prompt, response=example.response
            )
        )
        return response.strip().upper().startswith("YES")

    def reflect_all(self, episodes: list[Episode]) -> list[JudgeOutput]:
        return [self.reflect(ep) for ep in episodes]
