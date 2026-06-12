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

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

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
You are reviewing one candidate training example before it is used to \
fine-tune a model. Your default is suspicion: a bad example that slips \
through damages the model, while a rejected good example costs little.

LESSON: {lesson}
PROMPT: {prompt}
RESPONSE: {response}

Check, in order:
1. Is the response factually and technically correct?
2. Would following it cause harm (destructive commands, data loss, security risks)?
3. Does it actually answer the prompt?
4. Is it consistent with the lesson?

State any problems you find in one or two short sentences, then give your \
verdict on the last line in exactly this form:
VERDICT: YES
or
VERDICT: NO
"""

VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def parse_verify_verdict(text: str) -> bool:
    """Last ``VERDICT: YES/NO`` wins; anything unparseable fails closed."""
    matches = VERDICT_RE.findall(text or "")
    return bool(matches) and matches[-1].upper() == "YES"


JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LM response."""
    match = JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"No JSON object in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


def output_from_dict(record: dict) -> JudgeOutput:
    """Rebuild a JudgeOutput from its asdict() form (cache & artifact files)."""
    examples = [TrainingExample(**ex) for ex in record.get("examples", [])]
    return JudgeOutput(**(record | {"examples": examples}))


class ReflectionJudge:
    """Reflect over gated episodes; optionally cache outputs on disk.

    The cache is keyed by everything that determines a reflection — the
    full prompt (which embeds the transcript), the judge's identity and
    sampling settings, and the verification flags — so sweep points that
    share gate/judge settings pay for the judge exactly once.
    """

    def __init__(self, lm: BaseLM, config: JudgeConfig, cache_path: str | Path | None = None):
        self.lm = lm
        self.config = config
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache_hits = 0
        self._cache: dict[str, dict] = {}
        if self.cache_path is not None and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _cache_key(self, prompt: str) -> str:
        cfg = self.config
        ident = "\x1e".join(
            [
                prompt,
                self.lm.model_name,
                str(getattr(self.lm, "temperature", None)),
                str(cfg.n_samples),
                str(cfg.min_confidence),
                str(cfg.verify_examples),
                # Cached outputs embed verified flags, so a change to either
                # prompt template must invalidate them.
                REFLECT_PROMPT + VERIFY_PROMPT,
            ]
        )
        return hashlib.sha256(ident.encode("utf-8")).hexdigest()

    def reflect(self, episode: Episode) -> JudgeOutput:
        """Run ``n_samples`` reflections over one episode and aggregate."""
        cfg = self.config
        prompt = REFLECT_PROMPT.format(
            max_examples=cfg.max_examples_per_episode,
            transcript=episode.transcript(max_chars=cfg.max_transcript_chars),
        )
        key = self._cache_key(prompt)
        if key in self._cache:
            self.cache_hits += 1
            return output_from_dict(self._cache[key])
        output = self._reflect_uncached(episode, prompt)
        if self.cache_path is not None:
            self._cache[key] = asdict(output)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return output

    def _reflect_uncached(self, episode: Episode, prompt: str) -> JudgeOutput:
        cfg = self.config
        raw_responses = [self.lm.completion(prompt) for _ in range(cfg.n_samples)]
        # A sample that fails to parse counts as a (silent) skip vote: a judge
        # that can't even produce valid JSON shouldn't get a model update.
        parsed = []
        n_parse_failures = 0
        for r in raw_responses:
            try:
                parsed.append(extract_json(r))
            except (ValueError, json.JSONDecodeError):
                n_parse_failures += 1

        learn_votes = [p for p in parsed if p.get("verdict") == "learn"]
        if len(learn_votes) * 2 <= len(raw_responses):
            return JudgeOutput(
                episode_id=episode.episode_id,
                verdict="skip",
                confidence=1.0 - len(learn_votes) / len(raw_responses),
                lesson="",
                raw_responses=raw_responses,
                n_parse_failures=n_parse_failures,
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
                n_parse_failures=n_parse_failures,
            )

        examples = []
        for ex in list(best.get("examples", []))[: cfg.max_examples_per_episode]:
            if not isinstance(ex, dict) or "prompt" not in ex or "response" not in ex:
                continue  # malformed example from the judge; verdict still stands
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
            n_parse_failures=n_parse_failures,
        )

    def verify(self, example: TrainingExample) -> bool:
        """The verification gate: an adversarial YES/NO check on one example."""
        prompt = VERIFY_PROMPT.format(
            lesson=example.lesson, prompt=example.prompt, response=example.response
        )
        # Verification is discrimination, not generation: greedy-decode it on
        # judges that expose a sampling temperature (sampling only adds noise).
        sampled_temperature = getattr(self.lm, "temperature", None)
        if sampled_temperature is not None:
            self.lm.temperature = 0.0
        try:
            response = self.lm.completion(prompt)
        finally:
            if sampled_temperature is not None:
                self.lm.temperature = sampled_temperature
        return parse_verify_verdict(response)

    def reflect_all(self, episodes: list[Episode]) -> list[JudgeOutput]:
        return [self.reflect(ep) for ep in episodes]
