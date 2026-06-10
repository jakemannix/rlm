"""Real-text episode source for milestone-3 (`docs/design_review_capacity.md` is
milestone-2/capacity; this is "useful on real text").

The synthetic :class:`EpisodeGenerator` draws nonce entities + nonce answers from
~25 fixed templates.  The real question is whether a skill meta-trained on *real*
text recalls *held-out real* facts across a session boundary — i.e. whether the
mechanism survives the synthetic→real distribution shift, or whether Gemma's
~10-dimensional answer-prefix geometry caps it even after whitening.

This adapter is deliberately minimal: it **subclasses EpisodeGenerator and
overrides only ``make_fact``**, drawing a real (entity, statement, question,
answer) record instead of a nonce one.  All episode machinery (recall / multifact
/ multisession / abstain / control / sample_mix / session_with_facts / filler) is
inherited unchanged, so the cache builder and trainer need no changes — pass a
``RealCorpusGenerator`` via ``build_cache(..., generator=...)``.

Two properties this buys for free:
- **Content-addressable queries (A1b):** each fact's *paraphrase* prompt is a
  natural **question** ("When was the Eiffel Tower completed?"), not a verbatim
  prefix, so any ``paraphrase_prob > 0`` query tests meaning-binding, not surface
  matching.  The ingested *statement* is the declarative form.
- **Strict held-out split by entity:** ``split="train"`` and ``split="eval"`` draw
  from disjoint entity sets (deterministic, independent of the episode seed), so
  eval facts are about entities never seen in meta-training.

``BUNDLED_FACTS`` is a small hand-curated real set so the adapter is unit-testable
on CPU with no download; :func:`load_squad_records` scales it to a real QA corpus.
"""

from __future__ import annotations

import random

from rlm.memory.episodes import EpisodeGenerator, Fact

# (entity, verbatim_prompt, question, answer) — answer's leading space is added
# if missing; the ingested statement is verbatim_prompt + answer + ".".  Answers
# are kept short (1–2 tokens) so first-token recall is meaningful; the generative
# arm (exact-match / F1) handles longer answers.
BUNDLED_FACTS: list[tuple[str, str, str, str]] = [
    ("Eiffel Tower", "The Eiffel Tower was completed in", "When was the Eiffel Tower completed?", "1889"),
    ("Mount Everest", "The height of Mount Everest is about", "How tall is Mount Everest in metres?", "8849"),
    ("Pacific Ocean", "The largest ocean on Earth is the", "Which is the largest ocean on Earth?", "Pacific"),
    ("water", "The chemical formula for water is", "What is the chemical formula for water?", "H2O"),
    ("speed of light", "The speed of light is roughly", "How fast does light travel in km per second?", "300000"),
    ("Mona Lisa", "The Mona Lisa was painted by", "Who painted the Mona Lisa?", "Leonardo"),
    ("Australia", "The capital city of Australia is", "What is the capital of Australia?", "Canberra"),
    ("DNA", "The double-helix structure of DNA was described by Watson and", "Who, with Watson, described DNA's structure?", "Crick"),
    ("Mars", "The largest volcano in the solar system, found on Mars, is Olympus", "What is the tallest volcano on Mars called?", "Mons"),
    ("photosynthesis", "Plants convert sunlight into energy through a process called", "How do plants make energy from sunlight?", "photosynthesis"),
    ("Great Wall", "The Great Wall is located in the country of", "In which country is the Great Wall?", "China"),
    ("Beethoven", "The Ninth Symphony was composed by", "Who composed the Ninth Symphony?", "Beethoven"),
    ("gold", "The chemical symbol for gold is", "What is the chemical symbol for gold?", "Au"),
    ("Amazon River", "The longest river in South America is the", "What is the longest river in South America?", "Amazon"),
    ("penicillin", "Penicillin was discovered by Alexander", "Who discovered penicillin?", "Fleming"),
    ("Saturn", "The planet famous for its prominent rings is", "Which planet is famous for its rings?", "Saturn"),
    ("Tokyo", "The capital of Japan is", "What is the capital of Japan?", "Tokyo"),
    ("oxygen", "The most abundant gas in Earth's atmosphere is", "What is the most abundant gas in the atmosphere?", "nitrogen"),
    ("Shakespeare", "The play Hamlet was written by", "Who wrote Hamlet?", "Shakespeare"),
    ("Sahara", "The largest hot desert in the world is the", "What is the largest hot desert?", "Sahara"),
    ("electron", "The negatively charged particle in an atom is the", "Which atomic particle carries negative charge?", "electron"),
    ("Nile", "The river flowing through Egypt to the Mediterranean is the", "Which river flows through Egypt?", "Nile"),
    ("Einstein", "The theory of general relativity was developed by", "Who developed general relativity?", "Einstein"),
    ("Jupiter", "The largest planet in the solar system is", "What is the largest planet?", "Jupiter"),
    ("Berlin", "The capital of Germany is", "What is the capital of Germany?", "Berlin"),
    ("heart", "The organ that pumps blood through the body is the", "Which organ pumps blood?", "heart"),
    ("Pi", "The mathematical constant relating a circle's circumference to its diameter is", "What constant relates circumference to diameter?", "pi"),
    ("Antarctica", "The coldest continent on Earth is", "Which is the coldest continent?", "Antarctica"),
    ("Newton", "The laws of motion were formulated by Isaac", "Who formulated the laws of motion?", "Newton"),
    ("carbon", "Diamond is made of the element", "What element is diamond made of?", "carbon"),
    ("Rome", "The capital of Italy is", "What is the capital of Italy?", "Rome"),
    ("Moon", "Earth's only natural satellite is the", "What is Earth's natural satellite?", "Moon"),
    ("Darwin", "The theory of evolution by natural selection is credited to Charles", "Who proposed natural selection?", "Darwin"),
    ("Brazil", "The largest country in South America by area is", "What is the largest country in South America?", "Brazil"),
    ("liver", "The organ that filters toxins from the blood is the", "Which organ filters toxins from blood?", "liver"),
    ("Mercury", "The closest planet to the Sun is", "Which planet is closest to the Sun?", "Mercury"),
    ("Cairo", "The capital of Egypt is", "What is the capital of Egypt?", "Cairo"),
    ("hydrogen", "The lightest chemical element is", "What is the lightest element?", "hydrogen"),
    ("Pyramids", "The Great Pyramids stand near the city of Giza in", "In which country are the Great Pyramids?", "Egypt"),
    ("Pasteur", "The process of pasteurization is named after Louis", "Who is pasteurization named after?", "Pasteur"),
]


def _to_records(rows: list[tuple[str, str, str, str]]) -> list[dict]:
    out = []
    for entity, prompt, question, ans in rows:
        a = ans if ans.startswith(" ") else " " + ans
        out.append({"entity": entity, "verbatim_prompt": prompt, "question": question,
                    "answer": a, "statement": prompt + a + "."})
    return out


class RealCorpusGenerator(EpisodeGenerator):
    """Drop-in :class:`EpisodeGenerator` whose facts come from a real corpus.

    Override surface is one method (``make_fact``); episode types, ``sample_mix``
    and assembly are all inherited.  ``same_relation_control_prob`` is unsupported
    here (real facts have no shared relation-template registry) and left at 0.
    """

    def __init__(
        self,
        records: list[dict] | None = None,
        split: str = "train",
        seed: int = 0,
        paraphrase_prob: float = 0.5,
        eval_frac: float = 0.25,
        multifact_k: tuple[int, int] = (2, 6),
    ):
        super().__init__(seed=seed, paraphrase_prob=paraphrase_prob, multifact_k=multifact_k)
        records = records if records is not None else _to_records(BUNDLED_FACTS)
        entities = sorted({r["entity"] for r in records})
        # Deterministic held-out split, independent of the episode RNG, so train and
        # eval entities never overlap regardless of seed.
        split_rng = random.Random(0x5151)
        split_rng.shuffle(entities)
        n_eval = max(1, int(round(len(entities) * eval_frac)))
        eval_entities = set(entities[:n_eval])
        keep = eval_entities if split == "eval" else (set(entities) - eval_entities)
        self.records = [r for r in records if r["entity"] in keep]
        if not self.records:
            raise ValueError(f"no records for split={split!r} ({len(records)} records, eval_frac={eval_frac})")
        self.split = split

    def make_fact(self, relation: dict | None = None) -> Fact:
        r = self.rng.choice(self.records)
        return Fact(
            relation_id=r["entity"],  # entity doubles as the dedup/label key
            entity=r["entity"],
            statement=r["statement"],
            verbatim_prompt=r["verbatim_prompt"],
            paraphrase_prompts=[r["question"]],
            answer=r["answer"],
        )


def load_squad_records(n: int = 5000, hf_split: str = "train") -> list[dict]:
    """Build records from SQuAD via HF ``datasets`` (optional, for scale).

    statement = the context sentence containing the answer; question = the SQuAD
    question (the content-addressable query); entity = the article title (the
    held-out split key).  Filtered to short answers so first-token recall is clean.
    """
    from datasets import load_dataset

    ds = load_dataset("squad", split=hf_split)
    out: list[dict] = []
    for ex in ds:
        ans_list = ex["answers"]["text"]
        if not ans_list:
            continue
        ans = ans_list[0].strip()
        if not ans or len(ans.split()) > 2:
            continue
        ctx, start = ex["context"], ex["answers"]["answer_start"][0]
        lo = ctx.rfind(".", 0, start) + 1
        hi = ctx.find(".", start)
        sentence = ctx[lo : hi if hi != -1 else len(ctx)].strip()
        if ans not in sentence:
            continue
        a = " " + ans
        out.append({
            "entity": ex["title"],
            "verbatim_prompt": sentence[: sentence.index(ans)].rstrip(),
            "question": ex["question"].strip(),
            "answer": a,
            "statement": sentence if sentence.endswith(".") else sentence + ".",
        })
        if len(out) >= n:
            break
    return out
