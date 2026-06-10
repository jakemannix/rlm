"""
Synthetic episode generator for meta-training the memory skill.

Design doc §1/Q5.  The skill must generalise over *associations*, not over one
task, so episodes are sampled from: ~25 relation templates × nonce entities ×
**nonce answers** (low-prior alphanumerics / pseudo-words, 1–4 tokens), embedded in
neutral filler prose, in five episode types:

- ``recall``       ingest one fact (+ filler), query it.  CE on the answer.
- ``multifact``    ingest k facts, query one.  Teaches non-interference.
- ``multisession`` ingest segments split across 2–3 sessions with filler between;
                   the queried fact is in an *early* session.  This is what trains
                   the forget gate to not forget while later text streams through —
                   the actual cross-session persistence skill.
- ``abstain``      ingest a fact, query unrelated neutral text.  No CE; the
                   KL-to-base term teaches "defer to the base" (A2).
- ``control``      ingest fact A, query fact B's question.  No CE on B's answer
                   (it was never shown!); KL keeps the distribution at base, and
                   B's gold is carried as an eval-only *probe* (A1 control (ii)).

Everything here is plain text + spans; tokenisation happens in
``rlm/memory/cache.py`` against the real tokenizer (or a mock in tests).
Answers carry their leading space; query prompts end exactly where the answer
begins, so ``prompt + answer`` concatenates naturally.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Vocabulary of relations.  statement: "<prefix>{e}<mid>{a}." ; the *verbatim*
# query is the statement truncated right before {a}; paraphrases are separate
# templates ending right before the answer.
# --------------------------------------------------------------------------
RELATIONS: list[dict] = [
    {"id": "access_code", "stmt": "The access code for {e} is{a}.", "para": ["If you need to get into {e}, the code is{a}.", "{e} can be unlocked with the code{a}."]},
    {"id": "codename", "stmt": "The internal codename for {e} is{a}.", "para": ["Inside the company, {e} goes by the codename{a}."]},
    {"id": "serial", "stmt": "The serial number of {e} is{a}.", "para": ["{e} carries the serial number{a}."]},
    {"id": "password", "stmt": "The maintenance password for {e} is{a}.", "para": ["To service {e}, technicians enter the password{a}."]},
    {"id": "frequency", "stmt": "The radio frequency assigned to {e} is{a}.", "para": ["{e} broadcasts on the frequency{a}."]},
    {"id": "registry", "stmt": "The registry key for {e} is{a}.", "para": ["{e} is filed under the registry key{a}."]},
    {"id": "shipment", "stmt": "The tracking number for {e} is{a}.", "para": ["The shipment {e} can be traced with the number{a}."]},
    {"id": "license", "stmt": "The license plate of {e} is{a}.", "para": ["{e} drives a vehicle with the plate{a}."]},
    {"id": "gate", "stmt": "The departure gate for {e} is{a}.", "para": ["{e} leaves from gate{a}."]},
    {"id": "locker", "stmt": "The locker assigned to {e} is{a}.", "para": ["{e} keeps belongings in locker{a}."]},
    {"id": "extension", "stmt": "The phone extension for {e} is{a}.", "para": ["You can reach {e} at extension{a}."]},
    {"id": "badge", "stmt": "The badge number of {e} is{a}.", "para": ["{e} wears badge number{a}."]},
    {"id": "vault", "stmt": "The vault combination at {e} is{a}.", "para": ["The safe in {e} opens with{a}."]},
    {"id": "compound", "stmt": "The active compound in {e} is called{a}.", "para": ["Chemists refer to the key ingredient of {e} as{a}."]},
    {"id": "star", "stmt": "The catalog designation of the star {e} is{a}.", "para": ["Astronomers list {e} as{a}."]},
    {"id": "ship", "stmt": "The hull identifier of the vessel {e} is{a}.", "para": ["The ship {e} is registered as{a}."]},
    {"id": "project", "stmt": "The budget line for project {e} is{a}.", "para": ["Project {e} bills against the budget line{a}."]},
    {"id": "patient", "stmt": "The case file for {e} is indexed as{a}.", "para": ["Clinicians pull {e} under the index{a}."]},
    {"id": "exhibit", "stmt": "The exhibit number for {e} is{a}.", "para": ["In the catalog, {e} appears as exhibit{a}."]},
    {"id": "wifi", "stmt": "The wifi network at {e} is named{a}.", "para": ["Guests of {e} connect to the network{a}."]},
    {"id": "train", "stmt": "The platform for the {e} service is{a}.", "para": ["The {e} train departs from platform{a}."]},
    {"id": "satellite", "stmt": "The transponder id of {e} is{a}.", "para": ["{e} relays through transponder{a}."]},
    {"id": "recipe", "stmt": "The secret spice in {e} is called{a}.", "para": ["What makes {e} distinctive is a spice named{a}."]},
    {"id": "mine", "stmt": "The shaft designation at {e} is{a}.", "para": ["Miners at {e} work shaft{a}."]},
    {"id": "archive", "stmt": "The microfilm reel for {e} is{a}.", "para": ["The records of {e} live on reel{a}."]},
]

FILLER_SENTENCES: list[str] = [
    "The afternoon light settled evenly across the courtyard.",
    "Routine inspections continued without interruption through the week.",
    "A faint hum from the ventilation system was the only sound.",
    "Records from the previous quarter were archived on schedule.",
    "The corridor smelled faintly of fresh paint and dust.",
    "Visitors signed the ledger and waited near the front desk.",
    "Rain had been forecast, but the sky stayed a flat grey.",
    "The committee adjourned earlier than expected on Thursday.",
    "Several crates remained stacked beside the loading dock.",
    "Staff rotated through the morning shift as usual.",
    "The garden beds had been turned over for the season.",
    "A delivery van idled briefly outside the east entrance.",
    "Notices on the board had not changed since last month.",
    "The stairwell lights flickered once and then held steady.",
    "Lunch service ran long because of a visiting delegation.",
    "The river was low for this time of year.",
    "Maps of the old district hung along the west wall.",
    "Two clerks compared figures quietly at the corner table.",
    "The elevator had been serviced the previous Tuesday.",
    "Wind moved through the poplars beyond the fence line.",
    "The printer in the annex was out of toner again.",
    "Morning fog lifted slowly off the parking lot.",
    "An old radio played softly in the maintenance office.",
    "The quarterly newsletter went out a day late.",
    "Folding chairs were stacked neatly against the back wall.",
    "The custodian propped the side door open to air the hall.",
    "Telephone lines were tested at the start of each month.",
    "A bicycle leaned unattended against the railing.",
    "The cafeteria switched to its winter menu this week.",
    "Boxes of outdated forms awaited the shredder.",
]

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"


@dataclass
class Fact:
    relation_id: str
    entity: str
    statement: str  # full sentence including the answer
    verbatim_prompt: str  # statement truncated right before the answer
    paraphrase_prompts: list[str]
    answer: str  # includes its leading space

    def query_prompt(self, rng: random.Random, paraphrase_prob: float) -> str:
        if self.paraphrase_prompts and rng.random() < paraphrase_prob:
            return rng.choice(self.paraphrase_prompts)
        return self.verbatim_prompt


@dataclass
class Segment:
    text: str
    role: str  # "filler" | "fact"


@dataclass
class EpisodeText:
    episode_type: str  # recall | multifact | multisession | abstain | control
    sessions: list[list[Segment]]  # outer = session boundaries
    query_prompt: str
    answer: str | None  # None ⇒ no CE term (abstain)
    probe_answer: str | None  # eval-only gold (control episodes)
    facts: list[Fact] = field(default_factory=list)
    queried_fact: Fact | None = None


class EpisodeGenerator:
    def __init__(self, seed: int = 0, paraphrase_prob: float = 0.0):
        self.rng = random.Random(seed)
        self.paraphrase_prob = paraphrase_prob

    # -- nonce builders -------------------------------------------------
    def nonce_word(self, syllables: int = 2, capitalize: bool = True) -> str:
        rng = self.rng
        w = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables))
        if rng.random() < 0.5:
            w += rng.choice(CONSONANTS)
        return w.capitalize() if capitalize else w

    def nonce_entity(self) -> str:
        rng = self.rng
        style = rng.randrange(3)
        if style == 0:
            return f"the {self.nonce_word(2)} {rng.choice(['facility', 'annex', 'depot', 'station', 'archive', 'lab'])}"
        if style == 1:
            return f"{self.nonce_word(2)} {self.nonce_word(2)}"  # a person-ish name
        return f"{self.nonce_word(2)}-{rng.randrange(2, 99)}"

    def nonce_answer(self) -> str:
        rng = self.rng
        style = rng.randrange(4)
        if style == 0:  # "QX-4471"
            return " " + "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + "-" + "".join(rng.choice(string.digits) for _ in range(4))
        if style == 1:  # "ZR7-90213"
            return " " + "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + rng.choice(string.digits) + "-" + "".join(rng.choice(string.digits) for _ in range(5))
        if style == 2:  # pseudo-word: "flumetrine"
            return " " + self.nonce_word(3, capitalize=False)
        return " " + self.nonce_word(2) + "-" + self.nonce_word(2, capitalize=False)  # "Velka-dorin"

    def make_fact(self) -> Fact:
        rel = self.rng.choice(RELATIONS)
        entity = self.nonce_entity()
        answer = self.nonce_answer()
        stmt_full = rel["stmt"].format(e=entity, a=answer)
        verbatim = rel["stmt"].format(e=entity, a="\u0000").split("\u0000")[0]
        paras = [p.format(e=entity, a="\u0000").split("\u0000")[0] for p in rel["para"]]
        return Fact(rel["id"], entity, stmt_full, verbatim, paras, answer)

    # -- assembly helpers ------------------------------------------------
    def filler(self, n_sentences: int) -> Segment:
        s = " ".join(self.rng.choice(FILLER_SENTENCES) for _ in range(n_sentences))
        return Segment(" " + s, "filler")

    def session_with_facts(self, facts: list[Fact], filler_lo: int = 1, filler_hi: int = 3) -> list[Segment]:
        rng = self.rng
        segs = [self.filler(rng.randint(filler_lo, filler_hi))]
        for f in facts:
            segs.append(Segment(" " + f.statement, "fact"))
            segs.append(self.filler(rng.randint(filler_lo, filler_hi)))
        return segs

    # -- episode types ----------------------------------------------------
    def recall(self) -> EpisodeText:
        f = self.make_fact()
        return EpisodeText(
            "recall", [self.session_with_facts([f])],
            f.query_prompt(self.rng, self.paraphrase_prob), f.answer, None, [f], f,
        )

    def multifact(self, k: int | None = None) -> EpisodeText:
        k = k or self.rng.randint(2, 6)
        facts = [self.make_fact() for _ in range(k)]
        target = self.rng.choice(facts)
        return EpisodeText(
            "multifact", [self.session_with_facts(facts)],
            target.query_prompt(self.rng, self.paraphrase_prob), target.answer, None, facts, target,
        )

    def multisession(self, n_sessions: int | None = None) -> EpisodeText:
        n_sessions = n_sessions or self.rng.randint(2, 3)
        target = self.make_fact()
        facts = [target]
        sessions = [self.session_with_facts([target])]
        for _ in range(n_sessions - 1):
            later = [self.make_fact() for _ in range(self.rng.randint(0, 2))]
            facts.extend(later)
            sessions.append(self.session_with_facts(later, filler_lo=2, filler_hi=4))
        return EpisodeText(
            "multisession", sessions,
            target.query_prompt(self.rng, self.paraphrase_prob), target.answer, None, facts, target,
        )

    def abstain(self) -> EpisodeText:
        f = self.make_fact()
        neutral = self.filler(2).text.lstrip()
        # Query = a neutral sentence the base should continue as itself; split it
        # so there is real next-token structure under the KL mask.
        words = neutral.split()
        cut = max(3, len(words) // 2)
        prompt = " ".join(words[:cut])
        return EpisodeText("abstain", [self.session_with_facts([f])], prompt, None, None, [f], None)

    def control(self) -> EpisodeText:
        f_in, f_out = self.make_fact(), self.make_fact()
        return EpisodeText(
            "control", [self.session_with_facts([f_in])],
            f_out.query_prompt(self.rng, self.paraphrase_prob), None, f_out.answer, [f_in], f_out,
        )

    DEFAULT_MIX: dict[str, float] = None  # set below

    def sample(self, episode_type: str) -> EpisodeText:
        return {
            "recall": self.recall,
            "multifact": self.multifact,
            "multisession": self.multisession,
            "abstain": self.abstain,
            "control": self.control,
        }[episode_type]()

    def sample_mix(self, mix: dict[str, float] | None = None) -> EpisodeText:
        mix = mix or {"recall": 0.45, "multifact": 0.2, "multisession": 0.15, "abstain": 0.1, "control": 0.1}
        types, weights = zip(*mix.items(), strict=True)
        return self.sample(self.rng.choices(types, weights=weights, k=1)[0])
