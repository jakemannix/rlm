"""Real-corpus episode adapter (milestone-3 item #1): held-out-by-entity split and
content-addressable (question) queries, reusing the synthetic episode machinery."""

import pytest

torch = pytest.importorskip("torch")

from rlm.memory.real_corpus import BUNDLED_FACTS, RealCorpusGenerator, _to_records  # noqa: E402


def test_train_eval_splits_are_entity_disjoint():
    tr = {r["entity"] for r in RealCorpusGenerator(split="train", seed=0).records}
    ev = {r["entity"] for r in RealCorpusGenerator(split="eval", seed=0).records}
    assert tr and ev, "both splits must be non-empty"
    assert not (tr & ev), f"train/eval entity overlap: {tr & ev}"
    assert tr | ev == {e for e, *_ in BUNDLED_FACTS}  # partition covers everything


def test_recall_query_is_the_question_under_paraphrase():
    """A1b by construction: the *query* is a natural question, the *ingested*
    statement is the declarative form, and the statement contains the answer."""
    gen = RealCorpusGenerator(split="train", seed=1, paraphrase_prob=1.0)
    for _ in range(20):
        ep = gen.recall()
        f = ep.queried_fact
        assert ep.query_prompt == f.paraphrase_prompts[0] and ep.query_prompt.endswith("?")
        assert ep.answer == f.answer and f.answer.startswith(" ")
        assert f.answer.strip() in f.statement


def test_verbatim_query_when_paraphrase_off():
    gen = RealCorpusGenerator(split="train", seed=1, paraphrase_prob=0.0)
    ep = gen.recall()
    assert ep.query_prompt == ep.queried_fact.verbatim_prompt
    assert ep.query_prompt + ep.answer + "." == ep.queried_fact.statement


def test_multifact_and_control_draw_real_facts():
    gen = RealCorpusGenerator(split="train", seed=2, multifact_k=(3, 3))
    mf = gen.multifact()
    assert len(mf.facts) == 3
    entities = {e for e, *_ in BUNDLED_FACTS}
    assert all(f.entity in entities for f in mf.facts)
    ctl = gen.control()  # same_relation_control_prob=0 → random real pair, no RELATIONS lookup
    assert ctl.episode_type == "control" and ctl.probe_answer is not None


def test_to_records_normalizes_leading_space_and_statement():
    rec = _to_records([("X", "The X is", "What is X?", "Y")])[0]
    assert rec["answer"] == " Y" and rec["statement"] == "The X is Y."
