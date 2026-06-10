"""Tests for the episode generator and the pure tokenisation/alignment layer.

A char-level mock tokenizer stands in for SentencePiece: encoding is exactly
compositional, so these tests pin the *alignment contract* (CE positions are
predictor positions; fact masks cover fact segments; probes carried for
controls) independent of any real model or network access.
"""

import pytest

torch = pytest.importorskip("torch")

from rlm.memory.cache import encode_segments, tokenize_episode  # noqa: E402
from rlm.memory.episodes import EpisodeGenerator  # noqa: E402

BOS = 1


class CharTok:
    bos_token_id = BOS

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def test_generator_is_deterministic_per_seed():
    a = EpisodeGenerator(seed=7).sample_mix()
    b = EpisodeGenerator(seed=7).sample_mix()
    assert a.episode_type == b.episode_type
    assert a.query_prompt == b.query_prompt
    assert a.answer == b.answer


def test_episode_types_have_expected_targets():
    gen = EpisodeGenerator(seed=0)
    recall = gen.recall()
    assert recall.answer is not None and recall.answer.startswith(" ")
    assert recall.probe_answer is None
    control = gen.control()
    assert control.answer is None and control.probe_answer is not None
    abstain = gen.abstain()
    assert abstain.answer is None and abstain.probe_answer is None
    multi = gen.multisession()
    assert len(multi.sessions) >= 2
    assert multi.queried_fact is not None


def test_query_prompt_concatenates_with_answer():
    gen = EpisodeGenerator(seed=3)
    f = gen.make_fact()
    assert f.verbatim_prompt + f.answer + "." == f.statement


def test_encode_segments_marks_fact_tokens():
    gen = EpisodeGenerator(seed=1)
    ep = gen.recall()
    ids, fact_mask = encode_segments(CharTok(), ep.sessions[0], BOS)
    assert ids[0] == BOS and fact_mask[0] == 0
    text = "".join(s.text for s in ep.sessions[0])
    assert CharTok().decode(ids[1:]) == text
    # The fact statement's characters are exactly the masked ones.
    fact_text = next(s.text for s in ep.sessions[0] if s.role == "fact")
    assert int(fact_mask.sum()) == len(fact_text)


def test_tokenize_episode_ce_positions_are_predictor_positions():
    gen = EpisodeGenerator(seed=2)
    ep = gen.recall()
    te = tokenize_episode(ep, CharTok(), BOS)
    P = 1 + len(ep.query_prompt)  # bos + prompt chars
    n = len(ep.answer)
    assert te.ce_in_loss
    assert te.ce_pos.tolist() == list(range(P - 1, P - 1 + n))
    assert te.ce_tgt.tolist() == [ord(c) for c in ep.answer]
    assert len(te.query_ids) == P + n
    # KL owns everything except CE positions and the final position.
    assert te.kl_mask[te.ce_pos].sum() == 0
    assert te.kl_mask[-1] == 0
    assert te.kl_mask.sum() == len(te.query_ids) - n - 1


def test_tokenize_control_carries_probe_not_ce():
    gen = EpisodeGenerator(seed=4)
    ep = gen.control()
    te = tokenize_episode(ep, CharTok(), BOS)
    assert not te.ce_in_loss and len(te.ce_pos) == 0
    n = len(ep.probe_answer)
    P = 1 + len(ep.query_prompt)
    assert te.probe_pos.tolist() == list(range(P - 1, P - 1 + n))
    assert te.probe_tgt.tolist() == [ord(c) for c in ep.probe_answer]


def test_tokenize_episode_roundtrip_guard_fires():
    class LossyTok(CharTok):
        def decode(self, ids):
            return "corrupted"

    gen = EpisodeGenerator(seed=5)
    with pytest.raises(ValueError, match="round-trip"):
        tokenize_episode(gen.recall(), LossyTok(), BOS)


def test_multifact_k_range_is_respected():
    gen = EpisodeGenerator(seed=9, multifact_k=(8, 8))
    ep = gen.multifact()
    assert len(ep.facts) == 8


def test_same_relation_hard_negative_controls():
    gen = EpisodeGenerator(seed=11, same_relation_control_prob=1.0)
    for _ in range(10):
        ep = gen.control()
        f_in, f_out = ep.facts[0], ep.queried_fact
        assert f_in.relation_id == f_out.relation_id
        assert f_in.entity != f_out.entity and f_in.answer != f_out.answer


def test_tokenized_episode_carries_relation_label():
    gen = EpisodeGenerator(seed=12)
    ep = gen.recall()
    te = tokenize_episode(ep, CharTok(), BOS)
    assert te.relation_id == ep.queried_fact.relation_id
