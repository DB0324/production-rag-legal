"""
Fast unit tests - no GPU, no Qdrant, no LLM calls.
    pytest tests/test_units.py -v
"""
import pytest

from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, RRF_K
from src.generation.generate import extract_citations
from src.generation.prompt_templates import build_prompt, SYSTEM_PROMPT


def _chunk(cid, doc="d1", title="Case A", text="some text"):
    return {"chunk_id": cid, "doc_id": doc, "case_title": title, "text": text}


def test_rrf_rewards_agreement():
    """A doc ranked top by both retrievers must win the fusion."""
    a = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    b = [_chunk("c1"), _chunk("c3"), _chunk("c2")]
    fused = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "c1"


def test_rrf_scores_match_formula():
    a = [_chunk("c1"), _chunk("c2")]
    fused = reciprocal_rank_fusion([a], top_k=2)
    assert fused[0]["rrf_score"] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[1]["rrf_score"] == pytest.approx(1.0 / (RRF_K + 2))


def test_rrf_deduplicates_same_chunk():
    fused = reciprocal_rank_fusion([[_chunk("c1"), _chunk("c1")]], top_k=10)
    assert len(fused) == 1


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([[]], top_k=5) == []


def test_extract_citations():
    assert extract_citations("Held X [Case A, 2018_1_EN].") == ["Case A, 2018_1_EN"]
    assert extract_citations("no citations here") == []


def test_build_prompt_carries_chunk_metadata():
    p = build_prompt("Q?", [_chunk("c1", title="Kesavananda")])
    assert "Kesavananda" in p
    assert "c1" in p
    assert "Q?" in p
    assert SYSTEM_PROMPT[:40] in p
