import pytest
from rank_files.ranker import FakeRanker, ModelProvider, build_ranker
from rank_files.document import StrDocument


mark_ollama = pytest.mark.skipif("not config.getoption('ollama')")
mark_anthropic = pytest.mark.skipif("not config.getoption('anthropic')")

def test_fake_ranker():
    doc1 = StrDocument("foo")
    doc2 = StrDocument("bar")
    assert FakeRanker().choose_better("just pick one", doc1, doc2) in [doc1, doc2]


@mark_ollama
def test_ollama_ranker():
    criteria = "The best document is the one with the most spelling errors."
    doc1 = StrDocument("In the long run, were all ded.")
    doc2 = StrDocument("In the long run, we're all dead.")
    ranker = build_ranker(ModelProvider.OLLAMA)
    assert ranker.choose_better(criteria, doc1, doc2) is doc1
    assert ranker.choose_better(criteria, doc2, doc1) is doc1


@mark_anthropic
def test_anthropic_ranker():
    criteria = "The best document is the one with the most spelling errors."
    doc1 = StrDocument("In the long run, were all ded.")
    doc2 = StrDocument("In the long run, we're all dead.")
    ranker = build_ranker(ModelProvider.ANTHROPIC)
    assert ranker.choose_better(criteria, doc1, doc2) is doc1
    assert ranker.choose_better(criteria, doc2, doc1) is doc1
