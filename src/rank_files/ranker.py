from anthropic import Anthropic
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import total_ordering
from hashlib import sha256
from pathlib import Path
from rank_files.cache import Cache, default_cache
from rank_files.document import Document
from typing import Optional, Self
import json
import os
import ollama



PAIRWISE_SYSTEM_PROMPT = Path(__file__).parent.joinpath("prompts", "pairwise-system.txt").read_text("utf8")


class InvalidLlmResponseException(Exception):
    pass


class ModelProvider(StrEnum):
    FAKE = "fake"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


def default_provider() -> ModelProvider:
    return ModelProvider(os.getenv("RANK_FILES_PROVIDER", "ollama"))


def default_model(provider: ModelProvider) -> str:
    model = os.getenv("RANK_FILES_MODEL")
    if model is not None:
        return model
    if provider == ModelProvider.FAKE:
        return "random"
    if provider == ModelProvider.OLLAMA:
        return "gemma3:4b"
    if provider == ModelProvider.ANTHROPIC:
        return "claude-3-5-haiku-latest"
    raise ValueError(f"Unsupported provider {provider}") # should be unreachable


# TODO How to protect against prompt injection could use more thought.
#      I'm currently just escaping angle brackets to make sure nothing breaks the "criteria"/"document-1"/"document-2"
#      tag structure, but I haven't tested
#      (a) how much this really protects against or
#      (b) how much model performance is harmed by superfluous escaping.
def escape_prompt_part(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def pairwise_user_prompt(criteria: str, doc1: Document, doc2: Document) -> str:
    c = escape_prompt_part(criteria)
    t1 = escape_prompt_part(doc1.read_text())
    t2 = escape_prompt_part(doc2.read_text())
    return f"<criteria>{c}</criteria>\n<document-1>{t1}</document-1>\n<document-2>{t2}</document-2>"


def extract_pairwise_response(doc1: Document, doc2: Document, resp_content: str) -> Document:
    if resp_content == "1":
        return doc1
    if resp_content == "2":
        return doc2
    raise InvalidLlmResponseException(f"Model was instructed to respond '1' for {doc1} or '2' for {doc2} but got: {resp_content}")


@total_ordering
class PairwiseWrapper:
    def __init__(self, wrapped: Document, ranker: "Ranker", criteria: str) -> None:
        self.wrapped = wrapped
        self.ranker = ranker
        self.criteria = criteria
    
    def __eq__(self, other: Self) -> bool:
        return self.wrapped == other.wrapped

    def __lt__(self, other: Self) -> bool:
        choice = self.ranker.choose_better(self.criteria, self.wrapped, other.wrapped)
        if choice is self.wrapped:
            return False
        return True


class Ranker(ABC):
    def choose_better(self, criteria: str, doc1: Document, doc2: Document) -> Document:
        if doc1.cheap_sort_key() > doc2.cheap_sort_key():
            # This ensures we'll use the same cache entry regardless of the order of arguments.
            return self.choose_better(criteria, doc2, doc1)
        return self._choose_better(criteria, doc1, doc2)

    @abstractmethod
    def _choose_better(self, criteria: str, doc1: Document, doc2: Document) -> Document:
        ...
    
    def wrap_for_pairwise_comparison(self, criteria: str, docs: list[Document]) -> list[PairwiseWrapper]:
        return [PairwiseWrapper(doc, self, criteria) for doc in docs]
    
    def unwrap(self, items: list[PairwiseWrapper]) -> list[Document]:
        return [item.wrapped for item in items]


class FakeRanker(Ranker):
    def _choose_better(self, criteria: str, doc1: Document, doc2: Document) -> Document:
        if doc1.read_text() < doc2.read_text():
            return doc1
        return doc2


class OllamaRanker(Ranker):
    def __init__(self, model: str, cache: Optional[Cache] = None, client: Optional[ollama.Client] = None) -> None:
        self.cache = cache if cache is not None else Cache(":memory:")
        self.model = model
        self.client = ollama.Client() if client is None else client

    def _choose_better(self, criteria: str, doc1: Document, doc2: Document) -> Document:
        user_prompt = pairwise_user_prompt(criteria, doc1, doc2)
        messages = [
            {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        cache_key = sha256(json.dumps({"provider": "ollama", "model": self.model, "messages": messages}).encode()).hexdigest()
        content = self.cache.fetch(cache_key)
        if content is None:
            # Initial testing suggested that whatever Ollama does for prompts that exceed the default context length
            # (I think it trims the beginning?) leads to poor results. So I try to make the context length long enough.
            # TODO Currently I'm using a rough heuristic to make sure the context length is long enough
            #      to hold the full prompt.
            #      It would be nice to calculate this more exactly; see https://github.com/ollama/ollama/issues/3582
            #      Ideally I'd also avoid setting it higher than the machine's memory can handle, and do something else
            #      (error out? partially replace the documents with summaries? some other technique?) in that case.
            options = {
                "num_predict": 1,
                "num_ctx": len(str(messages)) // 2,
                "temperature": 0,
            }
            resp = self.client.chat(model=self.model, messages=messages, options=options)
            content = resp.message.content
            self.cache.put(cache_key, content)
        return extract_pairwise_response(doc1, doc2, content)


class AnthropicRanker(Ranker):
    def __init__(self, model: str, cache: Cache, client: Optional[Anthropic] = None) -> None:
        self.cache = cache if cache is not None else Cache(":memory:")
        self.model = model
        self.client = Anthropic() if client is None else client
    
    def _choose_better(self, criteria: str, doc1: Document, doc2: Document) -> Document:
        user_prompt = pairwise_user_prompt(criteria, doc1, doc2)
        cache_key = sha256(json.dumps({"provider": "anthropic", "model": self.model, "system_prompt": PAIRWISE_SYSTEM_PROMPT, "user_prompt": user_prompt}).encode()).hexdigest()
        content = self.cache.fetch(cache_key)
        if content is None:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=10, # I tried 1, but for some reason that results in an empty content array in the response,
                system=PAIRWISE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            content = resp.content[0].text
            self.cache.put(cache_key, content)
        return extract_pairwise_response(doc1, doc2, content)


def build_ranker(provider: Optional[ModelProvider] = None, model: Optional[str] = None, cache: Optional[Cache] = None) -> Ranker:
    provider = default_provider() if provider is None else provider
    model = default_model(provider) if model is None else model
    if provider == ModelProvider.FAKE:
        return FakeRanker()
    cache = default_cache() if cache is None else cache
    if provider == ModelProvider.OLLAMA:
        return OllamaRanker(model, cache)
    if provider == ModelProvider.ANTHROPIC:
        return AnthropicRanker(model, cache)
    raise ValueError(f"Unsupported provider {provider}") # should be unreachable
