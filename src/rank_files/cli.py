from argparse import ArgumentParser
from pathlib import Path
from rank_files.cache import default_cache
from rank_files.document import FileDocument
from rank_files.ranker import build_ranker
from rank_files.algos import tournament, tournament_estimated_comparisons, ComparisonTracker, MaxComparisonsExceededError
from tqdm import tqdm
import os


MAX_COMPARISONS = int(os.getenv("RANK_FILES_MAX_COMPARISONS", "1000"))
MAX_COMPARISONS_MESSAGE = f"To protect against excessively slow and/or expensive jobs, the limit is {MAX_COMPARISONS}. You can override this limit by setting the RANK_FILES_MAX_COMPARISONS env var."


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("criteria", type=str, help="Ranking criteria, e.g. 'The best document is the one with the most elegant prose.'")
    parser.add_argument("input_dir", type=str, help="Path to directory containing files to rank")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="How many top documents to find")
    parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Only print final rankings, no stats or progress bar")
    args = parser.parse_args()
    docs = [FileDocument(p) for p in Path(args.input_dir).iterdir()]
    estimate = tournament_estimated_comparisons(args.top_k, len(docs))
    if estimate > MAX_COMPARISONS:
        raise MaxComparisonsExceededError(f"This job could require {estimate} pairwise comparisons. {MAX_COMPARISONS_MESSAGE}")
    docs.sort(key=lambda d: d.cheap_sort_key())
    cache = default_cache()
    ranker = build_ranker(cache=cache)
    docs = ranker.wrap_for_pairwise_comparison(args.criteria, docs)
    with tqdm(total=estimate, disable=args.quiet) as pbar:
        tracker = ComparisonTracker(max_comparisons=MAX_COMPARISONS, pbar=pbar)
        docs = tracker.wrap(docs)
        try:
            docs = tournament(args.top_k, docs)
        except MaxComparisonsExceededError as exc:
            raise MaxComparisonsExceededError(f"This job attempted to use more than the allowed number of pairwise comparisons. {MAX_COMPARISONS_MESSAGE}") from exc
        docs = tracker.unwrap(docs)
        docs = ranker.unwrap(docs)
        if not args.quiet:
            print(f"(Total comparisons: {tracker.total}. Cache hits: {cache.total_hits})")
        for doc in docs:
            print(doc)
