from argparse import ArgumentParser
from pathlib import Path
from rank_files.document import FileDocument
from rank_files.ranker import build_ranker
from rank_files.algos import tournament, tournament_estimated_comparisons, ComparisonTracker
from tqdm import tqdm
import os


MAX_FILES = int(os.getenv("RANK_FILES_MAX_FILES", "500"))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("criteria", type=str, help="Ranking criteria, e.g. 'The best document is the one with the most elegant prose.'")
    parser.add_argument("input_dir", type=str, help="Path to directory containing files to rank")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="How many top documents to find")
    parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Only print final rankings, no stats or progress bar")
    args = parser.parse_args()
    docs = [FileDocument(p) for p in Path(args.input_dir).iterdir()]
    if len(docs) > MAX_FILES:
        raise ValueError(f"You tried to rank {len(docs)} documents. To protect against excessively slow and/or expensive jobs, the limit is {MAX_FILES}. You can override this limit by setting the RANK_FILES_MAX_FILES env var.")
    ranker = build_ranker()
    docs = ranker.wrap_for_pairwise_comparison(args.criteria, docs)
    with tqdm(total=tournament_estimated_comparisons(args.top_k, len(docs)), disable=args.quiet) as pbar:
        tracker = ComparisonTracker(pbar)
        docs = tracker.wrap(docs)
        docs = tournament(args.top_k, docs)
        docs = tracker.unwrap(docs)
        if not args.quiet:
            print(f"(Total comparisons: {tracker.total})")
        for doc in docs:
            print(doc)
