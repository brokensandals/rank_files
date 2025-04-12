import random
from rank_files.algos import tournament, tournament_estimated_comparisons, ComparisonTracker
from hypothesis import given, strategies as st


@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=10000), st.integers())
def test_tournament(k, n, seed):
    random.seed(seed)
    # I had trouble getting st.lists() to generate anything but very small lists,
    # so I'm just having it generate a size and then making the list here
    nums = list(range(n))
    random.shuffle(nums)
    tracker = ComparisonTracker()
    result = tracker.unwrap(tournament(k, tracker.wrap(nums)))
    assert result == sorted(nums, reverse=True)[:k]
    assert tracker.total <= tournament_estimated_comparisons(k, len(nums))
