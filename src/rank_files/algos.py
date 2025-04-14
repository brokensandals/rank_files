from functools import total_ordering
from typing import Self, Optional
from tqdm import tqdm
import math


class MaxComparisonsExceededError(ValueError):
    """
    Raised when ComparisonTracker has already observed the maximum allowed number of comparisons
    and another comparison is attempted.
    """
    pass


@total_ordering
class ComparisonSpy:
    """
    Used by ComparisonTracker to wrap arbitrary objects and report on the comparison operation
    invoked on them.
    """
    def __init__(self, val, tracker: "ComparisonTracker") -> None:
        self.val = val
        self.tracker = tracker
    
    def __eq__(self, other) -> bool:
        return self.val == other.val
    
    def __lt__(self, other) -> bool:
        if self.tracker.max_comparisons is not None and self.tracker.total >= self.tracker.max_comparisons:
            raise MaxComparisonsExceededError()
        self.tracker.inc()
        return self.val < other.val


class ComparisonTracker:
    """
    This is used for keeping track of the number of comparisons done on a set of objects.
    To add tracking to some objects, use the wrap() method. The total number of comparisons done
    so far across wrapped objects at any point in time is available in the .total instance var.

    Note: Only less-than/greater-than operations are tracked; equality operations are
    not considered important in this project since they do not require LLM invocations.

    If max_comparisons is set, once the total grows to that number, any attempt to perform further
    comparisons will raise a MaxComparisonsExceededError.

    If pbar is provided, it will be updated every time a comparison occurs.
    """
    def __init__(self, max_comparisons: Optional[int] = None, pbar: Optional[tqdm] = None) -> None:
        self.total = 0
        self.max_comparisons = max_comparisons
        self.pbar = pbar

    def wrap(self, items: list) -> list[ComparisonSpy]:
        """
        Wraps all of the given objects with ComparisonSpy objects, which implement comparison
        operators such that the tracker's total is updated before delegating the comparison
        to the original object.
        """
        return [ComparisonSpy(x, self) for x in items]
    
    def unwrap(self, wrapped: list[ComparisonSpy]) -> list:
        """Call this on the result of wrap() to get back the original objects."""
        return [x.val for x in wrapped]
    
    def inc(self) -> None:
        """Used by ComparisonSpy to indicate that a new comparison should be recorded."""
        self.total += 1
        if self.pbar is not None:
            self.pbar.update(1)


class Node:
    """A binary tree node."""
    def __init__(self, val, left: Self = None, right: Self = None) -> None:
        self.val = val
        self.left = left
        self.right = right


def tournament(k: int, items: list) -> list:
    """
    This finds the top-k greatest items in the given list, sorted from greatest to least.

    This requires approximately (n-1)+(k-1)log(n) comparison operations.

    I compared it with a couple heap-based algorithms and a quickselect-based algorithm;
    for small n (e.g. 10000 or less) and small k relative to n (e.g k=10) it seems to
    require the fewest comparison operations. That's the usage pattern I expect for this
    tool; for other projects, other algorithms would be more appropriate.

    TODO: I don't think this is the same as Knuth's tournament algorithm, and I haven't
    compared performance with that.
    """
    k = min(k, len(items))
    if k == 0:
        return []

    def build_tree(array: list) -> Optional[Node]:
        if len(array) == 1:
            return Node(array[0])
        mid = len(array) // 2
        left = build_tree(array[:mid])
        right = build_tree(array[mid:])
        if left.val < right.val:
            return Node(right.val, left, right)
        return Node(left.val, right, left)
    
    def next_best(node: Node) -> Optional[Node]:
        if node is None:
            return None
        right = next_best(node.right)
        if right is None:
            return node.left
        if node.left.val < right.val:
            node.val = right.val
            node.right = right
        else:
            node.val = node.left.val
            node.right = node.left
            node.left = right
        return node

    root = build_tree(items)
    result = [root.val]
    while len(result) < k:
        root = next_best(root)
        result.append(root.val)
    return result


def tournament_estimated_comparisons(k: int, n: int) -> int:
    """
    Returns an estimate of the number of less-than operations the tournament()
    function will require when called with the given k and a list of size n.
    """
    if n <= 1 or k == 0:
        return 0
    k = min(k, n)
    # TODO: is it possible to give a tighter bound than this?
    return n-1 + math.ceil((k-1)*(math.log2(n)))
