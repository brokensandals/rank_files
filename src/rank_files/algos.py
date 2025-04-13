from functools import total_ordering
from typing import Self, Optional
from tqdm import tqdm
import math


@total_ordering
class ComparisonSpy:
    def __init__(self, val, tracker: "ComparisonTracker") -> None:
        self.val = val
        self.tracker = tracker
    
    def __eq__(self, other) -> bool:
        return False
    
    def __lt__(self, other) -> bool:
        self.tracker.inc()
        return self.val < other.val


class ComparisonTracker:
    def __init__(self, pbar: Optional[tqdm] = None) -> None:
        self.total = 0
        self.pbar = pbar
    
    def wrap(self, items: list) -> list[ComparisonSpy]:
        return [ComparisonSpy(x, self) for x in items]
    
    def unwrap(self, wrapped: list[ComparisonSpy]) -> list:
        return [x.val for x in wrapped]
    
    def inc(self) -> None:
        self.total += 1
        if self.pbar is not None:
            self.pbar.update(1)


class Node:
    def __init__(self, val, left: Self = None, right: Self = None) -> None:
        self.val = val
        self.left = left
        self.right = right


def tournament(k: int, items: list) -> list:
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
    if n <= 1 or k == 0:
        return 0
    k = min(k, n)
    return n-1 + math.ceil((k-1)*(math.log2(n)))
