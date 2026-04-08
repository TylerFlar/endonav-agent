"""Online topological map for DFS bookkeeping (no metric ground truth)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopoNode:
    id: str
    kind: str  # "junction" | "calyx" | "lumen"
    branches_total: int = 0
    branches_tried: list[int] = field(default_factory=list)
    parent: str | None = None


class TopoGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, TopoNode] = {}
        self.edges: list[tuple[str, str]] = []
        self._counter = 0
        self.current: str | None = None

    def new_id(self, prefix: str) -> str:
        i = self._counter
        self._counter += 1
        return f"{prefix}_{i:03d}"

    def add_node(self, kind: str, parent: str | None = None, branches_total: int = 0) -> TopoNode:
        node = TopoNode(id=self.new_id(kind), kind=kind, parent=parent, branches_total=branches_total)
        self.nodes[node.id] = node
        if parent is not None:
            self.edges.append((parent, node.id))
        self.current = node.id
        return node

    def mark_branch_tried(self, node_id: str, branch_idx: int) -> None:
        n = self.nodes[node_id]
        if branch_idx not in n.branches_tried:
            n.branches_tried.append(branch_idx)

    def untried_branch(self, node_id: str) -> int | None:
        n = self.nodes[node_id]
        for i in range(n.branches_total):
            if i not in n.branches_tried:
                return i
        return None
