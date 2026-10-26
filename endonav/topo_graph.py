"""Online topological map of the kidney collecting system.

Built incrementally as the agent explores. The DFS stack is the source of
truth for "current junction": mark_branch_explored() always operates on the
top of the stack, and a junction auto-pops once all its branches are done.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    type: str  # "ureter" | "junction" | "calyx" | "pelvis"
    branches_total: int = 0
    branches_explored: int = 0
    status: str = "unknown"  # "visited" | "stones_found" | "treating" | "verified_clear"
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TopoGraph:
    """Topological map + DFS stack."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.dfs_stack: list[str] = []
        self.current_node_id: str | None = None
        self._next_junction: int = 0
        self._next_calyx: int = 0
        self._next_other: int = 0

    # ------------------------------------------------------------------
    def _new_id(self, type_: str) -> str:
        if type_ == "junction":
            i = self._next_junction
            self._next_junction += 1
            return f"junction_{i}"
        if type_ == "calyx":
            i = self._next_calyx
            self._next_calyx += 1
            return f"calyx_{i}"
        i = self._next_other
        self._next_other += 1
        return f"{type_}_{i}"

    # ------------------------------------------------------------------
    def seed_root(self, type_: str = "ureter") -> str:
        """Create the root node and set it as current."""
        node_id = self._new_id(type_)
        self.nodes[node_id] = Node(
            id=node_id, type=type_, status="visited", parent_id=None
        )
        self.current_node_id = node_id
        return node_id

    # ------------------------------------------------------------------
    def register_junction(self, branch_count: int) -> str:
        """Add a junction child of the current node and push it onto the DFS stack."""
        if branch_count < 1:
            raise ValueError(f"branch_count must be >= 1, got {branch_count}")
        node_id = self._new_id("junction")
        node = Node(
            id=node_id,
            type="junction",
            branches_total=int(branch_count),
            status="visited",
            parent_id=self.current_node_id,
        )
        self.nodes[node_id] = node
        if self.current_node_id is not None:
            self.nodes[self.current_node_id].children_ids.append(node_id)
        self.dfs_stack.append(node_id)
        self.current_node_id = node_id
        return node_id

    # ------------------------------------------------------------------
    def register_dead_end(self) -> str:
        """Add a calyx (leaf) child of the current node. Does NOT push the stack."""
        node_id = self._new_id("calyx")
        node = Node(
            id=node_id,
            type="calyx",
            status="visited",
            parent_id=self.current_node_id,
        )
        self.nodes[node_id] = node
        if self.current_node_id is not None:
            self.nodes[self.current_node_id].children_ids.append(node_id)
        self.current_node_id = node_id
        return node_id

    # ------------------------------------------------------------------
    def current_junction_id(self) -> str | None:
        """Top of DFS stack, or None if stack is empty."""
        return self.dfs_stack[-1] if self.dfs_stack else None

    # ------------------------------------------------------------------
    def mark_branch_explored(self) -> dict:
        """Increment branches_explored on the top-of-stack junction.

        Auto-pops the junction from the DFS stack once all branches are done.
        Returns the new status: branches_explored, branches_total, all_explored.
        """
        if not self.dfs_stack:
            raise RuntimeError("mark_branch_explored called with empty DFS stack")
        jid = self.dfs_stack[-1]
        node = self.nodes[jid]
        node.branches_explored = min(node.branches_explored + 1, node.branches_total)
        all_explored = node.branches_explored >= node.branches_total
        if all_explored:
            self.dfs_stack.pop()
        return {
            "junction_id": jid,
            "branches_explored": node.branches_explored,
            "branches_total": node.branches_total,
            "all_explored": all_explored,
        }

    # ------------------------------------------------------------------
    def set_current(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self.current_node_id = node_id

    def set_status(self, node_id: str, status: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        valid = {"unknown", "visited", "stones_found", "treating", "verified_clear"}
        if status not in valid:
            raise ValueError(f"invalid status {status!r}; valid: {sorted(valid)}")
        self.nodes[node_id].status = status

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """JSON-safe full dump for get_mission_status."""
        return {
            "current_node_id": self.current_node_id,
            "dfs_stack": list(self.dfs_stack),
            "nodes": {
                nid: {
                    "id": n.id,
                    "type": n.type,
                    "branches_total": n.branches_total,
                    "branches_explored": n.branches_explored,
                    "status": n.status,
                    "parent_id": n.parent_id,
                    "children_ids": list(n.children_ids),
                    "metadata": dict(n.metadata),
                }
                for nid, n in self.nodes.items()
            },
        }

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Quick aggregate counts for the mission status header."""
        junctions = sum(1 for n in self.nodes.values() if n.type == "junction")
        calyces = sum(1 for n in self.nodes.values() if n.type == "calyx")
        verified_clear = sum(
            1 for n in self.nodes.values() if n.type == "calyx" and n.status == "verified_clear"
        )
        stones_found = sum(
            1 for n in self.nodes.values() if n.type == "calyx" and n.status == "stones_found"
        )
        return {
            "junctions": junctions,
            "calyces": calyces,
            "stones_found": stones_found,
            "verified_clear": verified_clear,
            "dfs_depth": len(self.dfs_stack),
        }
