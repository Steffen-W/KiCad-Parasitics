from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Hashable, Sequence
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class DisjointSets(Generic[T]):
    """Union-find over items joined pairwise; a new item is a group of its own."""

    def __init__(self) -> None:
        self._parent: dict[T, T] = {}

    def find(self, item: T) -> T:
        """The representative of the group of `item`."""
        parent = self._parent
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: T, b: T) -> None:
        self._parent[self.find(a)] = self.find(b)


def dijkstra(
    graph: dict[int, dict[int, float]], start_node: int
) -> tuple[dict[int, float], dict[int, int | None]]:
    distance = {node: float("inf") for node in graph}
    predecessor: dict[int, int | None] = {node: None for node in graph}
    distance[start_node] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start_node)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)

        for neighbor, weight in graph.get(current_node, {}).items():
            new_distance = current_distance + weight
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                predecessor[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))
    return distance, predecessor


def find_all_reachable_nodes(
    graph: dict[int, dict[int, float]], start_node: int
) -> set[int]:
    """Find all nodes reachable from start_node using BFS."""
    if start_node not in graph:
        return set()
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def find_shortest_path(
    graph: dict[int, dict[int, float]], start_node: int, end_node: int
) -> tuple[float, list[int]]:
    """
    Find the shortest path between two nodes in a weighted graph.

    Args:
        graph: Dictionary mapping nodes to their neighbors and edge weights
        start_node: Starting node ID
        end_node: Destination node ID

    Returns:
        Tuple of (distance, path) where path is a list of nodes

    Raises:
        ValueError: If start/end node not in graph or no path exists
    """
    if start_node not in graph:
        raise ValueError(
            f"Start node {start_node} not in graph. "
            f"Available nodes: {sorted(graph.keys())[:20]}..."
        )
    if end_node not in graph:
        raise ValueError(
            f"End node {end_node} not in graph. "
            f"Available nodes: {sorted(graph.keys())[:20]}..."
        )

    distance, predecessor = dijkstra(graph, start_node)

    if end_node not in distance or math.isinf(distance[end_node]):
        reachable = find_all_reachable_nodes(graph, start_node)
        raise ValueError(
            f"No path found from {start_node} to {end_node}. "
            f"Reachable from start: {len(reachable)} nodes. "
            f"End node is {'reachable' if end_node in reachable else 'NOT reachable'}."
        )

    path = [end_node]
    while path[-1] != start_node:
        prev = predecessor[path[-1]]
        if prev is None:
            raise RuntimeError(f"Broken predecessor chain at node {path[-1]}")
        path.append(prev)
    return distance[end_node], path[::-1]


def get_graph_from_edges(
    edges: Sequence[tuple[int, int, int | float]],
) -> dict[int, dict[int, float]]:
    graph: dict[int, dict[int, float]] = {}
    for n1, n2, weight in edges:
        graph.setdefault(n1, {})[n2] = weight
        graph.setdefault(n2, {})[n1] = weight
    return graph
