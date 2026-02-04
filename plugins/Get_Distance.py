import heapq
import math


def dijkstra(graph, start_node):
    distance = {node: float("inf") for node in graph}
    predecessor = {node: None for node in graph}
    distance[start_node] = 0
    queue = [(0, start_node)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)

        for neighbor, weight in graph[current_node].items():
            new_distance = current_distance + weight
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                predecessor[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))
    return distance, predecessor


def find_all_reachable_nodes(graph, start_node):
    """Find all nodes reachable from start_node using BFS."""
    if start_node not in graph:
        return set()
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    while queue:
        current = queue.pop(0)
        for neighbor in graph[current].keys():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def find_shortest_path(graph, start_node, end_node):
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
        path.append(predecessor[path[-1]])
    return distance[end_node], path[::-1]


def get_graph_from_edges(edges: list):
    res = {}
    nodes = []
    for e in edges:
        res[(e[0], e[1])] = e[2]
        res[(e[1], e[0])] = e[2]
        nodes.append(e[0])
        nodes.append(e[1])
    nodes = sorted(set(nodes))

    edges = [(k[0], k[1], res[k]) for k in res.keys()]

    # Converting to a graph representation
    graph: dict[int, dict[int, float]] = {node: {} for node in nodes}
    for edge in edges:
        graph[edge[0]][edge[1]] = edge[2]
    return graph


if __name__ == "__main__":
    start_node = 1
    end_node = 5

    edges = [
        (1, 2, 1),  # (node1, node2, distance)
        (1, 3, 4),
        (2, 4, 2),
        (3, 4, 6),
        (3, 5, 3),
        (6, 4, 1),
        (5, 3, 2),
    ]

    graph: dict[int, dict[int, float]] = get_graph_from_edges(edges)
    try:
        distance, path = find_shortest_path(graph, start_node, end_node)
        print(
            f"Shortest path {str(distance)} from {str(start_node)} to {str(end_node)}: {str(path)}"
        )
    except ValueError as e:
        print(f"Error: {e}")
