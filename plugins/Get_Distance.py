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


def find_shortest_path(graph, start_node, end_node):
    distance, predecessor_from_start1 = dijkstra(graph, start_node)
    if math.isinf(distance[end_node]):
        return None  # No path found
    path = [end_node]
    while path[-1] != start_node:
        path.append(predecessor_from_start1[path[-1]])
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
    graph = {node: {} for node in nodes}
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

    graph = get_graph_from_edges()
    distance, path = find_shortest_path(graph, start_node, end_node)
    if path:
        print(
            f"Shortest path {str(distance)} from {str(start_node)} to {str(end_node)}: {str(path)}"
        )
    else:
        print(f"No path found from {str(start_node)} to {str(end_node)}")
