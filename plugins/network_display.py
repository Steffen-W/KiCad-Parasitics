"""Network visualization and formatting for the 'Network Details' dialog."""

from collections import deque

try:
    from .pcb_types import WIRE, VIA, ZONE
except ImportError:
    from pcb_types import WIRE, VIA, ZONE


def simplify_graph(graph, start, end):
    """Simplify graph by removing dead-ends and compressing chains.

    Args:
        graph: adjacency dict {node: {neighbor: dist, ...}, ...}
        start, end: nodes that must not be removed

    Returns:
        (simplified_graph, chain_map)
        chain_map: {(min(a,c), max(a,c)): [a, ..., c]} for compressed chains
    """
    # Deep copy
    g = {n: dict(neighbors) for n, neighbors in graph.items()}

    # Step 1: Dead-end pruning (cascading)
    changed = True
    while changed:
        changed = False
        for node in list(g.keys()):
            if node == start or node == end:
                continue
            if len(g[node]) == 1:
                neighbor = next(iter(g[node]))
                del g[neighbor][node]
                del g[node]
                changed = True

    # Step 2: Chain compression
    chain_map = {}
    changed = True
    while changed:
        changed = False
        for node in list(g.keys()):
            if node == start or node == end:
                continue
            if node not in g:
                continue
            if len(g[node]) == 2:
                a, c = list(g[node].keys())
                # Skip if a↔c already connected (would create parallel edge)
                if c in g[a]:
                    continue
                dist_ab = g[node][a]
                dist_bc = g[node][c]
                new_dist = dist_ab + dist_bc

                # Build the chain by expanding existing chain_map entries
                key_a = (min(a, node), max(a, node))
                key_c = (min(node, c), max(node, c))
                left = chain_map.pop(key_a, [a, node])
                right = chain_map.pop(key_c, [node, c])

                # Orient left to end with node, right to start with node
                if left[0] == node:
                    left = list(reversed(left))
                if right[-1] == node:
                    right = list(reversed(right))

                # Merge: left ends with node, right starts with node
                full_chain = left + right[1:]

                chain_map[(min(a, c), max(a, c))] = full_chain

                # Remove node, connect a↔c
                del g[node]
                del g[a][node]
                del g[c][node]
                g[a][c] = new_dist
                g[c][a] = new_dist
                changed = True
                break  # restart iteration after modification

    return g, chain_map


def find_parallel_paths(graph, path, max_paths=50):
    """Find alternative paths between start and end via BFS.

    Expects a simplified graph (dead-ends removed, chains compressed).
    Only explores nodes reachable within 1 hop of the path.

    Args:
        graph: adjacency dict {node: {neighbor: dist, ...}}
        path: the shortest path [start, ..., end]
        max_paths: maximum number of paths to return
    Returns:
        list of paths (each path is a list of nodes)
    """
    if len(path) < 2:
        return [path]

    start, end = path[0], path[-1]

    # Only allow path nodes + their direct neighbors
    allowed = set(path)
    for node in path:
        if node in graph:
            allowed.update(graph[node])

    all_paths = []
    queue = deque([(start, [start])])

    while queue and len(all_paths) < max_paths:
        node, current_path = queue.popleft()

        if node == end:
            all_paths.append(current_path)
            continue

        if node not in graph:
            continue

        for neighbor in graph[node]:
            if neighbor in current_path:
                continue
            if neighbor not in allowed:
                continue
            queue.append((neighbor, current_path + [neighbor]))

    if path not in all_paths:
        all_paths.insert(0, path)

    return all_paths


def render_network_ascii(graph, path, network_info):
    """Render network path with parallel branches as ASCII.

    Output structure:
    - Prefix: Series elements common to all paths (nX / │ type R)
    - Parallel block: All alternative paths between split and merge node
    - Suffix: Series elements after merge point

    Uses a prefix trie for correct indentation of branching paths.
    """
    if len(path) < 2:
        return "Path too short"

    res_lookup = {}
    for info in network_info:
        n1, n2 = info["nodes"]
        res_lookup[(min(n1, n2), max(n1, n2))] = info

    def get_res(a, b):
        key = (min(a, b), max(a, b))
        return res_lookup[key]["resistance"] if key in res_lookup else 0

    def get_type(a, b):
        key = (min(a, b), max(a, b))
        return res_lookup[key]["type"][0] if key in res_lookup else "?"

    # Simplify graph before path search
    sgraph, chain_map = simplify_graph(graph, path[0], path[-1])

    def expand_edge(a, b):
        """Expand a simplified edge to full node list via chain_map."""
        key = (min(a, b), max(a, b))
        if key in chain_map:
            nodes = chain_map[key]
            if nodes[0] == a:
                return nodes
            return list(reversed(nodes))
        return [a, b]

    def expand_path(p):
        """Expand a simplified path to full node list."""
        if len(p) < 2:
            return p
        full = [p[0]]
        for i in range(len(p) - 1):
            expanded = expand_edge(p[i], p[i + 1])
            full.extend(expanded[1:])
        return full

    def path_res(p):
        ep = expand_path(p)
        return sum(get_res(ep[i], ep[i + 1]) for i in range(len(ep) - 1))

    # Rebuild shortest path in simplified graph
    spath = [n for n in path if n in sgraph]
    if len(spath) < 2:
        spath = [path[0], path[-1]]

    # Find all paths (bounded search) on simplified graph
    all_paths = find_parallel_paths(sgraph, spath)
    if not all_paths:
        all_paths = [spath]
    all_paths.sort(key=path_res)

    # Common prefix length
    plen = 0
    for i in range(min(len(p) for p in all_paths)):
        if len(set(p[i] for p in all_paths)) == 1:
            plen = i + 1
        else:
            break

    # Common suffix length
    slen = 0
    for i in range(1, min(len(p) for p in all_paths) + 1):
        if len(set(p[-i] for p in all_paths)) == 1:
            slen = i
        else:
            break

    lines = []

    # Check if all paths are identical (no parallel part)
    mids = [p[plen - 1 : len(p) - slen + 1] for p in all_paths]
    has_parallel = any(len(m) >= 2 for m in mids)

    if not has_parallel:
        # Pure series: render full expanded path
        ep = expand_path(all_paths[0])
        for i in range(len(ep) - 1):
            lines.append(f"n{ep[i]}")
            lines.append(
                f" │ {get_type(ep[i], ep[i + 1])} {get_res(ep[i], ep[i + 1]) * 1000:.2f}mΩ"
            )
        lines.append(f"n{ep[-1]}")
        return "\n".join(lines)

    # Prefix (series) — expand compressed edges
    for i in range(plen - 1):
        a, b = all_paths[0][i], all_paths[0][i + 1]
        expanded = expand_edge(a, b)
        for j in range(len(expanded) - 1):
            lines.append(f"n{expanded[j]}")
            lines.append(
                f" │ {get_type(expanded[j], expanded[j + 1])} {get_res(expanded[j], expanded[j + 1]) * 1000:.2f}mΩ"
            )

    # Parallel part
    lines.append(f"n{all_paths[0][plen - 1]}")

    # Build prefix trie for correct indentation
    trie = {}
    for mid in mids:
        node = trie
        for step in mid[1:]:  # skip split node (already printed)
            if step not in node:
                node[step] = {}
            node = node[step]
        node["_leaf"] = True

    merge_node = mids[0][-1]  # the node where all parallel paths rejoin

    def chain_res(parent, chain):
        """Resistance of a chain of nodes starting from parent."""
        full = [parent]
        for node in chain:
            expanded = expand_edge(full[-1], node)
            full.extend(expanded[1:])
        return sum(get_res(full[i], full[i + 1]) for i in range(len(full) - 1))

    def render_trie(node, parent, prefix_str):
        """Recursively render trie branches.  parent = last printed node."""
        children = {k: v for k, v in node.items() if k != "_leaf"}
        child_list = sorted(children.keys())
        for idx, child in enumerate(child_list):
            is_last = idx == len(child_list) - 1
            connector = "└─" if is_last else "├─"
            cont_line = "   " if is_last else "│  "

            subtree = children[child]
            # Compress linear chain (single non-leaf child) into one line
            chain = [child]
            sub = subtree
            while True:
                sub_children = {k: v for k, v in sub.items() if k != "_leaf"}
                if len(sub_children) == 1 and "_leaf" not in sub:
                    nxt = list(sub_children.keys())[0]
                    chain.append(nxt)
                    sub = sub_children[nxt]
                else:
                    break

            # Expand compressed edges for display
            display_nodes = [parent]
            for node in chain:
                expanded = expand_edge(display_nodes[-1], node)
                display_nodes.extend(expanded[1:])
            display_nodes = display_nodes[1:]  # remove parent (already printed)

            # Replace final merge node with ⤵ symbol
            if "_leaf" in sub and not any(k != "_leaf" for k in sub):
                chain_str = "→".join(
                    f"n{n}" if n != merge_node else "⤵" for n in display_nodes
                )
            else:
                chain_str = "→".join(f"n{n}" for n in display_nodes)
            r = chain_res(parent, chain)
            lines.append(f"{prefix_str}{connector} {chain_str} ({r * 1000:.2f}mΩ)")
            if any(k != "_leaf" for k in sub):
                render_trie(sub, chain[-1], prefix_str + cont_line)

    render_trie(trie, mids[0][0], " ")

    # Parallel resistance (total per path)
    mid_resistances = [path_res(mid) for mid in mids]
    if all(r > 0 for r in mid_resistances):
        r_par = 1 / sum(1 / r for r in mid_resistances)
        lines.append(f" R_parallel = {r_par * 1000:.2f}mΩ")

    lines.append(f"n{all_paths[0][-slen]}")

    # Suffix (series) — expand compressed edges
    for i in range(slen - 1):
        a, b = all_paths[0][-slen + i], all_paths[0][-slen + i + 1]
        expanded = expand_edge(a, b)
        for j in range(len(expanded) - 1):
            lines.append(
                f" │ {get_type(expanded[j], expanded[j + 1])} {get_res(expanded[j], expanded[j + 1]) * 1000:.2f}mΩ"
            )
            lines.append(f"n{expanded[j + 1]}")

    return "\n".join(lines)


def format_network_info(network_info, graph=None, path=None):
    """Format network info as human-readable string for debugging.

    Args:
        network_info: List of dicts with resistance element info
        graph: Optional adjacency dict for path visualization
        path: Optional list of nodes for path visualization
    """
    lines = ["RESISTANCE NETWORK", "-" * 40]

    # Normalize nodes (lower first) and sort by (n1, n2)
    sorted_info = sorted(network_info, key=lambda x: (min(x["nodes"]), max(x["nodes"])))

    for item in sorted_info:
        n1, n2 = sorted(item["nodes"])  # lower node first
        t = item["type"]
        r = item["resistance"]

        if t == WIRE:
            lines.append(
                f"WIRE  n{n1}--n{n2}  R={r * 1000:.3f}mΩ  "
                f"L={item['length'] * 1000:.3f}mm W={item['width'] * 1000:.3f}mm {item['layer_name']}"
            )
        elif t == VIA:
            lines.append(
                f"VIA   n{n1}--n{n2}  R={r * 1000:.3f}mΩ  "
                f"D={item['drill'] * 1000:.2f}mm {item['layer1_name']}<->{item['layer2_name']}"
            )
        elif t == ZONE:
            lines.append(
                f"ZONE  n{n1}--n{n2}  R={r * 1000:.3f}mΩ  {item['layer_name']}"
            )

    lines.append(f"Total: {len(network_info)} elements")

    # Path visualization with parallel branches
    if graph and path and len(path) >= 2:
        lines.append("")
        lines.append("PATH VISUALIZATION (start → end):")
        lines.append("-" * 40)
        lines.append(render_network_ascii(graph, path, network_info))

    return "\n".join(lines)


if __name__ == "__main__":
    # Minimal test: dead-ends (7,8), chain compression (3,4), parallel branch (2→9→5)
    #   1→2→3→4→5→6   dead-end 3→7→8   parallel 2→9→5
    def _w(n1, n2, r):
        return {
            "type": WIRE,
            "nodes": (n1, n2),
            "resistance": r,
            "length": 0.005,
            "width": 0.2e-3,
            "layer": 0,
            "layer_name": "F.Cu",
            "start": (0, 0),
            "end": (1, 0),
        }

    ni = [
        _w(1, 2, 0.01),
        _w(2, 3, 0.01),
        _w(3, 4, 0.01),
        _w(4, 5, 0.01),
        _w(5, 6, 0.01),
        _w(3, 7, 0.005),
        _w(7, 8, 0.005),
        _w(2, 9, 0.02),
        _w(9, 5, 0.02),
    ]
    g = {
        1: {2: 0.005},
        2: {1: 0.005, 3: 0.005, 9: 0.01},
        3: {2: 0.005, 4: 0.005, 7: 0.003},
        4: {3: 0.005, 5: 0.005},
        5: {4: 0.005, 6: 0.005, 9: 0.01},
        6: {5: 0.005},
        7: {3: 0.003, 8: 0.003},
        8: {7: 0.003},
        9: {2: 0.01, 5: 0.01},
    }

    sg, cm = simplify_graph(g, 1, 6)
    print(f"Simplified: {sorted(sg.keys())}  chains: {cm}")
    print(render_network_ascii(g, [1, 2, 3, 4, 5, 6], ni))
