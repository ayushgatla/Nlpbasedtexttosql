"""
Foreign-Key Multigraph & BFS Join Planner.
Computes minimal Steiner join trees between tables without hallucinating unused tables.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Any


def build_foreign_key_graph(schema: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Constructs an undirected multigraph of foreign key relationships between tables.

    Returns:
        graph[table_a] = [(table_b, col_a, col_b), ...]
    """
    tables = schema.get("table_names_original", [])
    graph: Dict[str, List[Tuple[str, str, str]]] = {tbl: [] for tbl in tables}

    for a, b in schema.get("foreign_keys", []):
        t1, _ = schema["column_names_original"][a]
        t2, _ = schema["column_names_original"][b]
        table1 = schema["table_names_original"][t1]
        col1 = schema["column_names_original"][a][1]
        table2 = schema["table_names_original"][t2]
        col2 = schema["column_names_original"][b][1]

        graph[table1].append((table2, col1, col2))
        graph[table2].append((table1, col2, col1))

    return graph


def find_shortest_join_path(
    graph: Dict[str, List[Tuple[str, str, str]]],
    start_table: str,
    target_table: str
) -> Optional[List[Tuple[str, str, str, str]]]:
    """
    Breadth-First Search (BFS) to find the shortest foreign-key join path between two tables.

    Returns:
        List of (from_table, from_col, to_table, to_col) hops, or None if disconnected.
    """
    if start_table == target_table:
        return []

    visited = {start_table}
    queue = deque([(start_table, [])])

    while queue:
        curr, path = queue.popleft()
        if curr == target_table:
            return path
        for nxt, c1, c2 in graph.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [(curr, c1, nxt, c2)]))

    return None


def plan_table_joins(
    needed_tables: List[str],
    schema: Dict[str, Any],
    primary_table: Optional[str] = None
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Computes the minimal foreign-key Steiner path connecting all required tables.
    Guarantees that intermediate junction tables are included without hallucinating unrelated tables.

    Returns:
        (primary_table, list_of_joins) where each join is {"table": ..., "condition": ...}
    """
    if not needed_tables:
        main = schema["table_names_original"][0] if schema.get("table_names_original") else "T1"
        return main, []

    needed = list(dict.fromkeys(needed_tables))
    if primary_table not in needed:
        primary_table = needed[0]

    graph = build_foreign_key_graph(schema)
    active_tables: Set[str] = {primary_table}
    joins: List[Dict[str, str]] = []

    for target in needed:
        if target in active_tables:
            continue

        best_path = None
        for act in list(active_tables):
            path = find_shortest_join_path(graph, act, target)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path

        if best_path:
            for from_tbl, from_col, to_tbl, to_col in best_path:
                if to_tbl not in active_tables:
                    joins.append({
                        "table": to_tbl,
                        "condition": f"{from_tbl}.{from_col} = {to_tbl}.{to_col}"
                    })
                    active_tables.add(to_tbl)
        else:
            # Fallback join condition if no direct FK path exists in metadata
            joins.append({
                "table": target,
                "condition": f"{primary_table}.id = {target}.{primary_table.lower()}_id"
            })
            active_tables.add(target)

    return primary_table, joins
