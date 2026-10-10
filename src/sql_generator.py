"""
Deterministic SQL Compiler.
Compiles structured Intermediate Representation (IR) into standard SQL with
clean clause formatting, join conditions, WHERE filters, GROUP BY, and ORDER BY.
"""

from typing import Dict, Any


def compile_advanced_sql(ir: Dict[str, Any]) -> str:
    """
    Compiles an IR dictionary into syntactically valid standard SQL.
    
    IR format expected:
      - main_table: str
      - select: List[str]
      - aggregations: List[Tuple[str, str]] e.g. [('AVG', 'singer.Age')]
      - distinct: bool
      - joins: List[Dict[str, str]] e.g. [{'table': 'concert', 'condition': '...'}]
      - where: List[Dict[str, Any]]
      - group_by: List[str]
      - order_by: Optional[str]
      - order_dir: Optional[str]
      - limit: Optional[int]
    """
    select_parts = []
    dist = "DISTINCT " if ir.get("distinct") else ""

    for col in ir.get("select", []):
        select_parts.append(col)

    for func, col in ir.get("aggregations", []):
        select_parts.append(f"{func}({col})")

    if not select_parts:
        select_parts.append("*")

    sql = f"SELECT {dist}{', '.join(select_parts)} FROM {ir['main_table']}"

    for j in ir.get("joins", []):
        sql += f" JOIN {j['table']} ON {j['condition']}"

    if ir.get("where"):
        where_parts = []
        for i, w in enumerate(ir["where"]):
            val = f"'{w['value']}'" if w.get("is_string") else str(w["value"])
            part_str = f"{w['column']} {w['operator']} {val}"
            if i > 0 and w.get("conjunction"):
                where_parts.append(f"{w['conjunction']} {part_str}")
            else:
                where_parts.append(part_str)
        sql += " WHERE " + " ".join(where_parts)

    if ir.get("group_by"):
        sql += " GROUP BY " + ", ".join(ir["group_by"])

    if ir.get("order_by"):
        sql += f" ORDER BY {ir['order_by']}"
        if ir.get("order_dir"):
            sql += f" {ir['order_dir']}"

    if ir.get("limit"):
        sql += f" LIMIT {ir['limit']}"

    return sql
