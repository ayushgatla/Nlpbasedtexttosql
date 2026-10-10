"""
Intermediate Representation (IR) Builder & Semantic Parser.
Extracts query intents, projections, multi-condition WHERE clauses, aggregations,
GROUP BY grouping attributes, and directional ORDER BY specifications.
"""

import re
from typing import Dict, List, Any, Optional
from src.schema_graph import plan_table_joins
from src.schema_linking import split_question_head_tail, link_schema_hybrid


def parse_multi_conditions(
    tail_text: str,
    tail_candidates: List[Dict[str, Any]],
    question: str
) -> List[Dict[str, Any]]:
    """
    Parses complex and multiple WHERE conditions joined by logical conjunctions (AND / OR).
    Handles patterns such as:
      - 'with population bigger than 1500 or smaller than 500'
      - 'older than 56 and younger than 75'
      - 'from France'
    """
    conditions = []

    # 1. Geographic origin check e.g. "from France", "from California"
    origin_m = re.search(r'\bfrom\s+([A-Z][a-z]+)\b', question)
    if origin_m and origin_m.group(1).lower() not in {"the", "a", "all", "each", "every", "descending", "ascending"}:
        origin_val = origin_m.group(1)
        loc_cand = next((c for c in tail_candidates if any(k in c["column"].lower() for k in ["country", "state", "city", "nation", "origin"])), None)
        if loc_cand:
            conditions.append({
                "column": f"{loc_cand['table']}.{loc_cand['column']}",
                "operator": "=",
                "value": origin_val,
                "is_string": True,
                "conjunction": None
            })

    if not tail_text or not tail_candidates:
        return conditions

    conj = "OR" if " or " in tail_text.lower() else ("AND" if " and " in tail_text.lower() else None)
    parts = re.split(r'\b(?:or|and)\b', tail_text, flags=re.IGNORECASE) if conj else [tail_text]

    rules = [
        ("greater than or equal", ">="), ("at least", ">="),
        ("less than or equal", "<="), ("at most", "<="),
        ("greater than", ">"), ("more than", ">"), ("older than", ">"),
        ("above age", ">"), ("above", ">"), ("bigger than", ">"), ("larger than", ">"),
        ("less than", "<"), ("smaller than", "<"), ("younger than", "<"),
        ("below age", "<"), ("below", "<"),
        ("not equal", "!="), ("not in", "!="), ("born outside", "!="),
        ("different from", "!="), ("equal to", "="), ("is not", "!="), ("isn't", "!=")
    ]

    for i, part in enumerate(parts):
        part_clean = part.strip()
        part_lower = part_clean.lower()

        op = "="
        for phrase, symbol in rules:
            if phrase in part_lower:
                op = symbol
                break

        quoted = re.findall(r"['\"]([^'\"]+)['\"]", part_clean)
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', part_clean)
        col = f"{tail_candidates[0]['table']}.{tail_candidates[0]['column']}"

        if quoted:
            conditions.append({
                "column": col, "operator": op, "value": quoted[0],
                "is_string": True, "conjunction": conj if i > 0 else None
            })
        elif nums:
            conditions.append({
                "column": col, "operator": op, "value": nums[0],
                "is_string": False, "conjunction": conj if i > 0 else None
            })
        elif not conditions:
            words = [
                w for w in re.findall(r'\b[A-Z][a-z]+\b', part_clean)
                if w.lower() not in {"what", "who", "where", "how", "when", "the", "in", "on", "at"}
            ]
            if words:
                conditions.append({
                    "column": col, "operator": op, "value": words[0],
                    "is_string": True, "conjunction": conj if i > 0 else None
                })

    return conditions


def build_advanced_ir(
    question: str,
    schema: Dict[str, Any],
    model: Any
) -> Dict[str, Any]:
    """
    Constructs an advanced Intermediate Representation (IR):
    - Separates head/tail context
    - Disambiguates sorting keywords from aggregations
    - Extracts multi-clause WHERE filters
    - Automatically infers GROUP BY when mixing aggregations and attributes
    """
    head_q, tail_q = split_question_head_tail(question)
    q_lower = question.lower()
    head_cands = link_schema_hybrid(head_q, schema, model, k=8)
    tail_cands = link_schema_hybrid(tail_q, schema, model, k=8) if tail_q else []

    # Flag if order phrase contains superlative words like oldest/youngest
    is_order_phrase = bool(
        re.search(r'ordered by.*from.*to', q_lower) or
        re.search(r'order by.*from.*to', q_lower) or
        "descending order" in q_lower or
        "ascending order" in q_lower
    )

    has_count = bool(re.search(r'\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b', q_lower))
    has_avg = bool(re.search(r'\baverage\b|\bavg\b|\bmean\b', q_lower))
    has_sum = bool(re.search(r'\btotal\b|\bsum\b', q_lower)) and not has_count
    has_max = bool(re.search(r'\bmaximum\b|\bmax\b', q_lower)) or (bool(re.search(r'\bhighest\b|\blargest\b|\bmost\b|\boldest\b', q_lower)) and not is_order_phrase)
    has_min = bool(re.search(r'\bminimum\b|\bmin\b', q_lower)) or (bool(re.search(r'\blowest\b|\bsmallest\b|\bleast\b|\byoungest\b', q_lower)) and not is_order_phrase)
    has_distinct = bool(re.search(r'\bdistinct\b|\bdifferent\b|\bunique\b', q_lower))

    # Parse WHERE conditions
    where_conditions = parse_multi_conditions(tail_q, tail_cands + head_cands, question)

    # SELECT and Aggregations
    select_items = []
    aggregations = []

    agg_funcs = []
    if has_avg: agg_funcs.append("AVG")
    if has_min and not is_order_phrase: agg_funcs.append("MIN")
    if has_max and not is_order_phrase: agg_funcs.append("MAX")
    if has_sum: agg_funcs.append("SUM")

    if agg_funcs and head_cands:
        for f in agg_funcs:
            aggregations.append((f, f"{head_cands[0]['table']}.{head_cands[0]['column']}"))
    elif has_count and not agg_funcs:
        aggregations.append(("COUNT", "*"))

    if not aggregations:
        head_words = set(re.findall(r'\w+', head_q.lower()))
        for c in head_cands:
            col_words = set(re.findall(r'\w+', c['column'].lower()))
            if col_words.intersection(head_words) or c['score'] > 0.45:
                ref = f"{c['table']}.{c['column']}"
                if ref not in select_items:
                    select_items.append(ref)
        if not select_items and head_cands:
            select_items.append(f"{head_cands[0]['table']}.{head_cands[0]['column']}")

    # ORDER BY & LIMIT
    order_by = None
    order_dir = None
    limit = None

    if re.search(r'\border\b|\bsorted\b|\branking\b', q_lower):
        m_ord = re.search(r'(?:order of|order by|sorted by|ordered by)\s+([a-zA-Z_]+)', q_lower)
        if m_ord:
            ord_cands = link_schema_hybrid(m_ord.group(1), schema, model, k=3)
            if ord_cands:
                order_by = f"{ord_cands[0]['table']}.{ord_cands[0]['column']}"
        if not order_by and head_cands:
            order_by = f"{head_cands[0]['table']}.{head_cands[0]['column']}"
        order_dir = "DESC" if "oldest to the youngest" in q_lower or "descending" in q_lower else "ASC"

    elif re.search(r'\b(youngest|oldest|highest|lowest|most|least)\b', q_lower) and not aggregations:
        limit = 1
        order_dir = "ASC" if "youngest" in q_lower or "lowest" in q_lower or "least" in q_lower else "DESC"
        age_cand = next((c for c in head_cands + tail_cands if "age" in c["column"].lower()), None)
        if age_cand:
            order_by = f"{age_cand['table']}.{age_cand['column']}"
        elif head_cands:
            order_by = f"{head_cands[0]['table']}.{head_cands[0]['column']}"

    # GROUP BY Inference: if aggregations exist AND select_items exist
    group_by = []
    if aggregations and select_items:
        group_by = list(select_items)
    if re.search(r'\bfor each\b|\bfor every\b|\bby each\b|\bper\b|\bgrouped by\b', q_lower):
        if select_items:
            group_by = list(select_items)

    needed_tables = set()
    for col_ref in select_items:
        if "." in col_ref: needed_tables.add(col_ref.split(".")[0])
    for func, col_ref in aggregations:
        if col_ref != "*" and "." in col_ref: needed_tables.add(col_ref.split(".")[0])
    for w in where_conditions:
        needed_tables.add(w["column"].split(".")[0])
    if order_by and "." in order_by:
        needed_tables.add(order_by.split(".")[0])

    if not needed_tables:
        needed_tables.add(head_cands[0]["table"] if head_cands else schema["table_names_original"][0])

    primary_table = head_cands[0]["table"] if head_cands else list(needed_tables)[0]
    main_table, joins = plan_table_joins(list(needed_tables), schema, primary_table=primary_table)

    return {
        "main_table": main_table,
        "select": select_items,
        "aggregations": aggregations,
        "distinct": has_distinct,
        "joins": joins,
        "where": where_conditions,
        "group_by": group_by,
        "order_by": order_by,
        "order_dir": order_dir,
        "limit": limit
    }
