"""
Official Spider AST Evaluation Harness and Component Matching Benchmark.
Wraps spider/process_sql.py and spider/evaluation.py for difficulty-stratified accuracy.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

# Ensure spider parser package is in sys.path
repo_root = Path(__file__).resolve().parent.parent
spider_path = repo_root / "spider"
if str(spider_path) not in sys.path:
    sys.path.append(str(spider_path))

try:
    from process_sql import Schema as SpiderSchema, get_sql
    from evaluation import (
        Evaluator,
        build_valid_col_units,
        rebuild_sql_val,
        rebuild_sql_col,
        build_foreign_key_map
    )
except ImportError:
    SpiderSchema = None
    get_sql = None
    Evaluator = None


def build_spider_schema(db_id: str, tables_data: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    """Constructs Spider AST Schema and Foreign-Key Map from tables.json entry."""
    entry = next(d for d in tables_data if d["db_id"] == db_id)
    t_names = [t.lower() for t in entry["table_names_original"]]
    schema_dict = {t: [] for t in t_names}
    for t_id, c_name in entry["column_names_original"]:
        if t_id >= 0:
            schema_dict[t_names[t_id]].append(c_name.lower())
    return SpiderSchema(schema_dict), build_foreign_key_map(entry)


def evaluate_single_ast_match(
    pred_sql: str,
    gold_sql: str,
    db_id: str,
    tables_data: List[Dict[str, Any]],
    evaluator: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    Evaluates AST exact match between a predicted SQL query and a gold reference query.
    Returns:
        (is_match: bool, hardness_tier: str)
    """
    if evaluator is None:
        evaluator = Evaluator()

    try:
        sp_sch, kmap = build_spider_schema(db_id, tables_data)
        g_sql = get_sql(sp_sch, gold_sql)
        hardness = evaluator.eval_hardness(g_sql)

        p_sql = get_sql(sp_sch, pred_sql)

        g_val = build_valid_col_units(g_sql['from']['table_units'], sp_sch)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_val, g_sql, kmap)

        p_val = build_valid_col_units(p_sql['from']['table_units'], sp_sch)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_val, p_sql, kmap)

        match = evaluator.eval_exact_match(p_sql, g_sql)
        return bool(match), hardness
    except Exception:
        return False, "unknown"


def evaluate_spider_ast_benchmark(
    dataset: List[Dict[str, Any]],
    tables_data: List[Dict[str, Any]],
    predict_fn: Callable[[str, str], str],
    num_samples: int = 100
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates any prediction function on a slice of Spider queries using the official AST parser.
    """
    evaluator = Evaluator()
    hardness_counts = {"easy": 0, "medium": 0, "hard": 0, "extra": 0, "all": 0}
    hardness_correct = {"easy": 0, "medium": 0, "hard": 0, "extra": 0, "all": 0}

    for i, ex in enumerate(dataset[:num_samples]):
        q = ex["question"]
        gold_sql = ex["query"]
        db_id = ex["db_id"]

        try:
            pred_sql = predict_fn(q, db_id)
        except Exception:
            pred_sql = ""

        try:
            sp_sch, kmap = build_spider_schema(db_id, tables_data)
            g_sql = get_sql(sp_sch, gold_sql)
            hardness = evaluator.eval_hardness(g_sql)
            hardness_counts[hardness] = hardness_counts.get(hardness, 0) + 1
            hardness_counts["all"] += 1

            p_sql = get_sql(sp_sch, pred_sql)
            g_val = build_valid_col_units(g_sql['from']['table_units'], sp_sch)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_val, g_sql, kmap)

            p_val = build_valid_col_units(p_sql['from']['table_units'], sp_sch)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_val, p_sql, kmap)

            match = evaluator.eval_exact_match(p_sql, g_sql)
            if match:
                hardness_correct[hardness] = hardness_correct.get(hardness, 0) + 1
                hardness_correct["all"] += 1
        except Exception:
            pass

    summary = {}
    for tier in ["easy", "medium", "hard", "extra", "all"]:
        total = hardness_counts.get(tier, 0)
        corr = hardness_correct.get(tier, 0)
        acc = round((corr / max(total, 1)) * 100, 2)
        summary[tier] = {"correct": corr, "total": total, "accuracy": acc}

    return summary
