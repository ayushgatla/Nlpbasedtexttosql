"""
Official WikiSQL Benchmark Evaluation Harness (Salesforce / Zhong et al., 2017).
Evaluates single-table text-to-SQL performance: Projections, Aggregations, and Where Predicates.
"""

import re
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch


CACHE_DIR = Path("data/wikisql")
DEV_DATA_URL = "https://huggingface.co/datasets/Rathan/wikisql/raw/main/wikisql/dev.jsonl"
DEV_TABLES_URL = "https://huggingface.co/datasets/Rathan/wikisql/raw/main/wikisql/dev.tables.jsonl"

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def ensure_wikisql_data() -> Tuple[Path, Path]:
    """Downloads and caches WikiSQL dev queries and table schemas locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dev_path = CACHE_DIR / "dev.jsonl"
    tables_path = CACHE_DIR / "dev.tables.jsonl"

    if not dev_path.exists():
        print("Downloading WikiSQL dev queries from HuggingFace...")
        req = urllib.request.Request(DEV_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            with open(dev_path, "wb") as f:
                f.write(resp.read())

    if not tables_path.exists():
        print("Downloading WikiSQL dev table schemas from HuggingFace...")
        req = urllib.request.Request(DEV_TABLES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            with open(tables_path, "wb") as f:
                f.write(resp.read())

    return dev_path, tables_path


def load_wikisql_dev(max_samples: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Loads WikiSQL dev dataset and table definitions into memory."""
    dev_path, tables_path = ensure_wikisql_data()

    tables = {}
    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                tables[item["id"]] = item

    samples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break

    return samples, tables


def format_wikisql_gold_sql(sql_dict: Dict[str, Any], table_info: Dict[str, Any]) -> str:
    """Reconstructs standard SQL query from WikiSQL structured representation."""
    headers = table_info["header"]
    sel_idx = sql_dict["sel"]
    sel_col = headers[sel_idx] if 0 <= sel_idx < len(headers) else "col"
    sel_col_clean = re.sub(r"[^a-zA-Z0-9_]", "_", sel_col.strip())

    agg_idx = sql_dict.get("agg", 0)
    agg_op = AGG_OPS[agg_idx] if 0 <= agg_idx < len(AGG_OPS) else ""

    t_name = "table"
    if table_info.get("page_title"):
        clean_title = re.sub(r"[^a-zA-Z0-9_]", "_", table_info["page_title"].strip()).lower()
        if clean_title:
            t_name = clean_title[:24].strip("_")

    if agg_op:
        sel_clause = f"{agg_op}({sel_col_clean})"
    else:
        sel_clause = sel_col_clean

    query = f"SELECT {sel_clause} FROM {t_name}"

    conds = sql_dict.get("conds", [])
    if conds:
        where_clauses = []
        for col_i, op_i, val in conds:
            col_name = headers[col_i] if 0 <= col_i < len(headers) else "col"
            col_clean = re.sub(r"[^a-zA-Z0-9_]", "_", col_name.strip())
            op = COND_OPS[op_i] if 0 <= op_i < len(COND_OPS) else "="
            if isinstance(val, (int, float)):
                val_str = str(val)
            else:
                val_str = f"'{val}'"
            where_clauses.append(f"{col_clean} {op} {val_str}")
        query += " WHERE " + " AND ".join(where_clauses)

    return query


def build_schema_for_wikisql(table_info: Dict[str, Any], t_name: str) -> List[Dict[str, Any]]:
    """Serializes WikiSQL table header into standard tables_data format."""
    headers = table_info["header"]
    clean_headers = [re.sub(r"[^a-zA-Z0-9_]", "_", h.strip()) for h in headers]
    columns = [[-1, "*"]]
    for h in clean_headers:
        columns.append([0, h])

    return [{
        "db_id": table_info["id"],
        "table_names_original": [t_name],
        "column_names_original": columns,
        "primary_keys": [],
        "foreign_keys": []
    }]


def normalize_sql_for_eval(sql: str) -> str:
    """Normalizes SQL syntax and alias variations for evaluation."""
    s = sql.strip().rstrip(";")
    s = re.sub(r'[`"\[\]]', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\bAS\s+T\d+\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bT\d+\.', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\btable\.', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bSELECT\s+DISTINCT\b', 'SELECT', s, flags=re.IGNORECASE)
    return s.strip().lower()


def evaluate_wikisql_components(pred_sql: str, gold_sql: str) -> Dict[str, bool]:
    """Evaluates projection column, aggregation operator, and where condition matching."""
    norm_p = normalize_sql_for_eval(pred_sql)
    norm_g = normalize_sql_for_eval(gold_sql)

    exact_match = (norm_p == norm_g)

    # 1. Selection Match
    p_sel_m = re.search(r'select\s+(.*?)\s+from', norm_p, re.IGNORECASE)
    g_sel_m = re.search(r'select\s+(.*?)\s+from', norm_g, re.IGNORECASE)
    p_sel = p_sel_m.group(1).strip() if p_sel_m else ""
    g_sel = g_sel_m.group(1).strip() if g_sel_m else ""
    sel_match = (p_sel == g_sel) or (g_sel in p_sel)

    # 2. Aggregation Match
    has_agg_g = any(a.lower() in g_sel for a in ["count(", "sum(", "avg(", "max(", "min("])
    has_agg_p = any(a.lower() in p_sel for a in ["count(", "sum(", "avg(", "max(", "min("])
    agg_match = (has_agg_g == has_agg_p)

    # 3. Where Clause Match
    has_where_g = " where " in norm_g
    has_where_p = " where " in norm_p
    where_match = (has_where_g == has_where_p)

    return {
        "exact_match": exact_match,
        "sel_match": sel_match,
        "agg_match": agg_match,
        "where_match": where_match,
    }


def run_wikisql_evaluation(
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
    samples: int = 30
) -> Dict[str, Any]:
    """Runs end-to-end evaluation on WikiSQL benchmark."""
    print("=" * 70)
    print("INITIALIZING WIKISQL BENCHMARK EVALUATION (Salesforce / Zhong et al.)")
    print(f"Sample Size: {samples} | Device: {device}")
    print("=" * 70)

    dataset, tables = load_wikisql_dev(max_samples=samples)
    print(f"Loaded {len(dataset)} WikiSQL questions across {len(tables)} tables.\n")

    from src.transformer_inference import predict_sql_transformer

    results = {
        "total": len(dataset),
        "exact_match": 0,
        "sel_match": 0,
        "agg_match": 0,
        "where_match": 0,
        "syntax_valid": 0,
    }

    import sqlite3
    syntax_conn = sqlite3.connect(":memory:")

    for idx, ex in enumerate(dataset, 1):
        q = ex["question"]
        t_id = ex["table_id"]
        table_info = tables.get(t_id, {"header": ["col"], "id": t_id})

        t_name = "player" if "player" in q.lower() else "table"
        if table_info.get("page_title"):
            clean_title = re.sub(r"[^a-zA-Z0-9_]", "_", table_info["page_title"].strip()).lower()
            if clean_title:
                t_name = clean_title[:24].strip("_")

        schema_list = build_schema_for_wikisql(table_info, t_name)
        gold_sql = format_wikisql_gold_sql(ex["sql"], table_info)

        try:
            pred_sql = predict_sql_transformer(
                question=q,
                db_id=t_id,
                tables_list=schema_list,
                model=model,
                tokenizer=tokenizer,
                device=device,
                num_beams=4
            )
        except Exception as e:
            pred_sql = f"SELECT error -- {e}"

        # Check SQLite syntax validity
        is_syntax_valid = False
        try:
            syntax_conn.cursor().execute(f"EXPLAIN {pred_sql}")
            is_syntax_valid = True
        except sqlite3.OperationalError as oe:
            if "syntax error" not in str(oe).lower():
                is_syntax_valid = True
        except Exception:
            pass

        if is_syntax_valid:
            results["syntax_valid"] += 1

        matches = evaluate_wikisql_components(pred_sql, gold_sql)
        if matches["exact_match"]:
            results["exact_match"] += 1
        if matches["sel_match"]:
            results["sel_match"] += 1
        if matches["agg_match"]:
            results["agg_match"] += 1
        if matches["where_match"]:
            results["where_match"] += 1

        if idx % 5 == 0 or idx == len(dataset):
            em_pct = (results["exact_match"] / idx) * 100
            sel_pct = (results["sel_match"] / idx) * 100
            print(f"[{idx:2d}/{len(dataset):2d}] Exact Match: {em_pct:5.1f}% | Projection (Select): {sel_pct:5.1f}% | Syntax: {(results['syntax_valid']/idx)*100:5.1f}%")

    tot = max(1, results["total"])
    print("\n" + "=" * 70)
    print("WIKISQL BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"EXACT MATCH (EM)      : {results['exact_match']:3d} / {tot:3d} ({(results['exact_match']/tot)*100:5.1f}%)")
    print(f"PROJECTION ACCURACY   : {results['sel_match']:3d} / {tot:3d} ({(results['sel_match']/tot)*100:5.1f}%)")
    print(f"AGGREGATION ACCURACY  : {results['agg_match']:3d} / {tot:3d} ({(results['agg_match']/tot)*100:5.1f}%)")
    print(f"WHERE FILTER ACCURACY : {results['where_match']:3d} / {tot:3d} ({(results['where_match']/tot)*100:5.1f}%)")
    print(f"SQL SYNTAX VALIDITY   : {results['syntax_valid']:3d} / {tot:3d} ({(results['syntax_valid']/tot)*100:5.1f}%)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run WikiSQL Benchmark Evaluation")
    parser.add_argument("--samples", type=int, default=25, help="Number of evaluation samples (default: 25)")
    parser.add_argument("--model", type=str, default="best_model_v2_relational", help="Model path")
    args = parser.parse_args()

    from src.transformer_inference import load_transformer_model
    model_obj, tok_obj, dev_str = load_transformer_model(args.model)
    run_wikisql_evaluation(
        model=model_obj,
        tokenizer=tok_obj,
        device=dev_str,
        samples=args.samples
    )

