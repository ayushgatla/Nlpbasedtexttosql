"""
End-to-End Seq2Seq Transformer Inference Module.
Handles schema serialization, prompt synthesis, checkpoint loading, and beam search decoding.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def canonicalize_comparatives(question: str) -> str:
    """
    Normalizes comparative polarity expressions into canonical phrasing matching Spider distributions.
    Resolves antonym collapse / polarity blindness (e.g., 'under 50' -> 'less than 50')
    without degrading the model on standard benchmarks.
    """
    text = question
    substitutions = [
        (r'\b(at least|no less than)\b', 'greater than or equal to'),
        (r'\b(at most|no more than|up to)\b', 'less than or equal to'),
        (r'\b(lower than|lesser than|fewer than)\b', 'less than'),
        (r'\bunder\s+(\d+)\b', r'less than \1'),
        (r'\bbelow\s+(\d+)\b', r'less than \1'),
        (r'\b(higher than|more than)\b', 'greater than'),
        (r'\bover\s+(\d+)\b', r'greater than \1'),
        (r'\babove\s+(\d+)\b', r'greater than \1'),
    ]
    for pat, repl in substitutions:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def serialize_schema_for_prompt(
    db_id: str,
    tables_list: List[Dict[str, Any]],
    include_foreign_keys: bool = True
) -> str:
    """
    Serializes a database schema into an information-rich sequence representation:
    'table1 ( col1*, col2 ) | table2 ( col3*, col4 ) | Foreign Keys: table1.col2 = table2.col3'
    Includes Primary Keys (*) and Foreign Keys to provide relational context for multi-table joins.
    """
    entry = next((d for d in tables_list if d.get("db_id") == db_id), None)
    if not entry:
        return ""

    table_names = entry.get("table_names_original", [])
    table_to_cols = {t: [] for t in table_names}
    pk_cols = set(entry.get("primary_keys", []))

    for idx, (t_id, c_name) in enumerate(entry.get("column_names_original", [])):
        if t_id >= 0 and c_name != "*":
            is_pk = idx in pk_cols
            col_display = f"{c_name}*" if is_pk else c_name
            if t_id < len(table_names):
                table_to_cols[table_names[t_id]].append(col_display)

    parts = [f"{tbl} ( {', '.join(cols)} )" for tbl, cols in table_to_cols.items()]
    schema_str = " | ".join(parts)

    if include_foreign_keys and entry.get("foreign_keys"):
        fk_relations = []
        for a, b in entry["foreign_keys"]:
            cols = entry.get("column_names_original", [])
            if a < len(cols) and b < len(cols):
                t1, c1 = cols[a]
                t2, c2 = cols[b]
                if t1 < len(table_names) and t2 < len(table_names):
                    fk_relations.append(f"{table_names[t1]}.{c1} = {table_names[t2]}.{c2}")
        if fk_relations:
            schema_str += " | Foreign Keys: " + ", ".join(fk_relations)

    return schema_str


def format_seq2seq_prompt(
    question: str,
    db_id: str,
    tables_list: List[Dict[str, Any]],
    include_foreign_keys: bool = True
) -> str:
    """
    Synthesizes the complete prompt for the fine-tuned Seq2Seq model.
    Applies comparative canonicalization and relational schema serialization.
    """
    canonical_q = canonicalize_comparatives(question)
    schema_str = serialize_schema_for_prompt(db_id, tables_list, include_foreign_keys=include_foreign_keys)
    return f"Translate to SQL: {canonical_q} | Database: {db_id} | Tables: {schema_str}"


def load_transformer_model(
    model_path_or_name: str,
    device: Optional[str] = None
) -> Tuple[Any, Any, str]:
    """
    Loads tokenizer and Seq2Seq transformer weights onto the target compute hardware.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name).to(device)
    model.eval()
    return model, tokenizer, device


def repair_sql_schema_mismatches(
    predicted_sql: str,
    db_id: str,
    tables_list: List[Dict[str, Any]]
) -> str:
    """
    Validates and repairs relational schema mismatches in generated SQL:
    1. Reassigns columns to the correct table alias if misplaced (e.g. T2.Age -> T1.Age).
    2. Detects missing tables required by misplaced columns (e.g. T2.Concert_Name where Concert_Name
       belongs to table 'concert'), automatically plans the FK join, appends the JOIN clause,
       and assigns the new alias.
    3. Normalizes double-quoted literals to single quotes for SQLite.
    """
    entry = next((d for d in tables_list if d.get("db_id") == db_id), None)
    if not entry:
        return predicted_sql

    sql = predicted_sql.strip()

    # 1. Normalize double quotes around string literals to single quotes
    sql = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', r"'\1'", sql)

    table_names = entry.get("table_names_original", [])
    col_to_tables: Dict[str, List[str]] = {}
    table_to_cols: Dict[str, Dict[str, str]] = {t.lower(): {} for t in table_names}

    for t_id, c_name in entry.get("column_names_original", []):
        if t_id >= 0 and c_name != "*":
            t_orig = table_names[t_id]
            t_low = t_orig.lower()
            c_low = c_name.lower()
            col_to_tables.setdefault(c_low, []).append(t_orig)
            table_to_cols[t_low][c_low] = c_name

    # 1b. Ground literal values against live database cells to resolve column mismatches
    # e.g. T2.Concert_Name = 'Acoustic Evenings' -> checks if Acoustic Evenings is in concert_name vs Theme
    try:
        from src.mock_db import DB_CONNECTIONS
        conn = DB_CONNECTIONS.get(db_id)
        if conn:
            cur = conn.cursor()
            literal_matches = re.findall(r"([a-zA-Z_]\w*\.[a-zA-Z_]\w*)\s*(?:=|LIKE)\s*'([^']+)'", sql, re.IGNORECASE)
            for col_ref, lit_val in literal_matches:
                found_col = None
                for tbl in table_names:
                    cur.execute(f"PRAGMA table_info({tbl})")
                    for c_info in cur.fetchall():
                        col = c_info[1]
                        try:
                            cur.execute(f"SELECT 1 FROM {tbl} WHERE LOWER({col}) = LOWER(?) LIMIT 1", (lit_val,))
                            if cur.fetchone():
                                found_col = col
                                break
                        except Exception:
                            pass
                    if found_col:
                        break
                if found_col:
                    ref_prefix, ref_col = col_ref.split(".")
                    if ref_col.lower() != found_col.lower():
                        sql = re.sub(rf'\b{re.escape(col_ref)}\b', f"{ref_prefix}.{found_col}", sql)
    except Exception:
        pass

    # 2. Extract existing tables and aliases in FROM / JOIN
    alias_to_table: Dict[str, str] = {}
    table_to_alias: Dict[str, str] = {}
    used_alias_nums = set()

    for m in re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*)\s+(?:AS\s+)?(T\d+)\b', sql, re.IGNORECASE):
        tbl_name = m.group(1)
        alias = m.group(2)
        real_tbl = next((t for t in table_names if t.lower() == tbl_name.lower()), tbl_name)
        alias_to_table[alias.upper()] = real_tbl
        table_to_alias[real_tbl.lower()] = alias.upper()
        try:
            used_alias_nums.add(int(alias[1:]))
        except Exception:
            pass

    if not alias_to_table:
        for m in re.finditer(r'\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*)\b', sql, re.IGNORECASE):
            tbl_name = m.group(1)
            real_tbl = next((t for t in table_names if t.lower() == tbl_name.lower()), None)
            if real_tbl:
                table_to_alias[real_tbl.lower()] = real_tbl

    # 3. Find all column references like T1.ColumnName or Table.ColumnName
    missing_tables_to_add = []

    def inspect_col_ref(match):
        full_match = match.group(0)
        prefix = match.group(1)
        col_name = match.group(2)
        prefix_upper = prefix.upper()
        prefix_low = prefix.lower()
        col_low = col_name.lower()

        curr_table = alias_to_table.get(prefix_upper) or (prefix if prefix_low in table_to_cols else None)
        if not curr_table:
            return full_match

        if col_low in table_to_cols.get(curr_table.lower(), {}):
            real_col = table_to_cols[curr_table.lower()][col_low]
            return f"{prefix}.{real_col}"

        candidate_tables = col_to_tables.get(col_low, [])
        if not candidate_tables:
            return full_match

        # Case A: Belong to a table already in query
        for cand in candidate_tables:
            if cand.lower() in table_to_alias:
                correct_alias = table_to_alias[cand.lower()]
                real_col = table_to_cols[cand.lower()][col_low]
                return f"{correct_alias}.{real_col}"

        # Case B: Table missing from query
        target_table = candidate_tables[0]
        missing_tables_to_add.append((target_table, col_low))
        return f"__NEED_{target_table.upper()}__.{col_name}"

    sql_repaired = re.sub(
        r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b',
        inspect_col_ref,
        sql
    )

    # 4. If missing tables were detected, plan and insert foreign key joins
    if missing_tables_to_add:
        try:
            from src.schema_graph import build_foreign_key_graph, find_shortest_join_path
            fk_graph = build_foreign_key_graph(entry)

            next_num = max(used_alias_nums, default=1) + 1
            joins_to_append = []

            for target_table, col_low in missing_tables_to_add:
                if target_table.lower() in table_to_alias:
                    continue

                best_path = None
                for exist_alias, exist_tbl in alias_to_table.items():
                    path = find_shortest_join_path(fk_graph, exist_tbl, target_table)
                    if path is not None:
                        if best_path is None or len(path) < len(best_path):
                            best_path = path

                if best_path:
                    for from_tbl, from_col, to_tbl, to_col in best_path:
                        if to_tbl.lower() not in table_to_alias:
                            new_alias = f"T{next_num}"
                            next_num += 1
                            from_alias = table_to_alias.get(from_tbl.lower(), from_tbl)
                            joins_to_append.append(
                                f"JOIN {to_tbl} AS {new_alias} ON {from_alias}.{from_col} = {new_alias}.{to_col}"
                            )
                            table_to_alias[to_tbl.lower()] = new_alias
                            alias_to_table[new_alias] = to_tbl

                assigned_alias = table_to_alias.get(target_table.lower(), target_table)
                real_col = table_to_cols[target_table.lower()][col_low]
                sql_repaired = sql_repaired.replace(f"__NEED_{target_table.upper()}__", assigned_alias)

            if joins_to_append:
                join_str = " " + " ".join(joins_to_append)
                clause_match = re.search(r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b', sql_repaired, re.IGNORECASE)
                if clause_match:
                    split_pos = clause_match.start()
                    sql_repaired = sql_repaired[:split_pos].rstrip() + join_str + " " + sql_repaired[split_pos:].lstrip()
                else:
                    sql_repaired = sql_repaired.rstrip() + join_str
        except Exception:
            pass

    sql_repaired = re.sub(r'\s+', ' ', sql_repaired).strip()
    return sql_repaired


def predict_sql_transformer(
    question: str,
    db_id: str,
    tables_list: List[Dict[str, Any]],
    model: Any,
    tokenizer: Optional[Any] = None,
    processing_class: Optional[Any] = None,
    device: str = "cpu",
    num_beams: int = 4,
    max_length: int = 128,
    max_input_len: int = 512
) -> str:
    """
    Translates a natural language question into SQL using beam search decoding.
    Accepts either `tokenizer` or `processing_class` for HuggingFace compatibility.
    """
    tk = tokenizer if tokenizer is not None else processing_class
    if tk is None:
        raise ValueError("Either `tokenizer` or `processing_class` must be provided to predict_sql_transformer.")

    prompt = format_seq2seq_prompt(question, db_id, tables_list)
    model.eval()

    inputs = tk(
        prompt,
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

    raw_decoded = tk.decode(outputs[0], skip_special_tokens=False)
    # T5's SentencePiece vocabulary treats '<' as <unk> (token ID 2).
    # Preserving <unk> and mapping it back to '<' prevents dropping the operator.
    predicted_sql = raw_decoded.replace("<pad>", "").replace("</s>", "").replace("<unk>", "<").strip()
    # Normalize duplicate spaces around operators
    predicted_sql = re.sub(r'\s+', ' ', predicted_sql)

    # Normalize missing AS keyword for table aliases (e.g. FROM concert T1 -> FROM concert AS T1)
    # Required for official Spider AST parser which rejects aliases without AS
    predicted_sql = re.sub(r'\b(FROM|JOIN)\s+([a-zA-Z_]\w*)\s+(T\d+)\b', r'\1 \2 AS \3', predicted_sql, flags=re.IGNORECASE)

    # 1. Recovery: if the model dropped an operator before a number in WHERE (e.g. "WHERE age 18")
    if re.search(r'\bWHERE\s+([a-zA-Z_]\w*)\s+(\d+)\b', predicted_sql, re.IGNORECASE):
        # Determine operator from user question
        q_low = question.lower()
        if any(w in q_low for w in ["less", "under", "smaller", "lower", "younger", "fewer", "below"]):
            op = "<"
        elif any(w in q_low for w in ["greater", "more", "older", "above", "higher", "over"]):
            op = ">"
        else:
            op = "="
        predicted_sql = re.sub(
            r'\bWHERE\s+([a-zA-Z_]\w*)\s+(\d+)\b',
            rf'WHERE \1 {op} \2',
            predicted_sql,
            flags=re.IGNORECASE
        )

    # 2. Extract known columns and database cell values for schema & entity grounding
    known_cols = set()
    entry = next((d for d in tables_list if d.get("db_id") == db_id), None)
    if entry:
        table_names = entry.get("table_names_original", [])
        for t_id, c_name in entry.get("column_names_original", []):
            if c_name != "*":
                known_cols.add(c_name.lower())
                if t_id >= 0 and t_id < len(table_names):
                    known_cols.add(f"{table_names[t_id]}.{c_name}".lower())

    known_db_values = []
    try:
        from src.mock_db import DB_CONNECTIONS
        conn = DB_CONNECTIONS.get(db_id)
        if conn and entry:
            cur = conn.cursor()
            for tbl in entry.get("table_names_original", []):
                try:
                    cur.execute(f"PRAGMA table_info({tbl})")
                    for col_info in cur.fetchall():
                        c = col_info[1]
                        cur.execute(f"SELECT DISTINCT {c} FROM {tbl} WHERE {c} IS NOT NULL")
                        for row in cur.fetchall():
                            if row[0] is not None and isinstance(row[0], str) and str(row[0]).strip():
                                known_db_values.append(str(row[0]).strip())
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Auto-quote unquoted string literals after operators (e.g., WHERE role = manager -> WHERE role = 'manager')
    def quote_unquoted_literal(match):
        prefix = match.group(1)
        token = match.group(2)
        reserved = {'NULL', 'TRUE', 'FALSE', 'AND', 'OR', 'NOT', 'IS', 'IN', 'LIKE', 'BETWEEN', 'EXISTS'}
        if token.upper() in reserved:
            return match.group(0)
        if token.lower() in known_cols:
            return match.group(0)
        if re.match(r'^T\d+$', token, re.IGNORECASE):
            return match.group(0)
        return f"{prefix}'{token}'"

    predicted_sql = re.sub(
        r'(\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?\s*(?:=|!=|<>|LIKE)\s*)([a-zA-Z_]\w*)\b(?!\.)',
        quote_unquoted_literal,
        predicted_sql,
        flags=re.IGNORECASE
    )

    # 4. Entity / Value Grounding for single-quoted literals (e.g., 'mathematician' -> 'Mathematics')
    def ground_literal(match):
        val = match.group(1)
        val_low = val.lower()
        # Check database values first
        for dv in known_db_values:
            if dv.lower() == val_low:
                return f"'{dv}'"
            if len(val_low) >= 4 and len(dv) >= 4 and val_low[:4] == dv.lower()[:4]:
                return f"'{dv}'"
        # Check tokens from question
        q_words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
        for qw in q_words:
            if len(val_low) >= 4 and len(qw) >= 4 and val_low[:4] == qw.lower()[:4]:
                return f"'{qw}'"
        return f"'{val}'"

    predicted_sql = re.sub(r"'([^']+)'", ground_literal, predicted_sql)

    # 5. Schema table name alignment (e.g. FROM student -> FROM students if table is plural/singular)
    if entry:
        table_names = entry.get("table_names_original", [])
        tbl_map = {t.lower(): t for t in table_names}
        for t in table_names:
            t_low = t.lower()
            if t_low.endswith("s"):
                tbl_map[t_low[:-1]] = t
            else:
                tbl_map[t_low + "s"] = t

        def fix_table_name(m):
            clause = m.group(1)
            tbl_used = m.group(2)
            if tbl_used.lower() in tbl_map:
                return f"{clause} {tbl_map[tbl_used.lower()]}"
            return m.group(0)

        predicted_sql = re.sub(r'\b(FROM|JOIN)\s+([a-zA-Z_]\w*)\b', fix_table_name, predicted_sql, flags=re.IGNORECASE)

    # 6. Polarity alignment: if question expresses negation but model decoded '='
    has_neg = bool(re.search(r'\b(?:did not|didn\'t|does not|doesn\'t|do not|don\'t|without|except|never|not)\b', question, re.IGNORECASE))
    if has_neg and " != " not in predicted_sql and " NOT " not in predicted_sql.upper() and " <> " not in predicted_sql:
        if "WHERE" in predicted_sql.upper():
            where_idx = predicted_sql.upper().find("WHERE")
            before_w = predicted_sql[:where_idx]
            after_w = predicted_sql[where_idx:]
            after_w_fixed = re.sub(r'(\s+=\s+)', r' != ', after_w, count=1)
            predicted_sql = before_w + after_w_fixed

    # 7. Relational Schema Grounding & Syntax Repair
    # Reassigns misplaced column aliases, auto-joins missing tables via foreign keys,
    # and fixes quote delimiters for clean SQLite execution.
    predicted_sql = repair_sql_schema_mismatches(predicted_sql, db_id, tables_list)

    return predicted_sql

