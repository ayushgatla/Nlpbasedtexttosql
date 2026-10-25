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


def serialize_schema_for_prompt(db_id: str, tables_list: List[Dict[str, Any]]) -> str:
    """
    Serializes a database schema into a compact text string suitable for transformer input:
    'table1 ( col1, col2 ) | table2 ( col3, col4 )'
    """
    entry = next((d for d in tables_list if d["db_id"] == db_id), None)
    if not entry:
        return ""

    table_names = entry["table_names_original"]
    table_to_cols = {t: [] for t in table_names}

    for t_id, c_name in entry["column_names_original"]:
        if t_id >= 0 and c_name != "*":
            table_to_cols[table_names[t_id]].append(c_name)

    parts = [f"{tbl} ( {', '.join(cols)} )" for tbl, cols in table_to_cols.items()]
    return " | ".join(parts)


def format_seq2seq_prompt(
    question: str,
    db_id: str,
    tables_list: List[Dict[str, Any]]
) -> str:
    """
    Synthesizes the complete prompt for the fine-tuned Seq2Seq model.
    Applies comparative canonicalization to ensure robust polarity parsing.
    """
    canonical_q = canonicalize_comparatives(question)
    schema_str = serialize_schema_for_prompt(db_id, tables_list)
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


def predict_sql_transformer(
    question: str,
    db_id: str,
    tables_list: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
    num_beams: int = 4,
    max_length: int = 128
) -> str:
    """
    Translates a natural language question into SQL using beam search decoding.
    """
    prompt = format_seq2seq_prompt(question, db_id, tables_list)
    model.eval()

    inputs = tokenizer(
        prompt,
        max_length=384,
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

    raw_decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # T5's SentencePiece vocabulary treats '<' as <unk> (token ID 2).
    # Preserving <unk> and mapping it back to '<' prevents dropping the operator.
    predicted_sql = raw_decoded.replace("<pad>", "").replace("</s>", "").replace("<unk>", "<").strip()
    # Normalize duplicate spaces around operators
    predicted_sql = re.sub(r'\s+', ' ', predicted_sql)

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
        return f"{prefix}'{token}'"

    predicted_sql = re.sub(
        r'(\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?\s*(?:=|!=|<>|LIKE)\s*)([a-zA-Z_]\w*)\b',
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

    # 7. Multi-Table Relational Join Grounding & Automatic Repair
    if entry:
        try:
            from src.schema_graph import plan_table_joins
            from_match = re.search(r'\bFROM\s+([a-zA-Z_]\w*)', predicted_sql, re.IGNORECASE)
            if from_match:
                main_table = from_match.group(1)
                existing_tables = set(re.findall(r'\b(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', predicted_sql, re.IGNORECASE))
                existing_lowers = {t.lower() for t in existing_tables}

                col_to_table = {}
                table_names = entry.get("table_names_original", [])
                for t_id, c_name in entry.get("column_names_original", []):
                    if t_id >= 0 and c_name != "*":
                        col_to_table[c_name.lower()] = table_names[t_id]

                missing_tables = set()
                for token in re.findall(r'\b[a-zA-Z_]\w*\b', predicted_sql):
                    tok_low = token.lower()
                    if tok_low in col_to_table:
                        tbl = col_to_table[tok_low]
                        if tbl.lower() not in existing_lowers:
                            missing_tables.add(tbl)

                q_low = question.lower()
                for tbl in table_names:
                    if tbl.lower() not in existing_lowers and (tbl.lower() in q_low or tbl.rstrip("s").lower() in q_low):
                        missing_tables.add(tbl)

                if missing_tables:
                    needed = list(existing_tables) + list(missing_tables)
                    _, joins = plan_table_joins(needed, entry, primary_table=main_table)
                    if joins:
                        new_joins = [j for j in joins if j["table"].lower() not in existing_lowers]
                        if new_joins:
                            join_str = " ".join(f"JOIN {j['table']} ON {j['condition']}" for j in new_joins)
                            predicted_sql = re.sub(
                                rf'\bFROM\s+{re.escape(main_table)}\b',
                                f"FROM {main_table} {join_str}",
                                predicted_sql,
                                count=1,
                                flags=re.IGNORECASE
                            )
        except Exception:
            pass

    return predicted_sql

