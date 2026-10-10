"""
Contextual Hybrid Schema Linking Module.
Combines dense MiniLM embeddings with token-level lexical overlap and question Head/Tail splitting.
"""

import re
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def split_question_head_tail(question: str) -> Tuple[str, str]:
    """
    Separates question into:
      1. Target Entity / Projection clause (Head: e.g. "What are the names of the heads")
      2. Condition / Filter clause (Tail: e.g. "who are born outside California")
    """
    q_clean = question.strip().rstrip("?").strip()
    pattern = r'\b(where|whose|with|that have|that has|that are|who have|who has|who are|which have|which has|which are|having|for which|older than|younger than|greater than|less than|more than|born in|born outside|after|before)\b'
    m = re.search(pattern, q_clean, re.IGNORECASE)
    if m:
        head = q_clean[:m.start()].strip()
        tail = q_clean[m.start():].strip()
        return head, tail
    return q_clean, ""


def link_schema_hybrid(
    text: str,
    schema: Dict[str, Any],
    model: Any,
    k: int = 8
) -> List[Dict[str, Any]]:
    """
    Performs hybrid dense + lexical schema linking:
    - Dense representation encodes full contextual string: "table {tbl} column {col}"
    - Lexical boost rewards exact token and stem overlap between question and schema names.
    """
    columns = []
    for table_id, column_name in schema.get("column_names_original", []):
        if table_id != -1:
            table_name = schema["table_names_original"][table_id]
            columns.append((table_name, column_name))

    if not columns or not text.strip():
        return []

    # Dense contextual embedding
    column_texts = [
        f"table {tbl} column {col} ({col.replace('_', ' ')})"
        for tbl, col in columns
    ]

    col_emb = model.encode(column_texts, normalize_embeddings=True)
    txt_emb = model.encode([text], normalize_embeddings=True)
    dense_scores = cosine_similarity(txt_emb, col_emb)[0]

    # Lexical overlap boost
    t_words = set(re.findall(r'\w+', text.lower()))
    ranked = []

    for (tbl, col), d_score in zip(columns, dense_scores):
        score = float(d_score)
        c_words = set(re.findall(r'\w+', col.lower()))

        # Exact word match
        overlap = c_words.intersection(t_words)
        if overlap:
            score += 0.25 * (len(overlap) / len(c_words))

        # Substring / exact phrase match
        col_norm = col.replace("_", " ").lower()
        if col.lower() in text.lower() or col_norm in text.lower():
            score += 0.15

        ranked.append(((tbl, col), score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    return [
        {
            "table": tbl,
            "column": col,
            "full_name": f"{tbl}.{col}",
            "score": sc
        }
        for (tbl, col), sc in ranked[:k]
    ]


def link_schema_lexical(
    text: str,
    schema: Dict[str, Any],
    k: int = 8
) -> List[Dict[str, Any]]:
    """Lexical token-overlap baseline schema linking."""
    columns = []
    for table_id, column_name in schema.get("column_names_original", []):
        if table_id != -1:
            table_name = schema["table_names_original"][table_id]
            columns.append((table_name, column_name))

    if not columns or not text.strip():
        return []

    t_words = set(re.findall(r'\w+', text.lower()))
    ranked = []
    for tbl, col in columns:
        c_words = set(re.findall(r'\w+', col.lower()))
        score = len(c_words.intersection(t_words)) / max(len(c_words), 1)
        if col.lower() in text.lower():
            score += 1.0
        ranked.append(((tbl, col), score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    return [
        {"table": tbl, "column": col, "full_name": f"{tbl}.{col}", "score": sc}
        for (tbl, col), sc in ranked[:k]
    ]


def link_schema_dense(
    text: str,
    schema: Dict[str, Any],
    model: Any,
    k: int = 8
) -> List[Dict[str, Any]]:
    """Dense embedding only schema linking."""
    columns = []
    for table_id, column_name in schema.get("column_names_original", []):
        if table_id != -1:
            table_name = schema["table_names_original"][table_id]
            columns.append((table_name, column_name))

    if not columns or not text.strip():
        return []

    column_texts = [f"table {tbl} column {col}" for tbl, col in columns]
    col_emb = model.encode(column_texts, normalize_embeddings=True)
    txt_emb = model.encode([text], normalize_embeddings=True)
    scores = cosine_similarity(txt_emb, col_emb)[0]

    ranked = sorted(zip(columns, scores), key=lambda x: x[1], reverse=True)
    return [
        {"table": tbl, "column": col, "full_name": f"{tbl}.{col}", "score": float(sc)}
        for (tbl, col), sc in ranked[:k]
    ]
