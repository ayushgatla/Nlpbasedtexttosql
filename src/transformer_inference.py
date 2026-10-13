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

    # Recovery: if the model dropped an operator before a number in WHERE (e.g. "WHERE age 18")
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

    return predicted_sql
