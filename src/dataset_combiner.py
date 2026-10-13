"""
Unified Cross-Domain Text-to-SQL Dataset Aggregator.
Combines Spider (7,000), Spider-Syn (7,000), SParC (3,034), and Counterfactual
Polarity Inversions into a unified, balanced 19,000+ example training corpus.
All datasets share the identical 200 Spider schemas (tables.json).
"""

import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import datasets


def load_local_spider_train(spider_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads the original Spider training split."""
    if spider_path is None:
        candidates = [
            Path("spider/evaluation_examples/examples/train_spider.json"),
            Path("../spider/evaluation_examples/examples/train_spider.json"),
            Path("/content/drive/MyDrive/nlp_project/spider/evaluation_examples/examples/train_spider.json"),
            Path("/content/nlp_project/spider/evaluation_examples/examples/train_spider.json"),
        ]
        spider_path = next((c for c in candidates if c.exists()), candidates[0])

    print(f"Loading base Spider train set from: {spider_path}")
    with open(spider_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        cleaned.append({
            "question": item["question"].strip(),
            "query": item["query"].strip(),
            "db_id": item["db_id"].strip(),
            "source": "spider_base"
        })
    print(f"  Loaded {len(cleaned)} examples from base Spider.")
    return cleaned


def load_spider_syn(cache_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads Spider-Syn (Gan et al., ACL 2021) which replaces schema and
    phrasing mentions with natural synonyms to enforce lexical invariance.
    """
    cache_file = cache_dir / "spider_syn_train.json"
    if not cache_file.exists():
        print("Fetching Spider-Syn dataset from official repository...")
        url = "https://raw.githubusercontent.com/ygan/Spider-Syn/master/Spider-Syn/train_spider.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)

    with open(cache_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = []
    for item in raw:
        q = item.get("SpiderSynQuestion") or item.get("SpiderQuestion") or item.get("question", "")
        cleaned.append({
            "question": q.strip(),
            "query": item["query"].strip(),
            "db_id": item["db_id"].strip(),
            "source": "spider_syn"
        })
    print(f"  Loaded {len(cleaned)} examples from Spider-Syn.")
    return cleaned


def load_sparc_train() -> List[Dict[str, Any]]:
    """
    Loads SParC (Semantic Parsing in Context, Yale LILY Lab).
    Uses the exact same 200 databases as Spider for compositional queries.
    """
    print("Loading SParC dataset from Hugging Face (jellyChiru/SParC)...")
    try:
        ds = datasets.load_dataset("jellyChiru/SParC", split="train")
        cleaned = []
        for item in ds:
            q = item.get("question", "").strip()
            sql = item.get("query", "").strip().rstrip(";")
            db = item.get("database_id", "").strip()
            if q and sql and db:
                cleaned.append({
                    "question": q,
                    "query": sql,
                    "db_id": db,
                    "source": "sparc_contextual"
                })
        print(f"  Loaded {len(cleaned)} examples from SParC.")
        return cleaned
    except Exception as e:
        print(f"  Warning: Could not load SParC via Hugging Face ({e}). Skipping.")
        return []


def generate_polarity_inversions(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Synthesizes counterfactual polarity-inverted pairs to eliminate antonym collapse.
    Flips:
      - '>' <-> '<'
      - '>=' <-> '<='
      - 'greater than' <-> 'less than'
      - 'older than' <-> 'younger than'
      - 'more than' <-> 'fewer than'
      - 'above' <-> 'below'
    """
    inverted_pairs = []
    nl_pairs = [
        (r'\bgreater than or equal to\b', '__LT_EQ__', r'\bless than or equal to\b', 'greater than or equal to', 'less than or equal to'),
        (r'\bgreater than\b', '__LT__', r'\bless than\b', 'greater than', 'less than'),
        (r'\bmore than\b', '__FEWER__', r'\bfewer than\b', 'more than', 'fewer than'),
        (r'\bolder than\b', '__YOUNGER__', r'\byounger than\b', 'older than', 'younger than'),
        (r'\babove\b', '__BELOW__', r'\bbelow\b', 'above', 'below'),
        (r'\bhigher than\b', '__LOWER__', r'\blower than\b', 'higher than', 'lower than'),
    ]

    for ex in examples:
        q = ex["question"]
        sql = ex["query"]
        db = ex["db_id"]

        flipped_q = q
        flipped_sql = None

        # Check SQL comparators
        has_gt = " > " in sql or " >= " in sql
        has_lt = " < " in sql or " <= " in sql

        if has_gt and not has_lt:
            # Flip SQL > to <
            flipped_sql = sql.replace(" >= ", " __LTE__ ").replace(" > ", " < ").replace(" __LTE__ ", " <= ")
            # Flip NL
            for gt_pat, placeholder, lt_pat, gt_word, lt_word in nl_pairs:
                flipped_q = re.sub(gt_pat, lt_word, flipped_q, flags=re.IGNORECASE)
        elif has_lt and not has_gt:
            # Flip SQL < to >
            flipped_sql = sql.replace(" <= ", " __GTE__ ").replace(" < ", " > ").replace(" __GTE__ ", " >= ")
            # Flip NL
            for gt_pat, placeholder, lt_pat, gt_word, lt_word in nl_pairs:
                flipped_q = re.sub(lt_pat, gt_word, flipped_q, flags=re.IGNORECASE)

        if flipped_sql and flipped_sql != sql and flipped_q != q:
            inverted_pairs.append({
                "question": flipped_q,
                "query": flipped_sql,
                "db_id": db,
                "source": "counterfactual_polarity"
            })

    print(f"  Synthesized {len(inverted_pairs)} counterfactual polarity-balanced examples.")
    return inverted_pairs


def serialize_schema(db_id: str, tables_data: List[Dict[str, Any]]) -> str:
    """Serializes database schema into standard compact sequence format."""
    entry = next((d for d in tables_data if d["db_id"] == db_id), None)
    if not entry:
        return ""
    table_names = entry["table_names_original"]
    table_to_cols = {t: [] for t in table_names}
    for t_id, c_name in entry["column_names_original"]:
        if t_id >= 0 and c_name != "*":
            table_to_cols[table_names[t_id]].append(c_name)
    parts = [f"{tbl} ( {', '.join(cols)} )" for tbl, cols in table_to_cols.items()]
    return " | ".join(parts)


def build_unified_dataset(
    output_dir: Path = Path("data/unified"),
    include_syn: bool = True,
    include_sparc: bool = True,
    include_counterfactual: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Constructs the unified Text-to-SQL corpus, deduplicates, serializes schemas,
    and produces distribution metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tables schema catalog
    tables_candidates = [
        Path("spider/evaluation_examples/examples/tables.json"),
        Path("../spider/evaluation_examples/examples/tables.json"),
    ]
    tables_path = next((c for c in tables_candidates if c.exists()), tables_candidates[0])
    with open(tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # 2. Gather datasets
    corpus: List[Dict[str, Any]] = []

    # Base Spider
    spider_base = load_local_spider_train()
    corpus.extend(spider_base)

    # Spider-Syn
    if include_syn:
        spider_syn = load_spider_syn(cache_dir)
        corpus.extend(spider_syn)

    # SParC
    if include_sparc:
        sparc_data = load_sparc_train()
        corpus.extend(sparc_data)

    # Counterfactual Inversions
    if include_counterfactual:
        cf_pairs = generate_polarity_inversions(spider_base)
        corpus.extend(cf_pairs)

    print(f"\nTotal raw examples collected: {len(corpus)}")

    # 3. Deduplicate by (question, db_id)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in corpus:
        key = (item["question"].strip().lower(), item["db_id"].strip().lower())
        if key not in seen:
            seen.add(key)
            # Add formatted Seq2Seq prompt
            schema_str = serialize_schema(item["db_id"], tables_data)
            prompt = f"Translate to SQL: {item['question']} | Database: {item['db_id']} | Tables: {schema_str}"
            item["prompt"] = prompt
            deduped.append(item)

    print(f"Total deduplicated examples: {len(deduped)}")

    # 4. Compute Statistics
    source_dist = {}
    comparator_counts = {"<": 0, ">": 0, "<=": 0, ">=": 0, "=": 0}
    for item in deduped:
        src = item["source"]
        source_dist[src] = source_dist.get(src, 0) + 1
        q = item["query"]
        if " < " in q:
            comparator_counts["<"] += 1
        if " > " in q:
            comparator_counts[">"] += 1
        if " <= " in q:
            comparator_counts["<="] += 1
        if " >= " in q:
            comparator_counts[">="] += 1
        if " = " in q:
            comparator_counts["="] += 1

    stats = {
        "total_examples": len(deduped),
        "source_distribution": source_dist,
        "comparator_distribution": comparator_counts,
        "database_count": len(tables_data)
    }

    # 5. Save unified dataset
    out_file = output_dir / "train_unified.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)

    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved unified dataset to: {out_file}")
    print(f"Saved dataset statistics to: {stats_file}")
    return deduped, stats


if __name__ == "__main__":
    unified_data, summary_stats = build_unified_dataset()
    print("\nSummary Statistics:")
    print(json.dumps(summary_stats, indent=2))
