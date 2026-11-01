"""
RAG (Retrieval-Augmented Generation) & Dense Embedding Pipeline for Text-to-SQL.
Combines:
1. Structure-Aware Exemplar RAG: Dense semantic search over 6,962 Spider training demonstrations via all-MiniLM-L6-v2,
   re-ranked using syntactic and structural intent heuristics with diversity filtering.
2. Dense Schema Linking & Sub-Graph Pruning: BFS Steiner tree foreign-key expansion to eliminate distractor tables.
3. RAG Prompt Builder: Dynamically formats relational schema representations with optional few-shot demonstrations.
"""

import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import torch
from sentence_transformers import SentenceTransformer

from src.schema_graph import build_foreign_key_graph, find_shortest_join_path


_GLOBAL_EMBEDDER: Optional[SentenceTransformer] = None


def get_sentence_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Returns a singleton cached SentenceTransformer instance."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = SentenceTransformer(model_name)
    return _GLOBAL_EMBEDDER


class ExemplarRetriever:
    """
    Structure-Aware Dense semantic retriever for in-context SQL generation exemplars.
    Searches pre-computed 384-dimensional embeddings of 6,962 Spider training queries
    and applies syntactic re-ranking for joins, aggregations, and filters.
    """

    def __init__(self, index_path: str = "data/rag/exemplar_index.pt", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.embedder = get_sentence_embedder(model_name)
        self.index_data: Optional[Dict[str, Any]] = None
        self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            try:
                self.index_data = torch.load(str(self.index_path), map_location="cpu")
            except Exception:
                self.index_data = None
        else:
            self.index_data = None

    def is_ready(self) -> bool:
        return self.index_data is not None and "embeddings" in self.index_data

    def retrieve_exemplars(
        self,
        query: str,
        k: int = 2,
        min_score: float = 0.35,
        preferred_db_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k structurally & semantically relevant question-SQL pairs.
        Uses cosine similarity + structural alignment bonuses + diversity filtering.
        """
        if not self.is_ready():
            return []

        q_lower = query.lower()
        q_emb = self.embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = torch.mv(self.index_data["embeddings"], q_emb)

        top_candidates = torch.topk(scores, min(40, len(scores)))

        # Intent detection
        has_join_intent = any(w in q_lower for w in [
            'who', 'performed', 'whose', 'which', 'with', 'from each', 'belonging', 'assigned'
        ]) or (' and ' in q_lower and any(w in q_lower for w in ['in the', 'at the', 'for the']))

        has_agg_count = any(w in q_lower for w in ['how many', 'number of', 'count', 'total number'])
        has_agg_avg = any(w in q_lower for w in ['average', 'avg', 'mean'])
        has_agg_extreme = any(w in q_lower for w in [
            'highest', 'maximum', 'max', 'lowest', 'minimum', 'min', 'most', 'least', 'oldest', 'youngest'
        ])
        has_multi_filter = ' and ' in q_lower or ' or ' in q_lower

        ranked = []
        for s, idx in zip(top_candidates.values, top_candidates.indices):
            i = idx.item()
            base_s = float(s.item())
            if base_s < min_score:
                continue

            sql = self.index_data["sqls"][i]
            cand_q = self.index_data["questions"][i]
            sql_up = sql.upper()

            adj_s = base_s

            # Structural alignment bonuses
            if has_join_intent and 'JOIN' in sql_up:
                adj_s += 0.12
            if has_agg_count and 'COUNT(' in sql_up:
                adj_s += 0.10
            if has_agg_avg and 'AVG(' in sql_up:
                adj_s += 0.10
            if has_agg_extreme and ('ORDER BY' in sql_up or 'MAX(' in sql_up or 'MIN(' in sql_up):
                adj_s += 0.10
            if has_multi_filter and (' AND ' in sql_up or ' OR ' in sql_up):
                adj_s += 0.08
            if preferred_db_id and self.index_data["db_ids"][i] == preferred_db_id:
                adj_s += 0.05

            ranked.append({
                "score": round(adj_s, 4),
                "base_score": round(base_s, 4),
                "question": cand_q,
                "sql": sql,
                "db_id": self.index_data["db_ids"][i]
            })

        # Sort by adjusted score
        ranked.sort(key=lambda x: x["score"], reverse=True)

        # Diversity selection: avoid returning near-duplicate queries or identical SQLs
        selected = []
        for item in ranked:
            is_dup = False
            for prev in selected:
                if item["sql"].strip().lower() == prev["sql"].strip().lower():
                    is_dup = True
                    break
                if difflib.SequenceMatcher(None, item["question"].lower(), prev["question"].lower()).ratio() >= 0.82:
                    is_dup = True
                    break
            if not is_dup:
                selected.append(item)
            if len(selected) >= k:
                break

        return selected


class DenseSchemaPruner:
    """
    Dense Schema Linking and Sub-Graph Pruning engine.
    Prunes distractor tables from database schemas using question-schema embedding similarity
    and automatically restores required intermediate bridge tables via BFS Foreign-Key traversal.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = get_sentence_embedder(model_name)

    def prune_schema(
        self,
        db_id: str,
        tables_list: List[Dict[str, Any]],
        question: str,
        max_tables: int = 4
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prunes tables_list for db_id down to the minimal Steiner sub-graph required for the question.
        Returns:
            (pruned_tables_list, pruning_stats)
        """
        entry = next((d for d in tables_list if d.get("db_id") == db_id), None)
        if not entry:
            return tables_list, {"status": "no_schema_found"}

        table_names = entry.get("table_names_original", [])
        if len(table_names) <= 2:
            return [entry], {
                "total_tables": len(table_names),
                "kept_tables": table_names,
                "pruned_tables": [],
                "bridging_tables": []
            }

        table_to_cols: Dict[str, List[str]] = {t: [] for t in table_names}
        pk_cols = set(entry.get("primary_keys", []))

        for idx, (t_id, c_name) in enumerate(entry.get("column_names_original", [])):
            if t_id >= 0 and c_name != "*":
                if t_id < len(table_names):
                    table_to_cols[table_names[t_id]].append(c_name)

        q_lower = question.lower()
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_]+\b', q_lower))

        # Check for literal cell matches in mock database if available
        matched_tables_from_literals = set()
        try:
            from src.mock_db import DB_CONNECTIONS
            conn = DB_CONNECTIONS.get(db_id)
            if conn:
                cur = conn.cursor()
                literal_candidates = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
                flat_lits = [c[0] or c[1] for c in literal_candidates]
                for lit in flat_lits:
                    lit_low = lit.lower()
                    for tbl in table_names:
                        cur.execute(f"PRAGMA table_info({tbl})")
                        for c_info in cur.fetchall():
                            col = c_info[1]
                            try:
                                cur.execute(f"SELECT 1 FROM {tbl} WHERE LOWER({col}) = LOWER(?) LIMIT 1", (lit_low,))
                                if cur.fetchone():
                                    matched_tables_from_literals.add(tbl)
                                    break
                            except Exception:
                                pass
        except Exception:
            pass

        # Compute semantic score for each table
        table_signatures = [
            f"Table {t} with columns: {', '.join(table_to_cols[t])}"
            for t in table_names
        ]

        q_emb = self.embedder.encode(question, convert_to_tensor=True, normalize_embeddings=True)
        table_embs = self.embedder.encode(table_signatures, convert_to_tensor=True, normalize_embeddings=True)
        similarities = torch.mv(table_embs, q_emb).tolist()

        table_scores = {}
        for idx, tbl in enumerate(table_names):
            score = float(similarities[idx])
            tbl_low = tbl.lower()

            # Boost if table name or singular/plural appears in question
            if tbl_low in q_lower or (tbl_low.endswith('s') and tbl_low[:-1] in q_lower):
                score += 0.40

            # Boost if column names appear in question tokens
            col_matches = sum(1 for c in table_to_cols[tbl] if c.lower() in q_tokens)
            if col_matches > 0:
                score += min(0.35, col_matches * 0.12)

            # High boost if database cell literal matched in this table
            if tbl in matched_tables_from_literals:
                score += 0.50

            table_scores[tbl] = score

        # Rank tables by score
        ranked_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        selected_seed_tables = {ranked_tables[0][0]}

        for tbl, sc in ranked_tables[1:]:
            if sc >= 0.45 and len(selected_seed_tables) < max_tables:
                selected_seed_tables.add(tbl)
            elif tbl in matched_tables_from_literals:
                selected_seed_tables.add(tbl)

        # Connect candidate tables via BFS shortest Foreign-Key path
        fk_graph = build_foreign_key_graph(entry)
        active_tables = set(selected_seed_tables)
        bridging_tables = set()

        seed_list = list(selected_seed_tables)
        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                t1, t2 = seed_list[i], seed_list[j]
                path = find_shortest_join_path(fk_graph, t1, t2)
                if path:
                    for from_t, _, to_t, _ in path:
                        if to_t not in active_tables:
                            active_tables.add(to_t)
                            bridging_tables.add(to_t)
                        if from_t not in active_tables:
                            active_tables.add(from_t)
                            bridging_tables.add(from_t)

        pruned_tables = [t for t in table_names if t not in active_tables]

        # Build clean pruned schema entry
        kept_tables_order = [t for t in table_names if t in active_tables]
        old_to_new_tbl_idx = {t: new_idx for new_idx, t in enumerate(kept_tables_order)}

        # Re-index columns
        new_column_names = [[-1, "*"]]
        new_column_names_orig = [[-1, "*"]]
        new_column_types = ["text"]
        new_primary_keys = []
        old_to_new_col_idx = {0: 0}

        for old_idx, (old_tid, col_orig) in enumerate(entry.get("column_names_original", [])):
            if old_idx == 0 or old_tid < 0 or old_tid >= len(table_names):
                continue
            old_tname = table_names[old_tid]
            if old_tname in active_tables:
                new_tid = old_to_new_tbl_idx[old_tname]
                new_cidx = len(new_column_names_orig)
                old_to_new_col_idx[old_idx] = new_cidx
                new_column_names_orig.append([new_tid, col_orig])
                new_column_names.append([new_tid, entry["column_names"][old_idx][1]])
                new_column_types.append(entry["column_types"][old_idx])
                if old_idx in pk_cols:
                    new_primary_keys.append(new_cidx)

        # Filter and re-index foreign keys
        new_foreign_keys = []
        for a, b in entry.get("foreign_keys", []):
            if a in old_to_new_col_idx and b in old_to_new_col_idx:
                new_foreign_keys.append([old_to_new_col_idx[a], old_to_new_col_idx[b]])

        pruned_entry = {
            "db_id": db_id,
            "table_names": [t.lower() for t in kept_tables_order],
            "table_names_original": kept_tables_order,
            "column_names": new_column_names,
            "column_names_original": new_column_names_orig,
            "column_types": new_column_types,
            "primary_keys": new_primary_keys,
            "foreign_keys": new_foreign_keys
        }

        stats = {
            "total_tables": len(table_names),
            "kept_tables": kept_tables_order,
            "pruned_tables": pruned_tables,
            "bridging_tables": list(bridging_tables),
            "table_scores": {k: round(v, 3) for k, v in table_scores.items()}
        }

        return [pruned_entry], stats


class RAGPipeline:
    """
    End-to-end RAG orchestrator integrating Exemplar Retrieval,
    Dense Schema Pruning, and Prompt Synthesis.
    """

    def __init__(
        self,
        index_path: str = "data/rag/exemplar_index.pt",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.retriever = ExemplarRetriever(index_path=index_path, model_name=model_name)
        self.pruner = DenseSchemaPruner(model_name=model_name)

    def prepare_rag_context(
        self,
        question: str,
        db_id: str,
        tables_list: List[Dict[str, Any]],
        num_exemplars: int = 2,
        prune_schema: bool = True,
        include_exemplars_in_prompt: bool = False
    ) -> Dict[str, Any]:
        """
        Gathers all RAG context for a user question:
        - Retrieves top-k demonstrations
        - Prunes distractor tables from schema
        - include_exemplars_in_prompt: If True, prepends few-shot examples to prompt.
          If False (recommended for fine-tuned Seq2Seq), preserves canonical input format
          to prevent token interference/copy-bias while keeping exemplars in context for UI display.
        """
        # 1. Retrieve exemplars
        exemplars = self.retriever.retrieve_exemplars(
            query=question,
            k=num_exemplars,
            preferred_db_id=db_id
        )

        # 2. Prune schema
        if prune_schema:
            pruned_tables_list, stats = self.pruner.prune_schema(db_id, tables_list, question)
        else:
            pruned_tables_list = tables_list
            entry = next((d for d in tables_list if d.get("db_id") == db_id), None)
            all_tbls = entry.get("table_names_original", []) if entry else []
            stats = {"total_tables": len(all_tbls), "kept_tables": all_tbls, "pruned_tables": [], "bridging_tables": []}

        # 3. Format serialized schema
        from src.transformer_inference import serialize_schema_for_prompt, canonicalize_comparatives
        canonical_q = canonicalize_comparatives(question)
        schema_str = serialize_schema_for_prompt(db_id, pruned_tables_list, include_foreign_keys=True)

        target_part = f"Translate to SQL: {canonical_q} | Database: {db_id} | Tables: {schema_str}"

        # 4. Build prompt
        if include_exemplars_in_prompt and exemplars:
            prompt_parts = []
            for i, ex in enumerate(exemplars, 1):
                prompt_parts.append(f"Example {i}: Question: {ex['question']} -> SQL: {ex['sql']}")
            full_prompt = " | ".join(prompt_parts) + " | " + target_part
        else:
            full_prompt = target_part

        return {
            "full_prompt": full_prompt,
            "target_prompt": target_part,
            "schema_str": schema_str,
            "pruned_tables_list": pruned_tables_list,
            "exemplars": exemplars,
            "pruning_stats": stats
        }


_GLOBAL_RAG_PIPELINE: Optional[RAGPipeline] = None


def get_rag_pipeline(
    index_path: str = "data/rag/exemplar_index.pt",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> RAGPipeline:
    """Returns a singleton instance of RAGPipeline."""
    global _GLOBAL_RAG_PIPELINE
    if _GLOBAL_RAG_PIPELINE is None:
        _GLOBAL_RAG_PIPELINE = RAGPipeline(index_path=index_path, model_name=model_name)
    return _GLOBAL_RAG_PIPELINE
