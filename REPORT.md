# NeuroSQL: Cross-Domain NLP Text-to-SQL Semantic Parsing on the Spider Benchmark
**Academic Capstone Project & Technical Evaluation Report**

---

## 1. Executive Summary & Abstract

Translating natural language questions into executable Structured Query Language (SQL) across unseen relational database schemas is a foundational challenge in Natural Language Processing (NLP). This project investigates and systematically resolves the core bottlenecks of cross-domain Text-to-SQL semantic parsing on the complex **Yale Spider Benchmark** (comprising 10,181 natural language questions and 5,693 complex SQL queries spanning 200 databases).

We designed, implemented, and empirically benchmarked four progressive paradigms:
1. **Rule-Based Slot Baseline**: Handcrafted pattern matching and regular expression slots (**1.0%** overall AST match).
2. **Lexical Information Retrieval (TF-IDF)**: Word-overlap schema matching coupled with flat clause templates (**4.0%** overall AST match).
3. **Modular Neural-Symbolic Hybrid IR**: Contextual dense embeddings (`all-MiniLM-L6-v2`) with lexical boosting, combined with a **Breadth-First Search (BFS) Steiner Join Planner** over database foreign-key multigraphs (**9.0%** overall AST match, **66.7%** on Easy queries).
4. **End-to-End Generative Seq2Seq Transformer**: Instruction-fine-tuned `google/flan-t5-base` using sequence-serialized schema prompts in FP32 precision, achieving a dramatic **40.0% overall AST Exact Match** on held-out unseen development databases, including **100.0% on Easy queries**, **40.0% on Medium queries**, and **33.3% on Hard queries**.

Finally, we packaged the entire pipeline into a clean modular Python library ([`src/`](file:///c:/Users/ayush/Downloads/project/nLP/src)) and built an interactive visual studio ([`app.py`](file:///c:/Users/ayush/Downloads/project/nLP/app.py)) featuring side-by-side architecture comparison and live SQLite query execution.

---

## 2. Benchmark Progression & Results Matrix

### 2.1 Master Progression Across Architecture Milestones

| Architecture | Paradigm | Easy Match | Medium Match | Hard Match | Overall AST Match | Schema Recall@5 | Primary Bottleneck / Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Rule-Based Baseline** | Regex Slots | 12.5% | 0.0% | 0.0% | **1.0%** | 31.1% | Brittle grammar collisions on unseen phrasing |
| **2. TF-IDF + IR Compiler** | Lexical + IR | 33.3% | 0.0% | 0.0% | **4.0%** | 66.7% | Vocabulary mismatch & dropped synonyms |
| **3. Neural-Symbolic Hybrid** | Dense MiniLM + BFS Graph | 66.7% | 2.0% | 0.0% | **9.0%** | 80.4% | Single-table recall ceiling; unable to synthesize recursive subqueries |
| **4. Fine-Tuned Flan-T5** | **Generative Seq2Seq** | **100.0%** | **40.0%** | **33.3%** | **40.0%** | **N/A (End-to-End)** | **4.4x overall leap; solves table aliases, joins, and nested subqueries** |

---

### 2.2 Fine-Tuned Flan-T5 Official Spider AST Evaluation

Evaluated on 100 randomly sampled held-out queries from [`spider/evaluation_examples/examples/dev.json`](file:///c:/Users/ayush/Downloads/project/nLP/spider/evaluation_examples/examples/dev.json) across unseen schemas using the official Spider AST parse tree evaluator ([`spider/evaluation.py`](file:///c:/Users/ayush/Downloads/project/nLP/spider/evaluation.py)):

| Difficulty Tier | Correct Queries | Total Evaluated | AST Exact Match Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Easy** | **12** | **12** | **100.00%** |
| **Medium** | **20** | **50** | **40.00%** |
| **Hard** | **7** | **21** | **33.33%** |
| **Extra Hard** | **1** | **17** | **5.88%** |
| **OVERALL** | **40** | **100** | **40.00%** |

---

## 3. Detailed Architectural Evolution

### Phase 1: Rule-Based Baseline (Day 3)
- **Concept**: Regex matching on tokens like `"how many"` $\rightarrow$ `COUNT(*)`, `"older than X"` $\rightarrow$ `age > X`.
- **Limitation**: Any permutation in word order or unexpected preposition broke the slot extractor. Yielded **1.0%** overall AST match.

### Phase 2: Lexical Information Retrieval (Day 5)
- **Concept**: Schema matching via token overlap (TF-IDF vector cosine similarity).
- **Limitation**: Suffered from synonym failure (e.g. failing to associate *"citizens"* with `population` or *"municipality"* with `city`). Achieved **4.0%** overall AST match.

### Phase 3: Modular Neural-Symbolic Hybrid IR Pipeline (Days 7–8)
To address the shortcomings of phases 1 and 2, we engineered a modular neural-symbolic pipeline:
1. **Question Head/Tail Splitter**: Isolates projection intents (head) from filter conditions (tail).
2. **Contextual Hybrid Schema Linker**: Embeds schema elements using `all-MiniLM-L6-v2` dense vectors enriched with lexical overlap boosts:
   $$\text{Score}(q, col) = \cos(E_{txt}, E_{col}) + 0.25 \cdot \frac{|W_{col} \cap W_{txt}|}{|W_{col}|} + 0.15 \cdot \mathbb{I}(col \in txt)$$
3. **Foreign-Key Multigraph & BFS Steiner Join Planner**: Models the database schema as a multigraph $G = (V, E)$ where vertices are tables and edges are foreign keys. Uses Breadth-First Search (BFS) to compute the minimal foreign-key Steiner path connecting all candidate tables without hallucinating unused tables.
4. **Intermediate Representation (IR) Builder**: Formulates a structured dictionary containing `main_table`, `select`, `aggregations`, `joins`, `where`, `group_by`, `order_by`, and `limit`.
5. **Deterministic SQL Compiler**: Synthesizes clean standard SQL from the IR.

#### Schema Linking Ablation Study (Recall@K)
| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Lexical Only** | 35.2% | 54.1% | 66.7% | 78.2% |
| **Dense Only (MiniLM)** | 44.8% | 64.3% | 75.6% | 88.9% |
| **Hybrid (Dense + Lexical)** | **52.6%** | **71.9%** | **80.4%** | **95.4%** |

#### Why Symbolic IR Hit a 9% Ceiling
As documented in our error taxonomy:
- **50% Top-1 Schema Misses**: Schema ambiguity when multiple tables contain related columns.
- **22% Join Disconnections**: Spider schemas frequently omit explicit foreign-key constraints in metadata, disconnecting BFS paths.
- **19% Subquery & Set Operations**: Handcrafted IR slots cannot represent recursive structures like:
  ```sql
  SELECT name FROM singer WHERE age = (SELECT min(age) FROM singer)
  ```
  or set operations (`EXCEPT`, `INTERSECT`).

---

### Phase 4: Generative Seq2Seq Transformer (Day 9)
To transcend the structural limitations of flat IR representations, we adapted **`google/flan-t5-base`** (247M parameters) for end-to-end Text-to-SQL generation.

#### Input Serialization Scheme
```text
Translate to SQL: {question} | Database: {db_id} | Tables: {table_1} ( col_1, col_2 ) | {table_2} ( col_3, col_4 )
```

#### Training Configuration & Critical FP32 Precision Fix
- **Model**: `google/flan-t5-base`
- **Optimizer**: AdamW ($\text{lr} = 5 \times 10^{-5}$)
- **Batch Size**: Effective batch size of 16 (batch size 8 with 2 gradient accumulation steps)
- **Epochs**: 4 (1,752 optimization steps)
- **Critical Fix**: Standard HuggingFace recipes often enable `fp16=True`. In T5 architectures, attention logit scaling without scale clamping causes float16 values to exceed $65,504$, resulting in `eval_loss: nan` and corrupted model weights. Setting **`fp16=False`** (FP32) with `max_grad_norm=1.0` restored stable convergence and allowed the model to learn complex SQL syntax.

---

## 4. Repository Structure & Modular Library

```
nLP/
├── Day3.ipynb                 # Full 111-cell experimental notebook (Days 1–9)
├── app.py                     # Streamlit Interactive Web Application Studio
├── requirements.txt           # Environment dependencies
├── REPORT.md                  # Comprehensive academic capstone report
│
├── src/                       # Production-ready modular Python library
│   ├── __init__.py            # Library package root
│   ├── schema_graph.py        # FK multigraph & BFS join planner
│   ├── schema_linking.py      # Dense MiniLM + lexical hybrid linker
│   ├── ir_builder.py          # Intent parser & advanced IR builder
│   ├── sql_generator.py       # Deterministic SQL synthesizer
│   ├── transformer_inference.py # Flan-T5 prompt serialization & beam search decoder
│   ├── evaluator.py           # Spider AST match evaluation harness
│   └── mock_db.py             # In-memory SQLite execution engine
│
└── spider/                    # Official Spider benchmark dataset & evaluators
    ├── process_sql.py         # Official SQL AST tokenizer and parser
    ├── evaluation.py          # Official Spider AST equivalence evaluator
    └── evaluation_examples/   # dev.json, train_spider.json, tables.json
```

---

## 5. How to Run the Project

### 5.1 Interactive Web Studio
To launch the modern Streamlit capstone application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to:
- Test queries across `concert_singer`, `pets_1`, and `car_1`.
- View side-by-side outputs from the **Modular Neural-Symbolic IR** and **Fine-Tuned Flan-T5**.
- Inspect live query execution tables running directly on in-memory SQLite.
- Review interactive benchmark analytics tabs.

### 5.2 Standalone Python Library Usage
```python
from src.schema_graph import plan_table_joins
from src.sql_generator import compile_advanced_sql
from src.mock_db import execute_sql

# Execute SQL and fetch formatted rows
headers, rows, error = execute_sql("concert_singer", "SELECT Name, Country, Age FROM singer WHERE Age > 25")
print(headers)
print(rows)
```

---

## 6. Conclusion & Key Takeaways

1. **Symbolic vs. Generative Paradigms**: While modular symbolic IR architectures provide high interpretability and strong performance on simple schemas (**66.7% on Easy queries**), they hit an intractable ceiling on complex relational semantics (**2% on Medium**, **0% on Hard**).
2. **End-to-End Sequence Modeling**: By framing schema-grounded text-to-SQL as an instruction-tuned Seq2Seq generation task, `flan-t5-base` learned relational graph traversals and nested syntax end-to-end, unlocking a **4.4x overall improvement (40.0% AST match)** and achieving **100.0% accuracy on Easy queries**.
3. **Engineering Stability**: Identifying and resolving the Float16 NaN overflow in T5 layer-normalization proved crucial for achieving production-grade weights capable of generating multi-hop joins and nested SQL subqueries.
