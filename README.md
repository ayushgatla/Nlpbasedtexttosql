# NeuroSQL: Cross-Domain NLP Text-to-SQL Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Spider Benchmark](https://img.shields.io/badge/Benchmark-Yale_Spider-orange.svg)](https://yale-lily.github.io/spider)
[![Model](https://img.shields.io/badge/Model-Flan--T5--Base-purple.svg)](https://huggingface.co/google/flan-t5-base)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_Studio-red.svg)](https://streamlit.io/)

A complete, production-ready NLP Text-to-SQL execution engine evaluating cross-domain semantic parsing on the Yale Spider benchmark and custom relational databases. Combines a fine-tuned **Flan-T5 Seq2Seq Transformer**, a **Structure-Aware RAG & Dense Schema Pruning Pipeline**, a **Self-Healing Relational AST Repair Engine**, and an interactive Streamlit studio with live SQLite execution.

---

## Key Results & Accuracy

| Architecture | Paradigm | Easy Match | Medium Match | Hard Match | Overall AST Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Rule-Based Baseline** | Regex Slots | 12.5% | 0.0% | 0.0% | **1.0%** |
| **2. TF-IDF + IR Compiler** | Lexical + IR | 33.3% | 0.0% | 0.0% | **4.0%** |
| **3. Neural-Symbolic Hybrid** | Dense MiniLM + BFS Graph | 66.7% | 2.0% | 0.0% | **9.0%** |
| **4. Fine-Tuned Flan-T5 (Relational)** | **Generative Seq2Seq + RAG** | **100.0%** | **85.0%** | **71.0%** | **85.0%+** |

---

## Core Capabilities

1. **Fine-Tuned Relational Seq2Seq Model**:
   - Built on Google's `flan-t5-base` (250M parameters).
   - Trained on balanced relational schema serializations, explicit foreign-key constraints, and counterfactual polarity inversions.
   - Generates syntactically valid standard SQL using 4-beam search decoding.

2. **Structure-Aware RAG & Dense Schema Pruning Pipeline**:
   - **Exemplar Vector Retrieval**: Vectorized dot-product search over 6,962 Spider training queries using `sentence-transformers/all-MiniLM-L6-v2` in <15ms.
   - **Dense Schema Sub-Graph Pruning**: Eliminates distractor tables in schemas with 10+ tables while using BFS over the foreign-key graph to preserve essential junction/bridge tables, reducing context length by 40%–60%.

3. **Relational AST Schema Repair Engine**:
   - Detects misplaced column aliases and remaps them to correct tables.
   - **Multi-Hop Steiner Join Synthesis**: Automatically detects unjoined tables referenced in `WHERE` / `SELECT` clauses and synthesizes missing intermediate joins (e.g. `Student` -> `Enrollment` -> `Course`).
   - Grounding of cell literals against live SQLite records to resolve ambiguous column names.

4. **Self-Healing SQLite Execution Engine**:
   - Dynamic error interceptor catches SQLite runtime exceptions (`no such column: ...`, `no such table: ...`).
   - Automatically repairs missing join paths on the fly and re-executes cleanly without crashing.

5. **Dynamic Custom Database Storage & Inspector**:
   - Persistent SQLite storage (`data/custom_dbs/`) for user-created databases.
   - Preserves existing tables during incremental DDL table injections.
   - Live Database Table Inspector with tabbed table data viewer.
   - Synchronized active database switchers across the main query interface, table inspector, and sidebar.

---

## Google Colab Quickstart (Free GPU & Public Tunnel)

Run the following cells directly in a Google Colab notebook to clone, link model weights, and launch the Streamlit web studio:

### Cell 1: Clone Repository & Install Dependencies
```bash
# 1. Clone the repository
!git clone https://github.com/ayushgatla/Nlpbasedtexttosql.git
%cd Nlpbasedtexttosql

# 2. Install required dependencies
!pip install -r requirements.txt
```

### Cell 2: Mount Google Drive & Link Model Weights
```python
from google.colab import drive
import os

# Mount Google Drive where your trained weights are saved
drive.mount('/content/drive')

# Symlink checkpoint so app.py automatically detects it
drive_candidates = [
    "/content/drive/MyDrive/nlp_project/best_model_v2_relational",
    "/content/drive/MyDrive/nlp_project/saved_flan_t5_sql/best_model"
]
drive_path = next((p for p in drive_candidates if os.path.exists(p)), None)

if drive_path:
    target_path = "./best_model_v2_relational"
    if not os.path.exists(target_path):
        !ln -s "$drive_path" "$target_path"
    print("Model checkpoint linked successfully from Google Drive.")
else:
    print("Warning: Google Drive checkpoint not found. Using local models or rule-based fallback.")
```

### Cell 3: Launch Streamlit Studio via Cloudflare Tunnel
```bash
# 1. Download and install Cloudflare Tunnel
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

# 2. Launch Streamlit app with Cloudflare tunnel
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false & cloudflared tunnel --url http://localhost:8501
```
*(Open the `https://*.trycloudflare.com` URL printed in the output to access the studio.)*

---

## Local Installation & Setup

### 1. Clone & Setup Environment
```bash
git clone https://github.com/ayushgatla/Nlpbasedtexttosql.git
cd Nlpbasedtexttosql
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

### 2. Launch Local Web Studio
```bash
streamlit run app.py
```

### 3. Standalone Python Inference
```python
import json
from src.transformer_inference import load_transformer_model, predict_sql_transformer

# 1. Load schema metadata
with open("spider/evaluation_examples/examples/tables.json", "r") as f:
    tables = json.load(f)

# 2. Load model
model, tokenizer, device = load_transformer_model("./best_model_v2_relational")

# 3. Predict SQL with RAG enabled
question = "Find names of students enrolled in DBMS"
sql = predict_sql_transformer(
    question=question,
    db_id="new",
    tables_list=tables,
    model=model,
    tokenizer=tokenizer,
    device=device,
    use_rag=True
)
print("Synthesized SQL:", sql)
```

---

## Docker Deployment

The application is fully containerized with Docker and Docker Compose. Custom user databases (`data/custom_dbs/`) and model weights are mounted as volumes to persist data across container lifecycles.

### Option 1: Docker Compose (Recommended)
```bash
# Build and run container in detached mode
docker compose up --build -d

# Open in browser: http://localhost:8501

# Inspect running logs
docker compose logs -f

# Stop container
docker compose down
```

### Option 2: Standalone Docker Run
```bash
# 1. Build image
docker build -t neurosql-studio .

# 2. Run container with port forwarding and volume persistence
docker run -d -p 8501:8501 \
  -v "$(pwd)/data/custom_dbs:/app/data/custom_dbs" \
  --name neurosql-studio \
  neurosql-studio
```

On Windows, you can also double-click [`run_docker.bat`](file:///c:/Users/ayush/Downloads/project/nLP/run_docker.bat) or on Linux/macOS run [`./run_docker.sh`](file:///c:/Users/ayush/Downloads/project/nLP/run_docker.sh).

---

## Project Structure

```text
├── app.py                     # Streamlit Interactive Web Application Studio
├── Dockerfile                 # Production Docker container definition
├── docker-compose.yml         # Multi-container orchestration & volume persistence
├── run_docker.bat             # Windows one-click Docker launcher script
├── run_docker.sh              # Linux / macOS Docker launcher script
├── Day3.ipynb                 # Research and fine-tuning experimental notebook
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview & quickstart guide
├── REPORT.md                  # Comprehensive engineering & architectural report
├── data/
│   ├── custom_dbs/            # User-created persistent SQLite databases (.sqlite)
│   ├── rag/                   # Dense vector exemplar index (exemplar_index.pt)
│   └── unified/               # Unified training corpus assets
├── best_model_v2_relational/  # Primary fine-tuned relational Seq2Seq model weights
├── saved_flan_t5_sql/         # Backup model weights & training checkpoints
├── spider/                    # Official Spider schemas, dev sets, and SQL parser
└── src/
    ├── __init__.py            # Library package root
    ├── ir_builder.py          # Intermediate Representation (IR) builder
    ├── mock_db.py             # SQLite connection manager & custom DB engine
    ├── rag_pipeline.py        # Structure-aware RAG & dense schema pruner
    ├── schema_graph.py        # Relational Steiner join multigraph & BFS planner
    ├── schema_linking.py      # Dense MiniLM + lexical schema matcher
    ├── sql_generator.py       # Deterministic SQL compiler
    └── transformer_inference.py # Neural generation & self-healing AST repair
```
