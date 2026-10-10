# NeuroSQL: Cross-Domain NLP Text-to-SQL Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Spider Benchmark](https://img.shields.io/badge/Benchmark-Yale_Spider-orange.svg)](https://yale-lily.github.io/spider)
[![Model](https://img.shields.io/badge/Model-Flan--T5--Base-purple.svg)](https://huggingface.co/google/flan-t5-base)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_Studio-red.svg)](https://streamlit.io/)

A complete, high-scoring university NLP Text-to-SQL capstone project evaluating cross-domain semantic parsing on the Yale Spider benchmark. Compares **Modular Neural-Symbolic IR** against a **Fine-Tuned Flan-T5 Seq2Seq Transformer** with official AST parsing and live SQLite execution.

---

## 📊 Key Results

| Architecture | Paradigm | Easy Match | Medium Match | Hard Match | Overall AST Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Rule-Based Baseline** | Regex Slots | 12.5% | 0.0% | 0.0% | **1.0%** |
| **2. TF-IDF + IR Compiler** | Lexical + IR | 33.3% | 0.0% | 0.0% | **4.0%** |
| **3. Neural-Symbolic Hybrid** | Dense MiniLM + BFS Graph | 66.7% | 2.0% | 0.0% | **9.0%** |
| **4. Fine-Tuned Flan-T5** | **Generative Seq2Seq** | **100.0%** | **40.0%** | **33.3%** | **40.0%** |

---

## ⚡ Google Colab Quickstart (Run with Free GPU & Public Tunnel)

Run the following cells directly in a Google Colab notebook to clone, load the fine-tuned model checkpoint, and launch the public Streamlit web app:

### Cell 1: Clone Repository & Install Dependencies
```bash
# 1. Clone the project repository
!git clone https://github.com/ayushgatla/Nlpbasedtexttosql.git
%cd Nlpbasedtexttosql

# 2. Install dependencies
!pip install -r requirements.txt
```

### Cell 2: Mount Google Drive & Link Model Weights
```python
from google.colab import drive
import os

# Mount Google Drive where your trained weights are saved
drive.mount('/content/drive')

# Symlink checkpoint so app.py automatically detects it
drive_model_path = "/content/drive/MyDrive/nlp_project/saved_flan_t5_sql/best_model"
local_model_path = "saved_flan_t5_sql/best_model"

if os.path.exists(drive_model_path):
    os.makedirs("saved_flan_t5_sql", exist_ok=True)
    if not os.path.exists(local_model_path):
        !ln -s "$drive_model_path" "$local_model_path"
    print("✅ Model checkpoint linked successfully!")
else:
    print(f"⚠️ Warning: Checkpoint not found at {drive_model_path}. Running with benchmark presets.")
```

### Cell 3: Launch Streamlit Studio via Public Tunnel

#### Option A: Cloudflare Tunnel (Recommended — No Passwords, Fast & Stable)
```bash
# 1. Install Cloudflare tunnel daemon
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

# 2. Launch Streamlit and tunnel
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false & cloudflared tunnel --url http://localhost:8501
```
*(Click the `https://xxxx.trycloudflare.com` URL printed in the output to open the live studio!)*

#### Option B: Localtunnel
```bash
# 1. Check Colab endpoint IP (used as tunnel password)
!curl ipv4.icanhazip.com

# 2. Launch Streamlit with localtunnel
!streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false & npx localtunnel --port 8501
```
*(Open the `*.loca.lt` URL, paste the IP address from step 1 into the "Tunnel Password" input, and click Submit!)*

---

### Cell 4: Standalone Python Inference (In Colab GPU Memory)
```python
import json
import torch
from src.transformer_inference import load_transformer_model, predict_sql_transformer

# 1. Load schemas
with open("spider/evaluation_examples/examples/tables.json", "r") as f:
    tables = json.load(f)

# 2. Load model on GPU
model, tokenizer, device = load_transformer_model("saved_flan_t5_sql/best_model", device="cuda")

# 3. Test question
test_q = "Show the name and release year of the song by the youngest singer."
pred_sql = predict_sql_transformer(test_q, "concert_singer", tables, model, tokenizer, device=device)

print(f"Question:  {test_q}")
print(f"Predicted: {pred_sql}")
```

---

## 💻 Local Installation

### 1. Clone & Setup
```bash
git clone https://github.com/ayushgatla/Nlpbasedtexttosql.git
cd Nlpbasedtexttosql
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Local Web Studio
```bash
streamlit run app.py
```

### 3. Open Research Notebook
Open [`Day3.ipynb`](file:///c:/Users/ayush/Downloads/project/nLP/Day3.ipynb) to inspect the 111-cell experimental research history (Days 1–9).

---

## 📁 Project Architecture

- [`src/`](file:///c:/Users/ayush/Downloads/project/nLP/src): Modular Python package:
  - `schema_graph.py`: FK multigraph & BFS Steiner Tree join planner.
  - `schema_linking.py`: Dense MiniLM + lexical overlap hybrid linker.
  - `ir_builder.py`: Head/tail splitter, multi-condition `WHERE`, `GROUP BY` inference.
  - `sql_generator.py`: Deterministic SQL compiler with clause ordering.
  - `transformer_inference.py`: Prompt serialization & beam search decoder.
  - `evaluator.py`: Official Spider AST parser & evaluation harness.
  - `mock_db.py`: In-memory SQLite execution engine with custom SQL/JSON database registration.
- [`app.py`](file:///c:/Users/ayush/Downloads/project/nLP/app.py): Interactive Streamlit studio with dual-model comparison and live SQLite execution.
- [`REPORT.md`](file:///c:/Users/ayush/Downloads/project/nLP/REPORT.md): Full academic capstone report detailing mathematical formalization, ablation studies, error taxonomy, and evaluation proofs.
- [`spider/`](file:///c:/Users/ayush/Downloads/project/nLP/spider): Official Spider benchmark evaluation scripts and datasets.
