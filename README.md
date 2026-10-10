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

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/ayushgatla/Nlpbasedtexttosql.git
cd Nlpbasedtexttosql
pip install -r requirements.txt
```

### 2. Launch Interactive Capstone Studio
```bash
streamlit run app.py
```

### 3. Open Full Research Notebook
Open [`Day3.ipynb`](file:///c:/Users/ayush/Downloads/project/nLP/Day3.ipynb) in Jupyter Notebook or Google Colab to step through all experimental phases (Days 1–9).

---

## 📁 Project Architecture

- [`src/`](file:///c:/Users/ayush/Downloads/project/nLP/src): Modular Python package (`schema_graph.py`, `schema_linking.py`, `ir_builder.py`, `sql_generator.py`, `transformer_inference.py`, `evaluator.py`, `mock_db.py`).
- [`app.py`](file:///c:/Users/ayush/Downloads/project/nLP/app.py): Interactive Streamlit studio with dual-model comparison and live SQLite execution.
- [`REPORT.md`](file:///c:/Users/ayush/Downloads/project/nLP/REPORT.md): Full academic report detailing methodology, ablation studies, error taxonomy, and evaluation proofs.
- [`spider/`](file:///c:/Users/ayush/Downloads/project/nLP/spider): Official Spider benchmark evaluation scripts and datasets.
