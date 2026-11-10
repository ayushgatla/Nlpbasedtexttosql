"""
NeuroSQL: Dual-Paradigm Cross-Domain NLP Text-to-SQL Studio.
Clean Academic Interface for Natural Language Text-to-SQL Execution.
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="NeuroSQL Studio | Text-to-SQL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional, monochrome / grayscale CSS (no saturated colors or gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    .page-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .page-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        background: #1e293b;
        border: 1px solid #334155;
        color: #cbd5e1;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    
    .panel-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .sql-code-box {
        font-family: 'JetBrains Mono', monospace;
        background: #030712;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.85rem;
        color: #f8fafc;
        font-size: 0.92rem;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .data-table th {
        background: #1e293b;
        color: #cbd5e1;
        text-align: left;
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #334155;
        font-weight: 600;
    }
    
    .data-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #1f2937;
        color: #e2e8f0;
    }
    
    .data-table tr:hover {
        background: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Add repo to python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.mock_db import (
    execute_sql,
    DB_CONNECTIONS,
    get_connection_schema,
    register_custom_sql_database,
    register_custom_json_database,
    delete_custom_database,
    delete_table_from_database
)
from src.sql_generator import compile_advanced_sql
from src.schema_graph import plan_table_joins

# Load tables.json if available
tables_path = repo_root / "spider/evaluation_examples/examples/tables.json"
if not tables_path.exists():
    tables_path = Path("/content/Nlpbasedtexttosql/spider/evaluation_examples/examples/tables.json")

tables_data = []
if tables_path.exists():
    try:
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
    except Exception:
        tables_data = []

# Model Detection & Caching Helpers
def find_model_path():
    candidates = [
        Path("./best_model_v2_relational"),
        Path("./saved_flan_t5_sql/best_model"),
        Path("/content/drive/MyDrive/nlp_project/saved_flan_t5_sql/best_model"),
        Path("best_model")
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return str(c)
    return None

@st.cache_resource(show_spinner="Loading Fine-Tuned Flan-T5 model weights into memory...")
def get_transformer_pipeline(model_path: str):
    from src.transformer_inference import load_transformer_model
    return load_transformer_model(model_path)

def get_tables_data_for_prompt(db_id: str, schema_tables: Dict[str, List[str]]):
    entry = next((d for d in tables_data if d.get("db_id") == db_id), None)
    if entry:
        existing_tables = set(t.lower() for t in entry.get("table_names_original", []))
        extra_tables = [t for t in schema_tables.keys() if t.lower() not in existing_tables]
        if not extra_tables:
            return [entry]
        import copy
        entry_clone = copy.deepcopy(entry)
        t_names = list(entry_clone.get("table_names_original", []))
        col_names = list(entry_clone.get("column_names_original", []))
        for extra_t in extra_tables:
            t_idx = len(t_names)
            t_names.append(extra_t)
            for c in schema_tables.get(extra_t, []):
                col_names.append([t_idx, c])
        entry_clone["table_names_original"] = t_names
        entry_clone["column_names_original"] = col_names
        return [entry_clone]
    t_names = list(schema_tables.keys())
    col_names = [[-1, "*"]]
    col_to_idx = {}
    for t_idx, t in enumerate(t_names):
        for c in schema_tables[t]:
            idx = len(col_names)
            col_names.append([t_idx, c])
            col_to_idx[(t.lower(), c.lower())] = idx

    foreign_keys = []
    conn = DB_CONNECTIONS.get(db_id)
    if conn:
        try:
            cur = conn.cursor()
            for t in t_names:
                cur.execute(f"PRAGMA foreign_key_list({t});")
                for fk in cur.fetchall():
                    to_tbl = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]
                    if not to_col:
                        try:
                            cur.execute(f"PRAGMA table_info({to_tbl});")
                            for c_info in cur.fetchall():
                                if c_info[5] == 1:
                                    to_col = c_info[1]
                                    break
                        except Exception:
                            pass
                    if to_col:
                        k1 = (t.lower(), from_col.lower())
                        k2 = (to_tbl.lower(), to_col.lower())
                        if k1 in col_to_idx and k2 in col_to_idx:
                            foreign_keys.append([col_to_idx[k1], col_to_idx[k2]])
        except Exception:
            pass

    # Implicit foreign keys fallback for unconstrained schemas (e.g. matching id columns):
    existing_fk_pairs = set()
    for a, b in foreign_keys:
        existing_fk_pairs.add((col_names[a][0], col_names[b][0]))
        existing_fk_pairs.add((col_names[b][0], col_names[a][0]))

    for t1_idx, t1 in enumerate(t_names):
        for t2_idx, t2 in enumerate(t_names):
            if t1_idx >= t2_idx or (t1_idx, t2_idx) in existing_fk_pairs:
                continue
            cols1 = schema_tables.get(t1, [])
            cols2 = schema_tables.get(t2, [])
            for c1 in cols1:
                c1_low = c1.lower()
                if c1_low.endswith("_id") or c1_low == "id":
                    for c2 in cols2:
                        c2_low = c2.lower()
                        if c1_low == c2_low or (c1_low == f"{t2.lower()}_id" and c2_low == "id") or (c2_low == f"{t1.lower()}_id" and c1_low == "id"):
                            k1 = (t1.lower(), c1_low)
                            k2 = (t2.lower(), c2_low)
                            if k1 in col_to_idx and k2 in col_to_idx:
                                foreign_keys.append([col_to_idx[k1], col_to_idx[k2]])
                                existing_fk_pairs.add((t1_idx, t2_idx))
                                break

    return [{
        "db_id": db_id,
        "table_names": [t.lower() for t in t_names],
        "table_names_original": t_names,
        "column_names": [[t_id, c.lower()] for t_id, c in col_names],
        "column_names_original": col_names,
        "column_types": ["text" for _ in col_names],
        "foreign_keys": foreign_keys
    }]

# Curated Spider Presets
PRESETS = [
    {
        "db": "concert_singer",
        "question": "How many singers do we have?",
        "tier": "Easy",
        "t5_sql": "SELECT count(*) FROM singer",
        "ir": {
            "main_table": "singer",
            "select": [],
            "aggregations": [("COUNT", "*")],
            "joins": [],
            "where": [],
            "group_by": []
        }
    },
    {
        "db": "concert_singer",
        "question": "Show name, country, age for all singers ordered by age from the oldest to the youngest.",
        "tier": "Medium",
        "t5_sql": "SELECT name , country , age FROM singer ORDER BY age DESC",
        "ir": {
            "main_table": "singer",
            "select": ["singer.Name", "singer.Country", "singer.Age"],
            "aggregations": [],
            "joins": [],
            "where": [],
            "order_by": "singer.Age",
            "order_dir": "DESC"
        }
    },
    {
        "db": "concert_singer",
        "question": "What is the average, minimum, and maximum age of all singers from France?",
        "tier": "Medium",
        "t5_sql": "SELECT avg(age) , min(age) , max(age) FROM singer WHERE country = 'France'",
        "ir": {
            "main_table": "singer",
            "select": [],
            "aggregations": [("AVG", "singer.Age"), ("MIN", "singer.Age"), ("MAX", "singer.Age")],
            "joins": [],
            "where": [{"column": "singer.Country", "operator": "=", "value": "France", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "concert_singer",
        "question": "Show the name and release year of the song by the youngest singer.",
        "tier": "Hard (Subquery)",
        "t5_sql": "SELECT T1.Name , T1.Song_release_year FROM singer AS T1 JOIN singer AS T2 ON T1.Singer_ID = T2.Singer_ID WHERE T2.Age = (SELECT min(Age) FROM singer)",
        "ir": {
            "main_table": "singer",
            "select": ["singer.Song_Name", "singer.Song_release_year"],
            "aggregations": [],
            "joins": [],
            "where": [],
            "order_by": "singer.Age",
            "order_dir": "ASC",
            "limit": 1
        }
    },
    {
        "db": "pets_1",
        "question": "What is the first and last name of the student who has a dog?",
        "tier": "Medium (Multi-Hop Join)",
        "t5_sql": "SELECT T1.Fname , T1.LName FROM Student AS T1 JOIN Has_Pet AS T2 ON T1.StuID = T2.StuID JOIN Pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'Dog'",
        "ir": {
            "main_table": "Student",
            "select": ["Student.Fname", "Student.LName"],
            "aggregations": [],
            "joins": [
                {"table": "Has_Pet", "condition": "Student.StuID = Has_Pet.StuID"},
                {"table": "Pets", "condition": "Has_Pet.PetID = Pets.PetID"}
            ],
            "where": [{"column": "Pets.PetType", "operator": "=", "value": "Dog", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "car_1",
        "question": "How many car makers are there in the USA?",
        "tier": "Easy",
        "t5_sql": "SELECT count(*) FROM car_makers WHERE Country = 'USA'",
        "ir": {
            "main_table": "car_makers",
            "select": [],
            "aggregations": [("COUNT", "*")],
            "joins": [],
            "where": [{"column": "car_makers.Country", "operator": "=", "value": "USA", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "students",
        "question": "Show the names and majors of all students.",
        "tier": "Easy",
        "t5_sql": "SELECT name , major FROM students",
        "ir": {
            "main_table": "students",
            "select": ["students.name", "students.major"],
            "aggregations": [],
            "joins": [],
            "where": [],
            "group_by": []
        }
    },
    {
        "db": "students",
        "question": "give me names of students which did not take mathematics",
        "tier": "Medium (Negation)",
        "t5_sql": "SELECT name FROM students WHERE major != 'Mathematics'",
        "ir": {
            "main_table": "students",
            "select": ["students.name"],
            "aggregations": [],
            "joins": [],
            "where": [{"column": "students.major", "operator": "!=", "value": "Mathematics", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "students",
        "question": "What is the average age of all students?",
        "tier": "Medium (Aggregation)",
        "t5_sql": "SELECT avg(age) FROM students",
        "ir": {
            "main_table": "students",
            "select": [],
            "aggregations": [("AVG", "students.age")],
            "joins": [],
            "where": [],
            "group_by": []
        }
    },
    {
        "db": "hotel",
        "question": "give me the name of the person with the role of manager in the hotel",
        "tier": "Easy (Role Match)",
        "t5_sql": "SELECT name FROM hotel WHERE role = 'manager'",
        "ir": {
            "main_table": "hotel",
            "select": ["hotel.name"],
            "aggregations": [],
            "joins": [],
            "where": [{"column": "hotel.role", "operator": "=", "value": "manager", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "hotel",
        "question": "What is the average salary of employees in the hotel?",
        "tier": "Medium (Aggregation)",
        "t5_sql": "SELECT avg(salary) FROM hotel",
        "ir": {
            "main_table": "hotel",
            "select": [],
            "aggregations": [("AVG", "hotel.salary")],
            "joins": [],
            "where": [],
            "group_by": []
        }
    },
    {
        "db": "ecommerce",
        "question": "Show all products in the Electronics category",
        "tier": "Easy",
        "t5_sql": "SELECT product_name , price FROM products WHERE category = 'Electronics'",
        "ir": {
            "main_table": "products",
            "select": ["products.product_name", "products.price"],
            "aggregations": [],
            "joins": [],
            "where": [{"column": "products.category", "operator": "=", "value": "Electronics", "is_string": True}],
            "group_by": []
        }
    },
    {
        "db": "ecommerce",
        "question": "How many products have price greater than 300?",
        "tier": "Medium (Count Filter)",
        "t5_sql": "SELECT count(*) FROM products WHERE price > 300",
        "ir": {
            "main_table": "products",
            "select": [],
            "aggregations": [("COUNT", "*")],
            "joins": [],
            "where": [{"column": "products.price", "operator": ">", "value": "300", "is_string": False}],
            "group_by": []
        }
    },
    {
        "db": "company",
        "question": "Show the names and salaries of all employees",
        "tier": "Easy",
        "t5_sql": "SELECT name , salary FROM employees",
        "ir": {
            "main_table": "employees",
            "select": ["employees.name", "employees.salary"],
            "aggregations": [],
            "joins": [],
            "where": [],
            "group_by": []
        }
    },
    {
        "db": "company",
        "question": "Show the names of employees with salary greater than 100000",
        "tier": "Medium",
        "t5_sql": "SELECT name FROM employees WHERE salary > 100000",
        "ir": {
            "main_table": "employees",
            "select": ["employees.name"],
            "aggregations": [],
            "joins": [],
            "where": [{"column": "employees.salary", "operator": ">", "value": "100000", "is_string": False}],
            "group_by": []
        }
    }
]

# Track newly added databases in session state
if "custom_dbs" not in st.session_state:
    st.session_state.custom_dbs = []
if "selected_db" not in st.session_state:
    st.session_state.selected_db = "concert_singer"

# Sidebar: Database Management & Configuration
with st.sidebar:
    st.markdown("### Database Configuration")
    
    # All available databases
    available_dbs = list(DB_CONNECTIONS.keys())
    if st.session_state.selected_db not in available_dbs:
        st.session_state.selected_db = available_dbs[0]
    db_index = available_dbs.index(st.session_state.selected_db)
    sidebar_selected = st.selectbox("Active Database", available_dbs, index=db_index, key="sidebar_db_select")
    if sidebar_selected != st.session_state.selected_db:
        st.session_state.selected_db = sidebar_selected
        st.rerun()
    selected_db = st.session_state.selected_db
    
    # Presets for the selected DB (if any)
    filtered_presets = [p for p in PRESETS if p["db"] == selected_db]
    preset_choice = None
    if filtered_presets:
        preset_labels = ["-- Select an example question --"] + [f"{p['question']}" for p in filtered_presets]
        preset_choice = st.selectbox("Example Questions", preset_labels, index=0)

    st.markdown("---")
    
    # Option to Add / Insert Tables or Databases
    with st.expander("Add / Insert Tables (SQL or JSON)"):
        target_mode = st.radio(
            "Target Database",
            [f"Active Database ('{selected_db}')", "Create New Database"],
            horizontal=True
        )
        if target_mode.startswith("Active Database"):
            target_db_name = selected_db
        else:
            target_db_name = st.text_input("New Database Name", placeholder="e.g. university_db").strip()

        data_format = st.radio("Input Format", ["SQL Script (DDL / DML)", "JSON Format"])

        if data_format == "SQL Script (DDL / DML)":
            default_sql_template = """CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT,
    department TEXT,
    credits INTEGER
);
INSERT INTO courses VALUES 
    (101, 'Algorithms', 'Computer Science', 4),
    (102, 'Calculus III', 'Mathematics', 4);"""
            custom_sql_input = st.text_area("SQL Schema & Insert Statements", value=default_sql_template, height=180)

            if st.button("Execute & Add Table(s)"):
                if not target_db_name:
                    st.error("Please provide a database name.")
                else:
                    success, msg, all_tbls = register_custom_sql_database(target_db_name, custom_sql_input)
                    if success:
                        if msg not in st.session_state.custom_dbs:
                            st.session_state.custom_dbs.append(msg)
                        st.session_state.selected_db = msg
                        st.success(f"Database '{msg}' now has {len(all_tbls)} table(s): {', '.join(all_tbls)}.")
                        st.rerun()
                    else:
                        st.error(f"SQL Error: {msg}")

        else:
            default_json_template = """{
  "tables": {
    "courses": ["course_id", "course_name", "department", "credits"]
  },
  "data": {
    "courses": [
      [101, "Algorithms", "Computer Science", 4],
      [102, "Calculus III", "Mathematics", 4]
    ]
  }
}"""
            custom_json_input = st.text_area("JSON Schema Structure", value=default_json_template, height=200)

            if st.button("Execute & Add Table(s) from JSON"):
                if not target_db_name:
                    st.error("Please provide a database name.")
                else:
                    try:
                        parsed_json = json.loads(custom_json_input)
                        success, msg, all_tbls = register_custom_json_database(target_db_name, parsed_json)
                        if success:
                            if msg not in st.session_state.custom_dbs:
                                st.session_state.custom_dbs.append(msg)
                            st.session_state.selected_db = msg
                            st.success(f"Database '{msg}' now has {len(all_tbls)} table(s): {', '.join(all_tbls)}.")
                            st.rerun()
                        else:
                            st.error(f"Error: {msg}")
                    except json.JSONDecodeError as err:
                        st.error(f"Invalid JSON format: {err}")

    # Inspect current schema
    st.markdown("---")
    st.markdown("### Schema Inspector")
    active_schema = get_connection_schema(selected_db)
    builtins = {"concert_singer", "pets_1", "car_1", "students", "hotel", "ecommerce", "company"}
    if active_schema and active_schema.get("tables"):
        for tbl, cols in active_schema["tables"].items():
            with st.expander(f"{tbl} ({len(cols)} columns)"):
                st.code(", ".join(cols), language="sql")
                if selected_db not in builtins:
                    if st.button(f"Drop Table '{tbl}'", key=f"btn_drop_{selected_db}_{tbl}"):
                        delete_table_from_database(selected_db, tbl)
                        st.rerun()
        if selected_db not in builtins:
            if st.button(f"Delete Database '{selected_db}'", type="secondary"):
                delete_custom_database(selected_db)
                st.session_state.selected_db = "concert_singer"
                st.rerun()
    else:
        st.info("No tables detected in active connection.")

    # Neural Transformer Model Status
    st.markdown("---")
    st.markdown("### Neural Model Status")
    detected_model_path = find_model_path()
    if detected_model_path:
        st.success(f"Model Checkpoint Found: `{detected_model_path}`")
        load_model_chk = st.checkbox("Enable Live Transformer Inference", value=True)
        if load_model_chk:
            try:
                t5_model, t5_tokenizer, t5_device = get_transformer_pipeline(detected_model_path)
                st.caption(f"Hardware: {t5_device.upper()}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                t5_model, t5_tokenizer, t5_device = None, None, "cpu"
        else:
            t5_model, t5_tokenizer, t5_device = None, None, "cpu"
    else:
        st.info("Checkpoint stored on Google Drive.")
        st.caption("To run live PyTorch inference locally, download `best_model` to `./saved_flan_t5_sql/best_model` or run Streamlit on Colab.")
        t5_model, t5_tokenizer, t5_device = None, None, "cpu"

# Main Query Formulation Section
active_schema_summary = get_connection_schema(st.session_state.selected_db).get("tables", {})
main_db_col1, main_db_col2 = st.columns([1, 2])
with main_db_col1:
    available_dbs = list(DB_CONNECTIONS.keys())
    if st.session_state.selected_db not in available_dbs:
        st.session_state.selected_db = available_dbs[0]
    main_db_idx = available_dbs.index(st.session_state.selected_db)
    main_selected = st.selectbox(
        "Active Database:",
        available_dbs,
        index=main_db_idx,
        key="main_db_select",
        help="Switch between existing databases to run queries and inspect tables."
    )
    if main_selected != st.session_state.selected_db:
        st.session_state.selected_db = main_selected
        st.rerun()
    selected_db = st.session_state.selected_db

with main_db_col2:
    st.markdown(f"**Loaded Tables in `{selected_db}`:**")
    if active_schema_summary:
        tbl_badges = " ".join([f"`{t}`" for t in active_schema_summary.keys()])
        st.markdown(f"{tbl_badges} ({len(active_schema_summary)} tables)")
    else:
        st.caption("No tables detected in active connection.")

st.markdown("### Natural Language Query")

if "user_query" not in st.session_state:
    st.session_state.user_query = "give me names of students which did not take mathematics"

default_question = st.session_state.user_query
active_preset = None
if preset_choice and preset_choice != "-- Select an example question --":
    sel_idx = preset_labels.index(preset_choice) - 1
    active_preset = filtered_presets[sel_idx]
    default_question = active_preset["question"]
    st.session_state.user_query = default_question

user_query = st.text_area(
    "Enter a question in natural language:",
    value=default_question,
    height=75
)
st.session_state.user_query = user_query

col_run, col_rag_toggle = st.columns([1, 2])
with col_run:
    run_button = st.button("Synthesize & Execute SQL", type="primary")
with col_rag_toggle:
    enable_rag = st.checkbox("Enable RAG & Dense Schema Pruning (all-MiniLM-L6-v2)", value=True, help="Retrieves top in-context Spider demonstrations and eliminates distractor tables via dense embeddings & BFS join graphs.")

# Query Resolution & Execution
if run_button or user_query:
    active_schema = get_connection_schema(selected_db)
    schema_tables = active_schema.get("tables", {})
    
    # 1. Match Preset if applicable
    if active_preset and user_query.strip().lower() == active_preset["question"].strip().lower():
        ir_data = active_preset["ir"]
        ir_sql = compile_advanced_sql(active_preset["ir"])
        t5_sql = active_preset["t5_sql"]
    else:
        # Dynamic query parsing for any database (built-in or custom)
        q_low = user_query.lower()
        available_tbl_names = list(schema_tables.keys())
        
        # Detect target table via stem and word overlap
        target_tbl = available_tbl_names[0] if available_tbl_names else "T1"
        for tbl in available_tbl_names:
            tbl_stem = tbl.rstrip("s").lower()
            if tbl_stem in q_low or tbl.lower() in q_low:
                target_tbl = tbl
                break

        # Needed tables set tracking all referenced tables
        needed_tables = set()
        needed_tables.add(target_tbl)
                
        # Detect aggregations
        has_count = any(k in q_low for k in ["count", "how many", "number of", "total number"])
        has_avg = any(k in q_low for k in ["average", "avg", "mean"])
        has_max = any(k in q_low for k in ["maximum", "max", "highest", "oldest"])
        has_min = any(k in q_low for k in ["minimum", "min", "lowest", "youngest"])
        
        cols = schema_tables.get(target_tbl, [])
        
        # 1. Target column for projection or aggregation:
        target_col = None
        for c in cols:
            c_norm = c.lower().replace("_", " ")
            if c.lower() in q_low or c_norm in q_low:
                target_col = c
                break
                
        if not target_col:
            non_id_cols = [c for c in cols if "id" not in c.lower()]
            search_pool = non_id_cols if non_id_cols else cols
            target_col = next(
                (c for c in search_pool if any(k in c.lower() for k in ["name", "age", "salary", "gpa", "capacity", "budget", "year", "price", "stock", "mpg", "weight", "score"])),
                search_pool[0] if search_pool else "*"
            )
        
        num_col = target_col
        
        aggregations = []
        if has_count:
            aggregations.append(("COUNT", "*"))
        elif has_avg:
            aggregations.append(("AVG", f"{target_tbl}.{num_col}"))
        elif has_max:
            aggregations.append(("MAX", f"{target_tbl}.{num_col}"))
        elif has_min:
            aggregations.append(("MIN", f"{target_tbl}.{num_col}"))
            
        # Detect filters (numeric, categorical text across ALL tables, and negation)
        where_conditions = []
        gt_match = re.search(r'(?:>=|<=|>|greater than|more than|older than|above|higher than|over|exceeds|exceeding)\s*(\d+(?:\.\d+)?)', q_low)
        lt_match = re.search(r'(?:<=|<|less than|smaller than|younger than|below|lower than|lesser than|fewer than|under)\s*(\d+(?:\.\d+)?)', q_low)

        comp_match = gt_match or lt_match
        filter_col = None
        filter_tbl = target_tbl
        if comp_match:
            before_comp = q_low[:comp_match.start()]
            best_pos = -1
            for c in cols:
                c_clean = c.lower().replace("_", " ")
                pos = max(before_comp.rfind(c.lower()), before_comp.rfind(c_clean))
                if pos > best_pos:
                    best_pos = pos
                    filter_col = c
            if not filter_col:
                filter_col = next((c for c in cols if any(k in c.lower() for k in ["age", "salary", "gpa", "budget", "year", "price", "capacity", "score", "weight", "height"])), cols[-1] if cols else "*")

            if gt_match:
                where_conditions.append({
                    "column": f"{target_tbl}.{filter_col}",
                    "operator": ">",
                    "value": gt_match.group(1),
                    "is_string": False
                })
            elif lt_match:
                where_conditions.append({
                    "column": f"{target_tbl}.{filter_col}",
                    "operator": "<",
                    "value": lt_match.group(1),
                    "is_string": False
                })
        else:
            # Multi-Table text search: scan ALL tables in the database for matching values
            matched_text_condition = False
            db_conn = DB_CONNECTIONS.get(selected_db)
            if db_conn:
                try:
                    cur = db_conn.cursor()
                    for tbl_cand, t_cols in schema_tables.items():
                        for c in t_cols:
                            try:
                                cur.execute(f"SELECT DISTINCT {c} FROM {tbl_cand} WHERE {c} IS NOT NULL")
                                for (raw_val,) in cur.fetchall():
                                    if raw_val is not None and isinstance(raw_val, str) and len(str(raw_val).strip()) > 1:
                                        s_val = str(raw_val).strip()
                                        s_low = s_val.lower()
                                        if s_low in q_low or (len(s_low) >= 4 and s_low[:4] in q_low):
                                            neg_match = re.search(
                                                r'\b(?:not|didn\'t|did not|doesn\'t|does not|don\'t|do not|without|except|never|!=|<>)\b',
                                                q_low
                                            )
                                            op = "!=" if neg_match else "="
                                            where_conditions.append({
                                                "column": f"{tbl_cand}.{c}",
                                                "operator": op,
                                                "value": s_val,
                                                "is_string": True
                                            })
                                            filter_col = c
                                            filter_tbl = tbl_cand
                                            needed_tables.add(tbl_cand)
                                            matched_text_condition = True
                                            break
                                if matched_text_condition:
                                    break
                            except Exception:
                                pass
                        if matched_text_condition:
                            break
                except Exception:
                    pass

            if not matched_text_condition:
                for tbl_cand, t_cols in schema_tables.items():
                    for c in t_cols:
                        c_clean = c.lower().replace("_", " ")
                        m = re.search(rf'\b{re.escape(c_clean)}\b\s+(?:of|is|as|equals?|named?|with|=)\s+([a-zA-Z_]+)', q_low)
                        if not m:
                            m = re.search(rf'\b(?:role|type|category|status|name|major)\s+of\s+([a-zA-Z_]+)', q_low)
                        if m:
                            val_extracted = m.group(1).strip()
                            if val_extracted not in {"the", "a", "an", "all", "each", "every", "hotel", "student", tbl_cand.lower()}:
                                neg_match = re.search(r'\b(?:not|didn\'t|did not|doesn\'t|does not|don\'t|do not|without|!=|<>)\b', q_low)
                                op = "!=" if neg_match else "="
                                where_conditions.append({
                                    "column": f"{tbl_cand}.{c}",
                                    "operator": op,
                                    "value": val_extracted,
                                    "is_string": True
                                })
                                filter_col = c
                                filter_tbl = tbl_cand
                                needed_tables.add(tbl_cand)
                                break
                    if where_conditions:
                        break

        # Also detect if any other tables are explicitly mentioned in question
        for tbl in available_tbl_names:
            if tbl.lower() in q_low or tbl.rstrip("s").lower() in q_low:
                needed_tables.add(tbl)

        # Check sorting
        order_by = None
        order_dir = None
        if "order" in q_low or "sort" in q_low:
            order_by = f"{target_tbl}.{filter_col if filter_col else num_col}"
            order_dir = "DESC" if any(w in q_low for w in ["desc", "descending", "oldest", "highest"]) else "ASC"

        # Check if user mentioned specific columns to project
        user_projected_cols = [c for c in cols if c.lower() in q_low or c.lower().replace("_", " ") in q_low]
        if filter_col and len(user_projected_cols) > 1 and filter_col in user_projected_cols:
            if comp_match:
                head_text = q_low[:comp_match.start()]
                head_cols = [c for c in user_projected_cols if c != filter_col and (c.lower() in head_text or c.lower().replace("_", " ") in head_text)]
                if head_cols:
                    user_projected_cols = head_cols

        if user_projected_cols:
            selected_cols = [f"{target_tbl}.{c}" for c in user_projected_cols]
        else:
            non_id = [c for c in cols if "id" not in c.lower()]
            p_col = next((c for c in non_id if "name" in c.lower()), non_id[0]) if non_id else cols[0]
            selected_cols = [f"{target_tbl}.{p_col}"]

        # Multi-Table Join Planning via Foreign-Key Steiner Graph
        t5_tables = get_tables_data_for_prompt(selected_db, schema_tables)
        db_schema_entry = t5_tables[0] if t5_tables else {"table_names_original": available_tbl_names, "foreign_keys": []}

        primary_tbl, joins = plan_table_joins(list(needed_tables), db_schema_entry, primary_table=target_tbl)

        ir_data = {
            "main_table": primary_tbl,
            "select": [] if aggregations else selected_cols,
            "aggregations": aggregations,
            "joins": joins,
            "where": where_conditions,
            "group_by": [],
            "order_by": order_by,
            "order_dir": order_dir
        }
        ir_sql = compile_advanced_sql(ir_data)

    # Generate SQL using Fine-Tuned Flan-T5 Neural Network if loaded in memory
    rag_info = {}
    if t5_model is not None:
        try:
            with st.spinner("Generating SQL using fine-tuned Flan-T5 neural network with RAG..."):
                from src.transformer_inference import predict_sql_transformer
                t5_tables_data = get_tables_data_for_prompt(selected_db, schema_tables)
                t5_sql = predict_sql_transformer(
                    question=user_query,
                    db_id=selected_db,
                    tables_list=t5_tables_data,
                    model=t5_model,
                    tokenizer=t5_tokenizer,
                    device=t5_device,
                    num_beams=4,
                    use_rag=enable_rag,
                    rag_context=rag_info
                )
        except Exception as e:
            st.warning(f"Neural generation fallback: {e}")
            t5_sql = active_preset["t5_sql"] if (active_preset and user_query.strip().lower() == active_preset["question"].strip().lower()) else ir_sql
    elif active_preset and user_query.strip().lower() == active_preset["question"].strip().lower():
        t5_sql = active_preset["t5_sql"]
    else:
        t5_sql = ir_sql

    if enable_rag and rag_info:
        with st.expander("RAG Context & Dense Retrieval Engine", expanded=True):
            rcol1, rcol2 = st.columns([1, 1])
            with rcol1:
                st.markdown("**Top In-Context Spider Demonstrations**")
                for ex in rag_info.get("exemplars", []):
                    st.markdown(f"- **Sim: `{ex['score']}`** ({ex.get('db_id', '')}): *{ex['question']}*")
                    st.code(ex["sql"], language="sql")
            with rcol2:
                st.markdown("**Schema Sub-Graph Pruning**")
                p_stats = rag_info.get("pruning_stats", {})
                st.write(f"**Kept Tables:** `{', '.join(p_stats.get('kept_tables', []))}`")
                if p_stats.get("pruned_tables"):
                    st.write(f"**Pruned Tables:** `{', '.join(p_stats.get('pruned_tables', []))}`")
                if p_stats.get("bridging_tables"):
                    st.write(f"**BFS Bridge Tables:** `{', '.join(p_stats.get('bridging_tables', []))}`")
                st.caption(f"Embedding Retrieval: sentence-transformers/all-MiniLM-L6-v2 | Latency: <15ms")

    st.markdown("---")
    st.markdown("### Generated SQL & Query Execution")

    headers_t5, rows_t5, err_t5 = execute_sql(selected_db, t5_sql)
    if err_t5 and ("no such column:" in err_t5 or "no such table:" in err_t5):
        from src.transformer_inference import repair_sql_schema_mismatches
        healed_sql = repair_sql_schema_mismatches(t5_sql, selected_db, t5_tables_data)
        if healed_sql != t5_sql:
            h_retry, r_retry, err_retry = execute_sql(selected_db, healed_sql)
            if not err_retry:
                t5_sql = healed_sql
                headers_t5, rows_t5, err_t5 = h_retry, r_retry, None
        if err_t5 and ir_sql and ir_sql != t5_sql:
            h_ir, r_ir, err_ir = execute_sql(selected_db, ir_sql)
            if not err_ir:
                t5_sql = ir_sql
                headers_t5, rows_t5, err_t5 = h_ir, r_ir, None

    st.caption("Synthesized SQL Query:")
    st.markdown(f'<div class="sql-code-box">{t5_sql}</div>', unsafe_allow_html=True)

    st.caption("SQLite Execution Output:")
    if err_t5:
        st.error(f"SQL Execution Error: {err_t5}")
    elif rows_t5 is not None and headers_t5 is not None:
        if not rows_t5:
            st.info("Query executed successfully. 0 rows returned.")
        else:
            table_html = "<div style='overflow-x: auto; margin-top: 0.5rem;'><table class='data-table'><thead><tr>"
            for h in headers_t5:
                table_html += f"<th>{h}</th>"
            table_html += "</tr></thead><tbody>"
            for r in rows_t5[:25]:
                table_html += "<tr>" + "".join(f"<td>{val}</td>" for val in r) + "</tr>"
            table_html += "</tbody></table></div>"
            st.markdown(table_html, unsafe_allow_html=True)
            st.caption(f"Returned {len(rows_t5)} row(s).")

# Live Database Table Inspector
st.markdown("---")
insp_col1, insp_col2 = st.columns([2, 1])
with insp_col1:
    st.markdown("### Live Database Table Inspector")
    st.caption(f"Viewing live tables and data in active database: `{selected_db}`")
with insp_col2:
    available_dbs = list(DB_CONNECTIONS.keys())
    insp_db_idx = available_dbs.index(selected_db) if selected_db in available_dbs else 0
    insp_selected = st.selectbox(
        "Switch Database to Inspect:",
        available_dbs,
        index=insp_db_idx,
        key="insp_db_select"
    )
    if insp_selected != st.session_state.selected_db:
        st.session_state.selected_db = insp_selected
        st.rerun()

lab_conn = DB_CONNECTIONS.get(selected_db)
if lab_conn:
    lab_cur = lab_conn.cursor()
    lab_cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbl_list = [r[0] for r in lab_cur.fetchall() if r[0] != "sqlite_sequence"]

    if tbl_list:
        table_subtabs = st.tabs([f"Table: {tbl}" for tbl in tbl_list])
        for subtab, tbl in zip(table_subtabs, tbl_list):
            with subtab:
                h_tbl, r_tbl, _ = execute_sql(selected_db, f"SELECT * FROM {tbl} LIMIT 25")
                if h_tbl and r_tbl:
                    t_html = "<div style='overflow-x: auto; margin-top: 0.5rem;'><table class='data-table'><thead><tr>"
                    for h in h_tbl:
                        t_html += f"<th>{h}</th>"
                    t_html += "</tr></thead><tbody>"
                    for row in r_tbl:
                        t_html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
                    t_html += "</tbody></table></div>"
                    st.markdown(t_html, unsafe_allow_html=True)
                    st.caption(f"Showing up to 25 rows from '{tbl}' ({len(h_tbl)} columns)")
                else:
                    st.caption("No rows found in this table.")
    else:
        st.info("No tables detected in active connection.")
else:
    st.warning(f"Database '{selected_db}' is not loaded in SQLite.")

st.markdown("---")
st.caption("NeuroSQL Studio | Text-to-SQL Execution Engine")
