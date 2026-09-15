import os
import sys
import io
import json
import threading
import socketserver
import webbrowser
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import streamlit as st
import streamlit.components.v1 as components

from agent.agent import DataAnalystAgent

# Create Flask App
flask_app = Flask(__name__, static_folder=".")
CORS(flask_app)

# Global Memory Storage for Active Dataset & Agent Instance
GLOBAL_STORE = {
    "df": None,
    "agent": None,
    "filename": None,
    "file_type": None,
    "file_size": None
}

@flask_app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@flask_app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(".", filename)

@flask_app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "has_dataset": GLOBAL_STORE["agent"] is not None,
        "filename": GLOBAL_STORE["filename"]
    })

@flask_app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """
    Handles CSV/Excel file uploads via FormData or JSON raw data payload.
    Triggers Automatic Dataset Analysis Phase and returns full dataset understanding.
    """
    filename = "uploaded_dataset.csv"
    file_type = "CSV File"
    size_kb = "0 KB"
    df = None

    if request.is_json:
        req_data = request.get_json() or {}
        filename = req_data.get("filename", filename)
        file_type = req_data.get("file_type", file_type)
        size_kb = req_data.get("file_size", size_kb)
        raw_rows = req_data.get("data", [])
        if not raw_rows:
            return jsonify({"error": "JSON dataset payload is empty."}), 400
        df = pd.DataFrame(raw_rows)
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        filename = file.filename
        file_bytes = file.read()
        size_kb = f"{round(len(file_bytes) / 1024, 1)} KB"

        try:
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(file_bytes))
                file_type = "Excel Workbook"
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
                file_type = "CSV File"
        except Exception as e:
            return jsonify({"error": f"Failed to parse dataset file: {str(e)}"}), 500
    else:
        return jsonify({"error": "No file or data payload provided."}), 400

    if df is None or df.empty:
        return jsonify({"error": "Uploaded dataset is empty."}), 400

    try:
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Initialize Agent
        agent = DataAnalystAgent(df)
        analysis_meta = agent.get_dataset_analysis()

        GLOBAL_STORE["df"] = df
        GLOBAL_STORE["agent"] = agent
        GLOBAL_STORE["filename"] = filename
        GLOBAL_STORE["file_type"] = file_type
        GLOBAL_STORE["file_size"] = size_kb

        return jsonify({
            "success": True,
            "filename": filename,
            "file_type": file_type,
            "file_size": size_kb,
            "domain": analysis_meta.get("domain", "General Analytics"),
            "health_score": analysis_meta.get("health_score", 95),
            "health_category": analysis_meta.get("health_category", "Excellent"),
            "fix_suggestions": analysis_meta.get("fix_suggestions", []),
            "total_rows": analysis_meta["total_rows"],
            "total_columns": analysis_meta["total_columns"],
            "duplicate_rows": analysis_meta["duplicate_rows"],
            "total_missing": analysis_meta["total_missing"],
            "kpi_cards": analysis_meta.get("kpi_cards", []),
            "fields": list(df.columns),
            "columns_metadata": analysis_meta["columns_metadata"],
            "executive_summary": analysis_meta["executive_summary_markdown"],
            "sample_rows": df.head(50).to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": f"Failed to analyze dataset: {str(e)}"}), 500

@flask_app.route("/api/query", methods=["POST"])
def handle_query():
    """
    Processes natural language question using DataAnalystAgent.
    Returns human-friendly response with direct answer, insight, recommendation, and charts.
    """
    if not GLOBAL_STORE["agent"]:
        return jsonify({"error": "No dataset uploaded yet. Please upload a CSV or Excel file."}), 400

    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question parameter is required."}), 400

    try:
        agent = GLOBAL_STORE["agent"]
        response_dict = agent.ask(question)
        return jsonify({
            "success": True,
            "question": question,
            **response_dict
        })
    except Exception as e:
        return jsonify({"error": f"Error during query execution: {str(e)}"}), 500

@flask_app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    """Returns dynamic Power BI style dashboard payload."""
    if not GLOBAL_STORE["agent"]:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        agent = GLOBAL_STORE["agent"]
        dashboard_data = agent.get_power_bi_dashboard()
        return jsonify({
            "success": True,
            "filename": GLOBAL_STORE["filename"],
            **dashboard_data
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate dashboard: {str(e)}"}), 500

@flask_app.route("/api/clean", methods=["POST"])
def clean_dataset():
    """Performs automated data cleaning (missing values & duplicate rows)."""
    if not GLOBAL_STORE["agent"]:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        agent = GLOBAL_STORE["agent"]
        cleaned_df, report = agent.tools.clean_dataset_auto()
        
        # Re-initialize Agent with cleaned dataset
        new_agent = DataAnalystAgent(cleaned_df)
        GLOBAL_STORE["df"] = cleaned_df
        GLOBAL_STORE["agent"] = new_agent
        meta = new_agent.get_dataset_analysis()

        return jsonify({
            "success": True,
            "message": "Dataset auto-cleaned successfully.",
            "report": report,
            "health_score": meta["health_score"],
            "total_rows": len(cleaned_df),
            "total_columns": len(cleaned_df.columns),
            "sample_rows": cleaned_df.head(50).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": f"Data cleaning failed: {str(e)}"}), 500

@flask_app.route("/api/preview", methods=["GET"])
def get_preview():
    """Returns paginated or searched dataset rows."""
    if GLOBAL_STORE["df"] is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    agent = GLOBAL_STORE["agent"]
    df = agent.active_df if agent else GLOBAL_STORE["df"]

    q = request.args.get("search", "").lower().strip()
    if q:
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
        df_filtered = df[mask]
    else:
        df_filtered = df

    return jsonify({
        "total_active_rows": len(df_filtered),
        "fields": list(df_filtered.columns),
        "rows": df_filtered.head(100).to_dict(orient="records")
    })

def run_flask_in_thread(port=8501):
    def start_server():
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    return port

def run_standalone_server(default_port=8501):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    port = default_port
    for p in range(default_port, default_port + 50):
        try:
            with socketserver.TCPServer(("", p), None):
                port = p
                break
        except OSError:
            continue

    print("\n" + "="*60)
    print(f"  TARS – UNIVERSAL AI DATA ANALYTICS PLATFORM RUNNING AT:")
    print(f"  http://localhost:{port}")
    print("="*60 + "\n")
    sys.stdout.flush()

    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    flask_app.run(host="0.0.0.0", port=port, debug=False)

# Streamlit Integration Handler
if __name__ == "__main__":
    if "STREAMLIT_SERVER_PORT" in os.environ or any("streamlit" in arg for arg in sys.argv):
        st.set_page_config(
            page_title="TARS – Universal AI Data Analytics Platform",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        st.markdown("""
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .block-container {
                    padding: 0rem !important;
                    max-width: 100% !important;
                }
                iframe {
                    border: none;
                    width: 100% !important;
                    min-height: 100vh !important;
                }
            </style>
        """, unsafe_allow_html=True)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "index.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(os.path.join(base_dir, "styles.css"), "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(os.path.join(base_dir, "app.js"), "r", encoding="utf-8") as f:
            js_content = f.read()

        bundle = html_content.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<style>{css_content}</style>'
        ).replace(
            '<script src="app.js"></script>',
            f'<script>{js_content}</script>'
        )

        run_flask_in_thread(port=8502)
        components.html(bundle, height=950, scrolling=True)
    else:
        run_standalone_server(default_port=8501)
