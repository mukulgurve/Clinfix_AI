# app.py - ClinFix AI (Hugging Face ready)

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from docx import Document
import tempfile
import uuid
import os
from datetime import datetime

# -----------------------------
# Configurable clinical limits
# -----------------------------
SBP_MIN, SBP_MAX = 90, 180
DBP_MIN, DBP_MAX = 60, 120
WT_MIN, WT_MAX = 40, 150

# -----------------------------
# Global storage
# -----------------------------
_df_store = {"df": None, "last_load_time": None}

# -----------------------------
# Utilities
# -----------------------------
def _safe_load(csv_file):
    try:
        df = pd.read_csv(csv_file.name)
        return df, None
    except Exception as e:
        return None, str(e)

def _generate_report_text(df):
    lines = []
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append("-"*60)
    lines.append("PREVIEW (first 5 rows):")
    lines.append(df.head().to_string())
    lines.append("")

    # Missing summary
    miss = df.isnull().sum()
    lines.append("MISSING VALUE SUMMARY:")
    lines.append(miss.to_string())
    lines.append("")

    # Exact missing locations
    rows, cols = np.where(df.isnull())
    if len(rows) > 0:
        lines.append("EXACT MISSING VALUE LOCATIONS:")
        for r, c in zip(rows, cols):
            lines.append(f" - Row {r+1}, Column '{df.columns[c]}'")
    else:
        lines.append("No missing values found.")
    lines.append("")

    # Imputation
    df_filled = df.copy()
    impute_msgs = []
    for col in df_filled.columns:
        if pd.api.types.is_numeric_dtype(df_filled[col]):
            if df_filled[col].isnull().any():
                mean_val = df_filled[col].mean()
                df_filled[col].fillna(mean_val, inplace=True)
                impute_msgs.append(f"Filled numeric missing in {col} with mean={mean_val:.2f}")
        else:
            if df_filled[col].isnull().any():
                mode_vals = df_filled[col].mode()
                if len(mode_vals) > 0:
                    df_filled[col].fillna(mode_vals[0], inplace=True)
                    impute_msgs.append(f"Filled categorical missing in {col} with mode='{mode_vals[0]}'")
                else:
                    df_filled[col].fillna("", inplace=True)
                    impute_msgs.append(f"Filled categorical missing in {col} with empty string")
    if impute_msgs:
        lines.append("IMPUTATION DONE:")
        lines += impute_msgs
    else:
        lines.append("No imputation needed.")
    lines.append("")

    # Abnormal checks
    abnormal_msgs = []
    if "SBP" in df_filled.columns:
        sbp_bad = df_filled[(df_filled["SBP"] < SBP_MIN) | (df_filled["SBP"] > SBP_MAX)]
        if not sbp_bad.empty:
            abnormal_msgs.append(f"SBP out of range ({SBP_MIN}-{SBP_MAX}):")
            abnormal_msgs.append(sbp_bad.to_string())
    if "DBP" in df_filled.columns:
        dbp_bad = df_filled[(df_filled["DBP"] < DBP_MIN) | (df_filled["DBP"] > DBP_MAX)]
        if not dbp_bad.empty:
            abnormal_msgs.append(f"DBP out of range ({DBP_MIN}-{DBP_MAX}):")
            abnormal_msgs.append(dbp_bad.to_string())
    if "WEIGHT" in df_filled.columns:
        wt_bad = df_filled[(df_filled["WEIGHT"] < WT_MIN) | (df_filled["WEIGHT"] > WT_MAX)]
        if not wt_bad.empty:
            abnormal_msgs.append(f"WEIGHT out of range ({WT_MIN}-{WT_MAX}):")
            abnormal_msgs.append(wt_bad.to_string())
    if abnormal_msgs:
        lines.append("ABNORMAL / OUT-OF-RANGE VALUES:")
        lines += abnormal_msgs
    else:
        lines.append("No abnormal values detected.")
    lines.append("")

    # Summary statistics
    try:
        lines.append("SUMMARY STATISTICS (numeric):")
        lines.append(df_filled.describe().to_string())
    except Exception:
        lines.append("Unable to compute summary statistics.")
    lines.append("")

    return "\n".join(lines), df_filled

# -----------------------------
# Feature functions
# -----------------------------
def fn_process_file(csv_file):
    if csv_file is None:
        return "⚠️ Please upload a CSV file."
    df, err = _safe_load(csv_file)
    if err:
        return f"❌ Error reading CSV: {err}"
    _df_store["df"] = df.copy()
    _df_store["last_load_time"] = datetime.utcnow().isoformat()
    report, _ = _generate_report_text(df)
    return report

def fn_adam_report(_csv=None):
    df = _df_store["df"]
    if df is None:
        return "⚠️ Load a CSV first (Process File)."
    df2 = df.copy()
    if "DATE" in df2.columns:
        df2["DTM"] = pd.to_datetime(df2["DATE"], errors="coerce")
    df2["VISIT"] = df2.get("VISIT", "VISIT 1")
    df2["VISITNUM"] = df2.get("VISITNUM", 1)
    txt = "ADaM-READY PREVIEW (top rows):\n\n" + df2.head().to_string()
    return txt

def fn_graphs(_csv=None):
    df = _df_store["df"]
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5))
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        ax.text(0.5, 0.5, "No numeric columns to plot", ha="center", va="center")
    else:
        col = numeric[0]
        ax.plot(df[col].reset_index(drop=True), marker='o', linestyle='-')
        ax.set_title(f"Trend: {col}")
        ax.set_xlabel("Record")
        ax.set_ylabel(col)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def fn_duplicates(_csv=None):
    df = _df_store["df"]
    if df is None:
        return "⚠️ Load a CSV first."
    dup = df[df.duplicated(keep=False)]
    if dup.empty:
        return "✅ No duplicate records found."
    return "⚠ DUPLICATES FOUND:\n\n" + dup.to_string()

def fn_inconsistent(_csv=None):
    df = _df_store["df"]
    if df is None:
        return "⚠️ Load a CSV first."
    msgs = []
    for col in ["SBP","DBP","WEIGHT"]:
        if col in df.columns:
            bad = df[(df[col].astype(float, errors='ignore') < 0)]
            if not bad.empty:
                msgs.append(f"{col} negative values:\n{bad.to_string()}")
    if "DATE" in df.columns:
        parsed = pd.to_datetime(df["DATE"], errors="coerce")
        bad = df[parsed.isna()]
        if not bad.empty:
            msgs.append("Unparseable DATE rows:\n" + bad.to_string())
    if not msgs:
        return "✅ No inconsistent values detected."
    return "\n\n".join(msgs)

# -----------------------------
# Export helpers
# -----------------------------
def _write_pdf(text):
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(tempfile.gettempdir(), f"clinfix_report_{uid}.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=10)
    for line in text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.output(path)
    return path

def _write_docx(text):
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(tempfile.gettempdir(), f"clinfix_report_{uid}.docx")
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)
    return path

def fn_download_pdf(_csv=None):
    df = _df_store["df"]
    if df is None:
        return None
    report, _ = _generate_report_text(df)
    full = "ClinFix AI - Full Corrected Report\n\n" + report
    return _write_pdf(full)

def fn_download_docx(_csv=None):
    df = _df_store["df"]
    if df is None:
        return None
    report, _ = _generate_report_text(df)
    full = "ClinFix AI - Full Corrected Report\n\n" + report
    return _write_docx(full)

def fn_download_cleaned_csv(_csv=None):
    df = _df_store["df"]
    if df is None:
        return None
    _, df_filled = _generate_report_text(df)
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(tempfile.gettempdir(), f"clinfix_cleaned_{uid}.csv")
    df_filled.to_csv(path, index=False)
    return path

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("<h2 style='text-align:center;color:#00bfa5;'>⚕️ ClinFix AI - Premium Clinical CSV Analyzer</h2>")

    with gr.Row():
        csv_input = gr.File(label="Upload CSV File (.csv)")

    result_box = gr.Textbox(label="🔍 Output / Report", lines=20)
    graph_box = gr.Image(label="📊 Graphs Preview", height=280)

    with gr.Row():
        btn_process = gr.Button("⚙️ Process File")
        btn_adam = gr.Button("📘 ADaM Report")
        btn_graphs = gr.Button("📊 Graphs")
        btn_dups = gr.Button("🔁 Duplicates")
        btn_incon = gr.Button("⚠ Inconsistent Values")

    with gr.Row():
        btn_pdf = gr.Button("⬇ PDF Report")
        btn_docx = gr.Button("⬇ DOCX Report")
        btn_csv = gr.Button("⬇ Cleaned CSV")

    pdf_out = gr.File(label="PDF Output")
    docx_out = gr.File(label="DOCX Output")
    csv_out = gr.File(label="Cleaned CSV Output")

    # Button actions
    btn_process.click(fn_process_file, csv_input, result_box)
    btn_adam.click(fn_adam_report, None, result_box)
    btn_graphs.click(fn_graphs, None, graph_box)
    btn_dups.click(fn_duplicates, None, result_box)
    btn_incon.click(fn_inconsistent, None, result_box)
    btn_pdf.click(fn_download_pdf, None, pdf_out)
    btn_docx.click(fn_download_docx, None, docx_out)
    btn_csv.click(fn_download_cleaned_csv, None, csv_out)

demo.launch()