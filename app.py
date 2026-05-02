import streamlit as st
import subprocess
from pathlib import Path
import torch

# Fix Torch Streamlit path error (bare mode)
torch.classes.__path__ = []

# Page Config & Style
st.set_page_config(page_title="Scientific PDF QnA", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #ffffff; }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stTextInput > div > input {
            background-color: #1e222a;
            color: white;
        }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Scientific Paper Q&A Assistant")
st.markdown("Enter the relative or absolute path of a scientific PDF to analyze and ask intelligent questions.")

# State Init
if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

# PDF Path
st.subheader("Provide PDF Path")
pdf_path_input = st.text_input("Enter relative or absolute path to the PDF")

debug_mode = False  # Set to True to show stdout/stderr in UI

if pdf_path_input and st.button("Index PDF"):
    st.session_state.document_ready = False
    failed_path = Path("indexing_failed.txt")
    if failed_path.exists():
        failed_path.unlink()

    with st.spinner("Indexing and extracting metadata... This may take a few minutes."):
        full_pdf_path = str(Path(pdf_path_input).resolve())
        try:
            process = subprocess.Popen(
                f'python essentials/pipeline/pipeline_runner.py --mode index --pdf "{full_pdf_path}"',
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            stdout, _ = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = "", "Indexing process timed out."

    if debug_mode:
        st.text("STDOUT:")
        st.code(stdout or "No stdout")
        # st.text("STDERR:")
        # st.code(stderr or "No stderr")

    if process.returncode == 0 and not failed_path.exists():
        st.success("✅ Document indexed successfully!")
        st.session_state.document_ready = True
    else:
        st.error("❌ Indexing failed.")
        if failed_path.exists():
            st.warning("Error Log from indexing_failed.txt:")
            st.code(failed_path.read_text())
        else:
            st.code(stdout)

# QA Section
if st.session_state.document_ready:
    st.subheader("Ask Questions about the Paper")
    question = st.text_input("Type your question here")
    ask_button = st.button("Submit Question")

    if ask_button and question.strip():
        with st.spinner("Generating answer... This may take a few moments."):
            try:
                process = subprocess.Popen(
                    f'python essentials/pipeline/pipeline_runner.py --mode query --question "{question}"',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                stdout, _ = process.communicate(timeout=800)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = "", "Query process timed out."

        if debug_mode:
            st.text("STDOUT:")
            st.code(stdout or "No stdout")
            # st.text("STDERR:")
            # st.code(stderr or "No stderr")

        if process.returncode == 0:
            if "LLM Answer:" in stdout:
                answer = stdout.split("LLM Answer:")[-1].strip()
            else:
                answer = stdout.strip()
            st.success("Answer:")
            st.markdown(f"<div style='background-color:#1e1e1e; padding:1em; border-radius:8px'><strong>{answer}</strong></div>", unsafe_allow_html=True)
        else:
            st.error("❌ Query failed.")
            st.code(stdout)

st.markdown("---")
