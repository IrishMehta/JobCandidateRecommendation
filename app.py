import os
import streamlit as st
from dotenv import load_dotenv
from typing import List

from components.sidebar import render_filter_settings 
from components.file_uploader import upload_files
from components.results_display import (
    render_candidates_table,
    render_candidate_feedback,
)
from src.resume_parser import parse_resume_sync, extract_name_from_resume
from src.embeddings import JobResumeEmbedder
from config.settings import PAGE_TITLE, PAGE_ICON

load_dotenv()
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Ensure unselected tabs are clearly visible */
    button[role="tab"] {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    /* Fix the placeholder text color inside the file uploader */
    [data-testid="stFileUploaderDropzone"] > div > p {
        color: #475569 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Force light mode and professional font via CSS (in addition to config)
st.markdown(
    """
    <style>
    :root { color-scheme: light; }

    html, body, [data-testid="stAppViewContainer"] {
      background: #ffffff !important;
      color: #111111 !important;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, "Segoe UI", Roboto, "Noto Sans", sans-serif !important;
    }

    /* Header and general surfaces */
    [data-testid="stHeader"] { background: #ffffff !important; }
    [data-testid="stVerticalBlock"]>div { background: transparent; }

    /* Inputs and text areas */
    textarea,
    .stTextInput input,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
      background-color: #ffffff !important;
      color: #111111 !important;
      border: 1px solid #cbd5e1 !important;
      border-radius: 8px !important;
    }
    textarea::placeholder,
    .stTextInput input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
      color: #64748b !important;
    }

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
      background-color: #ffffff !important;
      border: 1px dashed #cbd5e1 !important;
      color: #111111 !important;
      border-radius: 10px !important;
    }

    /* Tabs and captions */
    [role="tablist"] { border-bottom: 1px solid #e5e7eb; }
    [data-testid="stCaptionContainer"] { color: #475569 !important; }

    /* Labels for widgets */
    label, .stSlider label, .stNumberInput label {
      color: #0f172a !important;
      font-weight: 600 !important;
    }

    /* Info/alert boxes (for How To) */
    div[role="alert"] {
      background: #e8f1ff !important;
      color: #0f172a !important;
      border: 1px solid #c6dbff !important;
    }
    div[role="alert"] * { color: #0f172a !important; }

    /* Button */
    .stButton>button {
      font-weight: 600;
      background: #2f5bea !important;
      color: #ffffff !important;
      border: 1px solid #2f5bea !important;
      border-radius: 8px !important;
    }
    .stButton>button:hover {
      background: #2449c6 !important;
      border-color: #2449c6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Candidate DataFrame: white background, black borders, dark text */
    [data-testid="stDataFrame"] {
      background: #ffffff !important;
      border: 1px solid #000000 !important;
      border-radius: 8px !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="gridcell"] {
      background: #ffffff !important;
      color: #111111 !important;
      border-right: 1px solid #000000 !important;
      border-bottom: 1px solid #000000 !important;
    }
    [data-testid="stDataFrame"] [role="row"]:last-child [role="gridcell"],
    [data-testid="stDataFrame"] [role="row"]:last-child [role="columnheader"] {
      border-bottom: 1px solid #000000 !important;
    }
    /* Support static tables too */
    [data-testid="stTable"] table {
      border-collapse: collapse !important;
      background: #ffffff !important;
      color: #111111 !important;
    }
    [data-testid="stTable"] th, [data-testid="stTable"] td {
      border: 1px solid #000000 !important;
    }

    /* File uploader: blue dropzone like primary button */
    [data-testid="stFileUploaderDropzone"] {
      background-color: #2f5bea !important;
      border: 1px dashed #2449c6 !important;
      color: #ffffff !important;
      border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #ffffff !important; }
    [data-testid="stFileUploaderDropzone"] svg { fill: #ffffff !important; }
    /* Make the 'Browse files' control text black on a white pill for contrast */
    [data-testid="stFileUploaderDropzone"] a {
      color: #111111 !important;
      background: #ffffff !important;
      padding: 2px 8px;
      border-radius: 6px;
      text-decoration: none !important;
      border: 1px solid #e5e7eb !important;
      display: inline-block;
    }

    /* Number input steppers: use brand blue */
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] [data-baseweb="button"] {
      background: #2f5bea !important;
      color: #ffffff !important;
      border: 1px solid #2f5bea !important;
      border-radius: 6px !important;
    }
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] [data-baseweb="button"]:hover {
      background: #2449c6 !important;
      border-color: #2449c6 !important;
    }
    [data-testid="stNumberInput"] svg { fill: #ffffff !important; }

    /* Expander: keep same light background when opened */
    [data-testid="stExpander"] {
      background: #ffffff !important;
      color: #111111 !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 8px !important;
    }
    [data-testid="stExpander"] div[role="button"] {
      background: #ffffff !important;
      color: #111111 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] *:not(svg) {
      color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(PAGE_TITLE)


st.info(
    "1) Paste the job description. \n"
    "2) Upload resumes or paste resume texts. \n"
    "3) Adjust threshold and top-k. \n"
    "4) Click ‘Find Top Candidates’."
)

# 1) Job description
st.header("Job description", divider="gray")
st.caption("Paste the role summary, responsibilities, and requirements for the position.")
job_description = st.text_area(
    "Job description",
    height=200,
    placeholder="Paste job description here...",
)

# 2) Entering Resume/s
st.header("Entering Resume/s", divider="gray")
st.caption("Upload one or more resume files or paste one or more resume texts.")

uploaded_files: List[str] = []
pasted_texts: List[str] = []

upload_tab, paste_tab = st.tabs(["Upload files", "Paste text"])
with upload_tab:
    uploaded_files = upload_files(show_header=False)
with paste_tab:
    num_texts = st.number_input(
        "Number of resume texts",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )
    for i in range(int(num_texts)):
        text_val = st.text_area(
            f"Resume {i + 1} text",
            height=200,
            key=f"resume_text_{i}",
        )
        if text_val and text_val.strip():
            pasted_texts.append(text_val.strip())

# 3) Top Candidates
st.header("Top Candidates", divider="gray")
st.caption("Adjust filters and find the highest-scoring candidates for this role.")
threshold, top_k = render_filter_settings()

run_disabled = not job_description or (not uploaded_files and not pasted_texts)
find_clicked = st.button("Find Top Candidates", type="primary", disabled=run_disabled)

# Placeholder for the ranked table
top_table_container = st.container()

# 4) Candidate Profile Feedback
st.header("Candidate Profile Feedback", divider="gray")
st.caption("Read a brief explanation of how each recommended candidate matches the role.")
feedback_container = st.container()

if find_clicked:
    with st.spinner("Processing resumes and generating rankings..."):
        raw_texts: List[str] = []
        names: List[str] = []

        for path in uploaded_files:
            text = parse_resume_sync(path)
            raw_texts.append(text)

            extracted = extract_name_from_resume(text or "")
            candidate_name = (
                extracted if extracted != "Name not found" else os.path.basename(path)
            )
            names.append(candidate_name)

        # From pasted text entries
        for idx, text in enumerate(pasted_texts):
            raw_texts.append(text)

            extracted = extract_name_from_resume(text or "")
            candidate_name = (
                extracted if extracted != "Name not found" else f"Pasted Resume {idx + 1}"
            )
            names.append(candidate_name)

        # Filter out parsing errors or empty content
        texts: List[str] = []
        valid_names: List[str] = []
        for n, t in zip(names, raw_texts):
            t_norm = (t or "").strip()
            if not t_norm:
                st.warning(f"Skipping {n}: empty content")
                continue
            if t_norm.startswith("Error") or t_norm == "No content extracted":
                st.warning(f"Skipping {n}: {t_norm[:120]}")
                continue
            texts.append(t_norm)
            valid_names.append(n)

        if not texts:
            with top_table_container:
                st.info("No valid parsed resumes to rank.")
        else:
            embedder = JobResumeEmbedder()
            ranked = embedder.rank_candidates(job_description, texts, valid_names)
            filtered = [(n, s, t) for n, s, t in ranked if s >= threshold][: top_k]

            with top_table_container:
                render_candidates_table(filtered)

            with feedback_container:
                render_candidate_feedback(job_description, filtered, embedder)

st.caption("Built with JobBERT-v2 embeddings and BART reasoning.") 