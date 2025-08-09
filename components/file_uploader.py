import os
import streamlit as st
from typing import List
from src.utils import ensure_dir, save_uploaded_file
from config.settings import MAX_FILE_SIZE, ALLOWED_EXTENSIONS

UPLOAD_DIR = os.path.join("data", "temp_uploads")

def _is_allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in {e.lower() for e in ALLOWED_EXTENSIONS}


def _is_size_ok(upload) -> bool:
    size_mb = (upload.size or 0) / (1024 * 1024)
    return size_mb <= float(MAX_FILE_SIZE)


def upload_files(show_header: bool = True) -> List[str]:
    """Renders the resume file uploader and handles saving."""
    if show_header:
        st.subheader("📄 Upload Resumes")

    uploads = st.file_uploader(
        "Upload one or more resumes (PDF, TXT, or DOCX).",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    saved_paths: List[str] = []
    if uploads:
        ensure_dir(UPLOAD_DIR)
        valid_uploads = []

        for f in uploads:
            if not _is_allowed(f.name):
                st.warning(f"Skipping '{f.name}': Invalid file type.")
                continue
            if not _is_size_ok(f):
                st.warning(f"Skipping '{f.name}': Exceeds {MAX_FILE_SIZE} MB limit.")
                continue
            valid_uploads.append(f)

        if valid_uploads:
            for f in valid_uploads:
                path = save_uploaded_file(UPLOAD_DIR, f.name, f.getbuffer())
                saved_paths.append(path)

            st.success(f"Successfully processed {len(saved_paths)} resume(s).")
            with st.expander("📄 View uploaded files"):
                for path in saved_paths:
                    st.write(f"• {os.path.basename(path)}")

    return saved_paths