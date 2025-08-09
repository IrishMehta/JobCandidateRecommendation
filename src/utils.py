import os
import streamlit as st
from typing import Iterable, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_uploaded_file(dir_path: str, filename: str, content: bytes) -> str:
    ensure_dir(dir_path)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


def clean_text(text: str) -> str:
    return (text or "").replace("\x00", " ").strip()


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable from either Streamlit secrets or OS environment.
    
    This function works in both local development (using .env files) and 
    Streamlit Cloud deployment (using st.secrets).
    
    Args:
        key: The environment variable key
        default: Default value if key is not found
        
    Returns:
        The environment variable value or default
    """
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        # Fallback to os.environ if st.secrets is not available or fails
        pass
    
    # Fallback to standard environment variables (for local development)
    return os.getenv(key, default) 