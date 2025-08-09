import os
from typing import Iterable, List


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