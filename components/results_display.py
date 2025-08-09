import streamlit as st
from typing import List, Tuple


def render_candidates_table(candidates: List[Tuple[str, float, str]]) -> None:
    if not candidates:
        st.info("No candidates to display.")
        return

    import pandas as pd

    table_rows = [
        {"Rank": index + 1, "Name": name, "Similarity": float(score) * 100.0}
        for index, (name, score, _resume_text) in enumerate(candidates)
    ]
    df = pd.DataFrame(table_rows, columns=["Rank", "Name", "Similarity"])

    df["Similarity"] = df["Similarity"].map(lambda x: f"{x:.1f}%")
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_candidate_feedback(
    job_description: str,
    candidates: List[Tuple[str, float, str]],
    embedder,
) -> None:
    if not candidates:
        st.info("No candidates to explain.")
        return

    for rank, (name, score, resume_text) in enumerate(candidates, 1):
        score_percent = f"{float(score):.1%}"
        with st.expander(f"#{rank} {name} | Similarity: {score_percent}"):
            with st.spinner("Generating reasoning..."):
                reasoning = embedder.generate_fit_reasoning(
                    job_description, resume_text, name
                )
            st.markdown(reasoning)
            with st.expander("View full resume text"):
                st.text(resume_text)
        st.divider()