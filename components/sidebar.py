import streamlit as st
from config.settings import SIMILARITY_THRESHOLD, TOP_CANDIDATES

def render_filter_settings():
    """Renders the filter settings widgets horizontally in a clean layout."""

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "Similarity score threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(SIMILARITY_THRESHOLD),
            step=0.01,
        )

    with col2:
        top_k = st.number_input(
            "Number of top candidates",
            min_value=1,
            max_value=50,
            value=int(TOP_CANDIDATES),
        )
    
    st.caption("Adjust these settings to filter the ranked list of candidates.")
    return threshold, top_k