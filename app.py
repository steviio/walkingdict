"""WalkingDict — Streamlit entry point.

Run:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="WalkingDict",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.sidebar import render_sidebar
from ui.main_panel import render_main_panel
from ui.right_column import render_right_column


def main() -> None:
    profile = render_sidebar()

    chat_col, extra_col = st.columns([3, 1])

    with chat_col:
        st.header("WalkingDict")
        st.caption("Look up any word, phrase, or slang and get an explanation that actually makes sense to you.")
        render_main_panel(profile)

    with extra_col:
        st.header(" ")  # Align with main header
        render_right_column()


if __name__ == "__main__":
    main()
