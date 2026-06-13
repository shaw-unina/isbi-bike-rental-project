import streamlit as st

st.set_page_config(
    page_title="Streamlit Concepts Playground",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Streamlit Concepts Playground")

st.write(
    """
Welcome!

Use the sidebar on the left to explore these interactive pages:

1. **Session State** – try to maintain values between reruns.
2. **Connection** – simulate a database query.
3. **Caching** – see how to avoid repeated heavy work.
4. **Pages** – understand how a multi-page app works in Streamlit.

Each page is interactive: you can click buttons, enter text,
view tables and loading times.
Each page also has a button that shows/hides professionally
commented example code.
"""
)

st.info(
    "Tip: this is a multi-page app. All files in the `pages/` folder "
    "are automatically shown as pages in the sidebar."
)

st.write("👉 Open the menu on the left and choose a page to get started.")
