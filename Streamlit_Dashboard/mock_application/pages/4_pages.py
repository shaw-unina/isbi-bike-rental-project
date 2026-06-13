import streamlit as st
import os

st.set_page_config(page_title="Pages", page_icon="📄")

st.title("Pages 📄")

st.write(
    """
**Objective:** understand how Streamlit handles multi-page apps.

Typical structure:
1. `app.py` is the entry point (the script you launch with `streamlit run app.py`).
2. There is a `pages/` folder.
3. Every `.py` file inside `pages/` automatically becomes a page in the sidebar.

This page you're reading is exactly one of those files inside `pages/`.
Streamlit detects it automatically — you don't need to write routers or manual menus.
"""
)

st.subheader("Files found in ./pages folder")
pages_dir = os.path.dirname(__file__)
page_files = [
    f for f in os.listdir(pages_dir)
    if f.endswith(".py")
]
st.write(page_files)

st.success(
    "See? All these files appear in the sidebar as navigation entries. "
    "Just putting them inside the `pages/` folder is enough."
)

# --- show/hide example code ---
st.divider()
if "show_code_pages" not in st.session_state:
    st.session_state.show_code_pages = False

if st.button("Show / Hide explained code"):
    st.session_state.show_code_pages = not st.session_state.show_code_pages

if st.session_state.show_code_pages:
    st.code(
        '''
# app.py
import streamlit as st

st.set_page_config(page_title="Main App", page_icon="🚀")
st.title("Main App")
st.write("Use the sidebar to navigate between other pages.")

"""
pages/1_overview.py
pages/2_analysis.py
pages/3_other.py
...
Every .py file inside `pages/` becomes a page in the sidebar automatically.
"""

# No need to create a complex custom menu.
# Streamlit scans the `pages/` folder and generates the navigation.
        ''',
        language="python",
    )
