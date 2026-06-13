import streamlit as st

st.set_page_config(page_title="Session State", page_icon="🧠")

st.title("Session State 🧠")

st.write(
    """
**Objective:** preserve values in memory between reruns,
in the same browser tab.

Why it's useful:
- Every interaction with Streamlit re-executes the script.
- Without state, you would lose information (e.g., counters, user notes).
- With `st.session_state` you can maintain those values for the entire current session.
"""
)

# --- initialize persistent values only the first time ---
if "counter" not in st.session_state:
    st.session_state.counter = 0

if "notes" not in st.session_state:
    st.session_state.notes = []

st.subheader("Interactive Demo")

col_a, col_b = st.columns(2)

with col_a:
    st.metric("Current counter value", st.session_state.counter)
    if st.button("➕ Increment counter"):
        st.session_state.counter += 1
        st.rerun()  # reload immediately to update the metric

with col_b:
    nuova_nota = st.text_input("Add a note to the session:")
    if st.button("💾 Save note"):
        if nuova_nota.strip():
            st.session_state.notes.append(nuova_nota.strip())
            st.success("Note saved to session state ✅")
        else:
            st.warning("The note is empty, not saved.")

st.write("Your notes in this session:")
st.write(st.session_state.notes)

st.caption(
    "Closing the tab or doing a hard refresh resets the session. "
    "As long as you stay here, values persist."
)

# --- show/hide example code ---
st.divider()
if "show_code_session" not in st.session_state:
    st.session_state.show_code_session = False

if st.button("Show / Hide explained code"):
    st.session_state.show_code_session = not st.session_state.show_code_session

if st.session_state.show_code_session:
    st.code(
        '''
import streamlit as st

# 1. Initialize variables in session state (only the first time)
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "notes" not in st.session_state:
    st.session_state.notes = []

# 2. Use these variables in the interface
st.write("Counter:", st.session_state.counter)

if st.button("Increment"):
    st.session_state.counter += 1
    st.rerun()  # force a refresh to see the new value immediately

note = st.text_input("Add a note:")
if st.button("Save note"):
    st.session_state.notes.append(note)

st.write("Saved notes:", st.session_state.notes)

# Key idea:
# - st.session_state is like a mini dictionary for each user / tab.
# - Each browser tab has its own separate session state.
        ''',
        language="python",
    )
