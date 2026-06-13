import streamlit as st
import pandas as pd
import time
import random

st.set_page_config(page_title="Connection", page_icon="🔌")

st.title("Connection 🔌")

st.write(
    """
**Objective:** show how to retrieve data from an external source
(for example a database) and display it in Streamlit.

Here we simulate a database query (mock).
In a "real" app, you would use `st.connection("db_name")` and credentials
would be stored in `.streamlit/secrets.toml`, not written in plain text in code.

Below you can click a button to execute a fake query.
"""
)

def finta_query_database():
    """
    Simulates database access and returns a DataFrame.
    We use a small delay to simulate real I/O.
    """
    time.sleep(0.5)  # simulate query latency
    righe = []
    for i in range(5):
        righe.append(
            {
                "sensor_id": i + 1,
                "temp_c": round(random.uniform(40, 80), 2),
                "rpm": random.randint(1100, 1700),
                "status": random.choice(["OK", "WARN", "FAIL"]),
            }
        )
    return pd.DataFrame(righe)

if st.button("📡 Execute database query"):
    df = finta_query_database()
    st.success("Data loaded from 'database' (mock).")
    st.dataframe(df, use_container_width=True)
else:
    st.info("Press the button above to retrieve data.")

st.caption(
    "In production: use `st.connection(...)` and store credentials in `.streamlit/secrets.toml`, "
    "never directly in published code."
)

# --- show/hide example code ---
st.divider()
if "show_code_connection" not in st.session_state:
    st.session_state.show_code_connection = False

if st.button("Show / Hide explained code"):
    st.session_state.show_code_connection = not st.session_state.show_code_connection

if st.session_state.show_code_connection:
    st.code(
        '''
import streamlit as st
import pandas as pd
import time
import random

def fake_database_query():
    """
    Mock of DB access.
    In a real case you would do:
        conn = st.connection("my_database")
        df = conn.query("SELECT * FROM my_table")
    Here we generate random data instead of actually reading from DB.
    """
    time.sleep(0.5)  # simulate a small network / disk delay
    rows = []
    for i in range(5):
        rows.append(
            {
                "sensor_id": i + 1,
                "temp_c": round(random.uniform(40, 80), 2),
                "rpm": random.randint(1100, 1700),
                "status": random.choice(["OK", "WARN", "FAIL"]),
            }
        )
    return pd.DataFrame(rows)

st.title("Connection Demo")

if st.button("Query DB"):
    df = fake_database_query()
    st.dataframe(df)

# Key points:
# - In production DO NOT put username/password in code.
# - Use .streamlit/secrets.toml and st.connection("name").
# - Then do df = conn.query("SELECT ...") and display the result.
        ''',
        language="python",
    )
