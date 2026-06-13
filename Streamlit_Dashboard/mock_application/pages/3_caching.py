import streamlit as st
import pandas as pd
import time
import random
import joblib
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Caching", page_icon="💾")

st.title("Caching 💾")

st.write(
    """
**Objective:** avoid redoing heavy work when it's not necessary.

Streamlit offers two important decorators:

- `@st.cache_data`
  Use this when the function returns **normal/serializable data**
  (DataFrame, dict, lists, numbers, strings…).
  Streamlit saves the result and if you call the function with the same inputs,
  it immediately returns the cached copy instead of recalculating.

- `@st.cache_resource`
  Use this when the function must return an **expensive resource to load**,
  for example a Machine Learning model already trained on disk,
  or a database connection.
  In this case Streamlit always gives you back **the same object** without reloading it.
"""
)

# ------------------------------------------------------------
# EXAMPLE 1: cache_data
# We simulate a "slow" data load from some source
# (remote API, huge CSV, heavy preprocessing...)
# ------------------------------------------------------------
@st.cache_data
def load_expensive_data():
    """Simulates slow loading/processing of raw data."""
    time.sleep(2)  # simulate 2 seconds of heavy work
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="min"),
            "temp_c": [round(random.uniform(40, 65), 2) for _ in range(5)],
        }
    )


# ------------------------------------------------------------
# EXAMPLE 2: cache_resource
# Here we load a real ML model from disk, only once.
#
# Suppose we have:
#  - a "models/" folder
#  - a "random_forest.pkl" file saved with joblib.dump(model, "models/random_forest.pkl")
#
# This is a realistic case: models can weigh tens/hundreds of MB.
# You don't want to reload them every time the user clicks something.
# ------------------------------------------------------------
@st.cache_resource
def load_ml_model():
    """
    Loads a pre-trained ML model from disk into memory.
    This operation can be very expensive (I/O + deserialization).

    Thanks to @st.cache_resource:
    - The first time it is actually loaded from the .pkl file
    - Subsequent times the object already in memory is reused
    """
    time.sleep(1.5)  # small fake delay to simulate load cost

    model_path = os.path.join("models", "random_forest.pkl")

    if not os.path.exists(model_path):
        # No model found: inform cleanly.
        return {
            "error": True,
            "message": (
                f"WARNING: model not found at {model_path}. "
                "Use the section below 'Train and save model' to create one."
            ),
        }

    model = joblib.load(model_path)
    return {
        "error": False,
        "model_object": model,
        "model_type": type(model).__name__,
        "parameters": getattr(model, "get_params", lambda: {})(),
    }


# ------------------------------------------------------------
# MODEL TRAIN + SAVE FUNCTION
# This is the new part that was previously given as a separate script.
# Now it's integrated in the app: we train a RandomForestClassifier on the Iris dataset,
# and save it in models/random_forest.pkl
# ------------------------------------------------------------
def train_and_save_random_forest_model():
    """
    Trains a Random Forest classifier on the Iris dataset and saves it to disk.
    Returns useful info to display in the UI.
    """
    # 1. Load a simple classification dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Train the Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    test_accuracy = clf.score(X_test, y_test)

    # 4. Create models/ folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # 5. Save the model
    model_path = os.path.join("models", "random_forest.pkl")
    joblib.dump(clf, model_path)

    return {
        "model_path": model_path,
        "accuracy_test": test_accuracy,
        "model_type": type(clf).__name__,
        "parameters": getattr(clf, "get_params", lambda: {})(),
    }


# ------------------------------------------------------------
# USER INTERFACE
# ------------------------------------------------------------

col_sx, col_dx = st.columns(2)

with col_sx:
    st.subheader("Cached data (cache_data)")
    st.write(
        "This function simulates expensive data acquisition/processing. "
        "Try clicking multiple times and see how long it takes."
    )

    if st.button("📊 Load data"):
        start = time.time()
        df = load_expensive_data()
        elapsed = time.time() - start

        st.dataframe(df, use_container_width=True)
        st.info(
            f"Loaded in {elapsed:.3f} seconds. "
            "From the second click onwards it should be much faster "
            "because the result is taken from cache."
        )

with col_dx:
    st.subheader("ML model in cache (cache_resource)")
    st.write(
        "This function loads an ML model saved on disk (random_forest.pkl). "
        "The model is read only the first time, then it's reused.\n\n"
        "If the file doesn't exist yet, go below and create it with 'Train and save model'."
    )

    if st.button("🧠 Load ML model"):
        start = time.time()
        model_info = load_ml_model()
        elapsed = time.time() - start

        if model_info["error"]:
            st.error(model_info["message"])
        else:
            st.success(
                f"Model loaded in {elapsed:.3f} seconds and cached!"
            )
            st.write("Model type:", model_info["model_type"])
            st.write("Model parameters:", model_info["parameters"])

            st.caption(
                "If you press the button again you'll see that loading is "
                "much faster, because the model is already in memory."
            )

st.caption(
    "`cache_data`: returns **a copy** of the cached data each time (you can manipulate it safely).\n"
    "`cache_resource`: always returns **the same shared object** (great for heavy ML models or DB connections)."
)

# ------------------------------------------------------------
# BLOCK: TRAIN AND SAVE MODEL
# ------------------------------------------------------------
st.divider()
st.subheader("🏗️ Train and save example model")
st.write(
    "Here we actually train a RandomForestClassifier on the Iris dataset "
    "and save it in `models/random_forest.pkl`. "
    "After this step the '🧠 Load ML model' button will use the saved file."
)

if st.button("🏁 Train and save RandomForest"):
    start = time.time()
    info_train = train_and_save_random_forest_model()
    elapsed = time.time() - start

    st.success("Model trained and saved!")
    st.write("File path:", info_train["model_path"])
    st.write("Test accuracy:", round(info_train["accuracy_test"], 3))
    st.write("Model type:", info_train["model_type"])
    st.write("Model parameters:", info_train["parameters"])
    st.info(
        f"Operation completed in {elapsed:.3f} seconds. "
        "Now you can click '🧠 Load ML model' above to test the cache."
    )

# ------------------------------------------------------------
# BUTTON TO SHOW EXPLAINED CODE
# ------------------------------------------------------------
st.divider()

if "show_code_caching" not in st.session_state:
    st.session_state.show_code_caching = False

if st.button("Show / Hide explained code"):
    st.session_state.show_code_caching = not st.session_state.show_code_caching

if st.session_state.show_code_caching:
    st.code(
        '''
import streamlit as st
import pandas as pd
import time
import random
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================
# EXAMPLE: @st.cache_data
# ==============================
@st.cache_data
def load_expensive_data():
    """
    "Slow" function: for example reading from external API,
    parsing a huge CSV, data cleaning, etc.
    The first time it's expensive. Then Streamlit saves the result and
    returns it instantly if the inputs don't change.
    """
    time.sleep(2)  # simulate slowness
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="min"),
        "temp_c": [round(random.uniform(40, 65), 2) for _ in range(5)],
    })
    return df


# ==============================
# EXAMPLE: @st.cache_resource
# ==============================
@st.cache_resource
def load_ml_model():
    """
    Loads a pre-trained ML model from disk.
    This can be quite expensive (I/O, deserialization).
    With cache_resource we do it only once:
    subsequent times Streamlit reuses the already loaded object.
    """
    time.sleep(1.5)  # simulate loading cost

    model_path = os.path.join("models", "random_forest.pkl")
    if not os.path.exists(model_path):
        return {
            "error": True,
            "message": (
                f"Model not found at {model_path}. "
                "Create the model with 'Train and save RandomForest'."
            ),
        }

    model = joblib.load(model_path)

    # Return useful info to show to the user
    return {
        "error": False,
        "model_object": model,
        "model_type": type(model).__name__,
        "parameters": getattr(model, "get_params", lambda: {})(),
    }


# ==============================
# TRAINING AND SAVING
# ==============================
def train_and_save_random_forest_model():
    """
    Trains a RandomForestClassifier on the Iris dataset
    and saves the model in models/random_forest.pkl.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    test_accuracy = clf.score(X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "random_forest.pkl")
    joblib.dump(clf, model_path)

    return {
        "model_path": model_path,
        "accuracy_test": test_accuracy,
        "model_type": type(clf).__name__,
        "parameters": getattr(clf, "get_params", lambda: {})(),
    }

# Typical usage in Streamlit UI:
# - Press "🏁 Train and save RandomForest" to create/update models/random_forest.pkl
# - Press "🧠 Load ML model" to see that the model is read only once
#   and then stays in cache thanks to @st.cache_resource.
        ''',
        language="python",
    )
