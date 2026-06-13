import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Forecast Simulation", layout="wide")

st.title("🔮 Real-Time Forecasting Simulation")
st.markdown("Scroll the timeline to generate predictions based on historical input data.")

# ==============================================================================
# 1. SHARED DATA CHECK
# ==============================================================================
if 'df_condiviso' not in st.session_state:
    st.warning("⚠️ You haven't loaded the historical data!")
    st.info("Go back to the main page (Home) and load the CSV file.")
    st.stop()

df = st.session_state['df_condiviso'].copy()

# ==============================================================================
# 2. MODEL LOADING
# ==============================================================================
with st.sidebar:
    st.header("Configuration")
    model_file = st.file_uploader("Load Model (.joblib)", type=["joblib"])

model = None
if model_file:
    try:
        model = joblib.load(model_file)
        st.success("Model ready for forecast.")
    except Exception as e:
        st.error(f"Loading error: {e}")

# ==============================================================================
# 3. DATA PREPARATION (DECEMBER 2012 FILTER)
# ==============================================================================
col_data = df.columns[0] # Assume the first column is the date
try:
    df[col_data] = pd.to_datetime(df[col_data])
except Exception as e:
    st.error(f"Unable to convert '{col_data}' to date.")
    st.stop()

# Required range
start_date = "2012-12-01"
end_date = "2013-01-01"

# Filter the dataset for the test month
mask = (df[col_data] >= start_date) & (df[col_data] < end_date)
df_test = df.loc[mask].sort_values(by=col_data).reset_index(drop=True)

if df_test.empty:
    st.error("No data found in December 2012 range.")
    st.stop()

# ==============================================================================
# 4. INTERFACE: TIME SLIDER
# ==============================================================================
if model is not None:
    st.subheader("🗓️ Select the prediction moment")

    # Formatted date list
    opzioni_date = df_test[col_data].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()

    # Slider
    data_selezionata_str = st.select_slider(
        "Move through time:",
        options=opzioni_date,
        value=opzioni_date[0] # Start from the beginning
    )

    # Find the row corresponding to the selection
    riga_selezionata = df_test[df_test[col_data].astype(str) == data_selezionata_str]
    
    if not riga_selezionata.empty:
        # ==========================================================================
        # 5. PREDICTION EXECUTION
        # ==========================================================================

        # Model input handling
        if hasattr(model, 'feature_names_in_'):
            cols_modello = model.feature_names_in_

            # Verify columns
            missing = [c for c in cols_modello if c not in riga_selezionata.columns]

            if not missing:
                # Input data for the model
                X_input = riga_selezionata[cols_modello]

                # PREDICTION
                predizione = model.predict(X_input)[0]

                # --- KPI DISPLAY ---
                st.markdown("### Forecast Result")
                c1, c2, c3 = st.columns(3)
                c1.info(f"📅 Date: {data_selezionata_str}")
                # Show the prediction prominently
                c2.metric("Predicted Rentals (cnt)", f"{predizione:.0f}")

                # Show a key input parameter (e.g., Temp) for context
                temp_val = riga_selezionata['temp'].values[0] if 'temp' in riga_selezionata.columns else 0
                c3.metric("Input Temperature", f"{temp_val:.2f}")

                st.markdown("---")

                # ==========================================================================
                # 6. SIMULATION CHART (HISTORICAL + PREDICTION)
                # ==========================================================================

                # Take historical data UP TO the selected date (excluding the future)
                # This simulates not knowing what happens after
                data_corrente_dt = pd.to_datetime(data_selezionata_str)
                df_history = df_test[df_test[col_data] <= data_corrente_dt]

                fig = go.Figure()

                # 1. HISTORICAL line (what has "already happened" until now)
                fig.add_trace(go.Scatter(
                    x=df_history[col_data],
                    y=df_history['cnt'],
                    mode='lines',
                    name='Acquired Historical',
                    line=dict(color='rgba(0,100,250, 0.5)', width=2)
                ))

                # 2. CURRENT PREDICTION point
                fig.add_trace(go.Scatter(
                    x=[data_corrente_dt],
                    y=[predizione],
                    mode='markers',
                    name='Model Forecast',
                    marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black'))
                ))

                fig.update_layout(
                    title="Prediction Monitoring",
                    xaxis_title="Time",
                    yaxis_title="cnt Value",
                    xaxis_range=[pd.to_datetime(start_date), pd.to_datetime(end_date)], # Fixed X axis for the whole month
                    yaxis_range=[0, df_test['cnt'].max() * 1.2], # Fixed Y axis for stability
                    showlegend=True
                )

                st.plotly_chart(fig, width='stretch')

            else:
                st.error(f"Missing columns in CSV: {missing}")
        else:
            st.error("The model does not have saved feature names.")

else:
    st.info("👈 Load the model in the sidebar to start the simulation.")