import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page configuration
st.set_page_config(page_title="Model Comparison", layout="wide")

# Title
st.title("📊 Model Performance Comparison")
st.markdown("Compare multiple ML models using standard metrics (RMSE, MAE, R², MAPE)")

# ==============================================================================
# 1. SHARED DATA CHECK
# ==============================================================================
if 'df_condiviso' not in st.session_state:
    st.warning("⚠️ You haven't loaded the historical data!")
    st.info("Go to the **Data Analysis** page first and load the CSV file.")
    st.stop()

df = st.session_state['df_condiviso'].copy()

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
# Set datetime index
col_data = df.columns[0]
try:
    df[col_data] = pd.to_datetime(df[col_data])
    df.index = df[col_data]
    df = df.iloc[:, 1:]  # Remove first column to avoid duplication
except Exception as e:
    st.error(f"Unable to convert '{col_data}' to date: {e}")
    st.stop()

# Filter to December 2012 test set
start_date = "2012-12-01"
end_date = "2013-01-01"
mask = (df.index >= start_date) & (df.index < end_date)
df_test = df.loc[mask].copy()

if df_test.empty:
    st.error("No data found in December 2012 range for testing.")
    st.stop()

st.success(f"✅ Test data loaded: {len(df_test)} records from December 2012")

# ==============================================================================
# 3. MODEL LOADING
# ==============================================================================
with st.sidebar:
    st.header("Load Models")
    st.markdown("Upload one or more `.joblib` model files")
    model_files = st.file_uploader(
        "Select model files",
        type=["joblib"],
        accept_multiple_files=True,
        help="Upload multiple models to compare their performance"
    )

# ==============================================================================
# 4. METRICS CALCULATION FUNCTIONS
# ==============================================================================
def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""
    # Handle potential division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': mape
    }

# ==============================================================================
# 5. MODEL COMPARISON
# ==============================================================================
if model_files:
    st.subheader("🔄 Model Evaluation")

    models = {}
    predictions = {}
    results = []

    # Load each model and generate predictions
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_file in enumerate(model_files):
        model_name = model_file.name.replace('.joblib', '')
        status_text.text(f"Loading {model_name}...")

        try:
            model = joblib.load(model_file)
            models[model_name] = model

            # Get feature names from model
            if hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_

                # Check for missing columns
                missing = [c for c in feature_cols if c not in df_test.columns]
                if missing:
                    st.warning(f"⚠️ {model_name}: Missing {len(missing)} columns")
                    continue

                # Prepare input data
                X_test = df_test[feature_cols]
                y_true = df_test['cnt'].values

                # Generate predictions
                y_pred = model.predict(X_test)
                predictions[model_name] = y_pred

                # Calculate metrics
                metrics = calculate_metrics(y_true, y_pred)
                metrics['Model'] = model_name
                results.append(metrics)

            else:
                st.warning(f"⚠️ {model_name}: No feature_names_in_ attribute found")

        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {e}")

        progress_bar.progress((i + 1) / len(model_files))

    status_text.empty()
    progress_bar.empty()

    # ==============================================================================
    # 6. RESULTS DISPLAY
    # ==============================================================================
    if results:
        df_results = pd.DataFrame(results)

        # Reorder columns
        cols_order = ['Model', 'RMSE', 'MAE', 'R²', 'MAPE (%)']
        df_results = df_results[cols_order]

        # Display metrics table
        st.subheader("📈 Performance Metrics Comparison")

        # Style the dataframe
        def highlight_best(s):
            if s.name in ['RMSE', 'MAE', 'MAPE (%)']:
                is_best = s == s.min()
            elif s.name == 'R²':
                is_best = s == s.max()
            else:
                return [''] * len(s)
            return ['background-color: #90EE90' if v else '' for v in is_best]

        styled_df = df_results.style.apply(highlight_best).format({
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R²': '{:.4f}',
            'MAPE (%)': '{:.2f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Best model summary
        best_rmse = df_results.loc[df_results['RMSE'].idxmin(), 'Model']
        best_r2 = df_results.loc[df_results['R²'].idxmax(), 'Model']

        col1, col2 = st.columns(2)
        col1.success(f"🏆 **Best RMSE:** {best_rmse}")
        col2.success(f"🏆 **Best R²:** {best_r2}")

        # ==============================================================================
        # 7. VISUALIZATIONS
        # ==============================================================================
        st.markdown("---")
        st.subheader("📊 Visual Comparisons")

        tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "📈 Prediction Overlay", "📉 Error Distribution"])

        with tab1:
            # Grouped bar chart of metrics
            df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Value')

            # Separate plots for different scales
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                df_error = df_melted[df_melted['Metric'].isin(['RMSE', 'MAE'])]
                fig_error = px.bar(
                    df_error,
                    x='Metric',
                    y='Value',
                    color='Model',
                    barmode='group',
                    title='Error Metrics (Lower is Better)',
                    text_auto='.2f'
                )
                fig_error.update_layout(template="plotly_white")
                st.plotly_chart(fig_error, width='stretch')

            with col_m2:
                df_perf = df_melted[df_melted['Metric'] == 'R²']
                fig_perf = px.bar(
                    df_perf,
                    x='Metric',
                    y='Value',
                    color='Model',
                    barmode='group',
                    title='R² Score (Higher is Better)',
                    text_auto='.4f'
                )
                fig_perf.update_layout(template="plotly_white")
                st.plotly_chart(fig_perf, width='stretch')

        with tab2:
            # Prediction overlay chart
            fig_pred = go.Figure()

            # Add actual values
            fig_pred.add_trace(go.Scatter(
                x=df_test.index,
                y=df_test['cnt'],
                name='Actual',
                mode='lines',
                line=dict(color='black', width=2)
            ))

            # Add predictions for each model
            colors = px.colors.qualitative.Set1
            for i, (model_name, y_pred) in enumerate(predictions.items()):
                fig_pred.add_trace(go.Scatter(
                    x=df_test.index,
                    y=y_pred,
                    name=f'{model_name} (Predicted)',
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
                    opacity=0.8
                ))

            fig_pred.update_layout(
                title='Actual vs Predicted Values (December 2012)',
                xaxis_title='Date',
                yaxis_title='Bike Rentals (cnt)',
                template='plotly_white',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pred, width='stretch')

        with tab3:
            # Error distribution
            y_true = df_test['cnt'].values

            fig_errors = go.Figure()
            for model_name, y_pred in predictions.items():
                errors = y_true - y_pred
                fig_errors.add_trace(go.Histogram(
                    x=errors,
                    name=model_name,
                    opacity=0.7,
                    nbinsx=50
                ))

            fig_errors.update_layout(
                title='Prediction Error Distribution',
                xaxis_title='Error (Actual - Predicted)',
                yaxis_title='Frequency',
                template='plotly_white',
                barmode='overlay'
            )
            st.plotly_chart(fig_errors, width='stretch')

        # ==============================================================================
        # 8. SUMMARY INSIGHTS
        # ==============================================================================
        st.markdown("---")
        st.subheader("💡 Model Comparison Insights")

        # Find best and worst
        best_model = df_results.loc[df_results['R²'].idxmax()]

        st.info(f"""
        **Best Overall Model: {best_model['Model']}**

        | Metric | Value |
        |--------|-------|
        | RMSE | {best_model['RMSE']:.2f} |
        | MAE | {best_model['MAE']:.2f} |
        | R² | {best_model['R²']:.4f} |
        | MAPE | {best_model['MAPE (%)']:.2f}% |

        *The model explains **{best_model['R²']*100:.1f}%** of the variance in bike rental demand.*
        """)

    else:
        st.warning("No models were successfully loaded and evaluated.")
else:
    st.info("👈 Upload one or more model files (.joblib) in the sidebar to start comparison.")

    st.markdown("""
    ### How to Use This Page

    1. **Load Data First**: Make sure you've loaded the CSV file in the Data Analysis page
    2. **Upload Models**: Use the sidebar to upload `.joblib` model files
    3. **View Comparison**: See metrics, prediction charts, and error distributions

    ### Available Metrics

    | Metric | Description | Better When |
    |--------|-------------|-------------|
    | **RMSE** | Root Mean Square Error | Lower |
    | **MAE** | Mean Absolute Error | Lower |
    | **R²** | Coefficient of Determination | Higher (closer to 1) |
    | **MAPE** | Mean Absolute Percentage Error | Lower |
    """)
