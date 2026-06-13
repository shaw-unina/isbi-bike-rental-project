import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Advanced Analysis", layout="wide")

# Title
st.title("🔬 Advanced Time Series Analysis")
st.markdown("Decomposition, Autocorrelation, and PCA Analysis")

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
col_data = df.columns[0]
try:
    df[col_data] = pd.to_datetime(df[col_data])
    df.index = df[col_data]
    df = df.iloc[:, 1:]
except Exception as e:
    st.error(f"Unable to convert '{col_data}' to date: {e}")
    st.stop()

st.success(f"✅ Data loaded: {len(df)} records")

# ==============================================================================
# 3. TIME SERIES DECOMPOSITION
# ==============================================================================
st.header("📈 Time Series Decomposition")
st.markdown("Decompose the time series into Trend, Seasonal, and Residual components")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose

    # First aggregate to daily data for cleaner decomposition
    df_daily = df.groupby(df.index.date)['cnt'].sum()
    df_daily.index = pd.to_datetime(df_daily.index)

    # User options
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        decomp_model = st.radio(
            "Decomposition Model",
            ["Additive", "Multiplicative"],
            horizontal=True,
            help="Additive: Y = Trend + Seasonal + Residual\nMultiplicative: Y = Trend × Seasonal × Residual"
        )
    with col_d2:
        period = st.selectbox(
            "Seasonal Period",
            [7, 30, 365],
            format_func=lambda x: f"Weekly ({x} days)" if x == 7 else f"Monthly ({x} days)" if x == 30 else f"Yearly ({x} days)",
            help="Select the expected seasonal period in days"
        )

    # Perform decomposition
    if len(df_daily) >= period * 2:
        result = seasonal_decompose(df_daily, model=decomp_model.lower(), period=period)

        # Create 4-panel subplot
        fig_decomp = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component'],
            vertical_spacing=0.08
        )

        # Original
        fig_decomp.add_trace(
            go.Scatter(x=df_daily.index, y=result.observed, name='Original', line=dict(color='#636EFA')),
            row=1, col=1
        )

        # Trend
        fig_decomp.add_trace(
            go.Scatter(x=df_daily.index, y=result.trend, name='Trend', line=dict(color='#EF553B')),
            row=2, col=1
        )

        # Seasonal
        fig_decomp.add_trace(
            go.Scatter(x=df_daily.index, y=result.seasonal, name='Seasonal', line=dict(color='#00CC96')),
            row=3, col=1
        )

        # Residual
        fig_decomp.add_trace(
            go.Scatter(x=df_daily.index, y=result.resid, name='Residual', mode='markers',
                      marker=dict(color='#AB63FA', size=3)),
            row=4, col=1
        )

        fig_decomp.update_layout(
            height=800,
            title_text=f"Time Series Decomposition ({decomp_model}, Period={period} days)",
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_decomp, width='stretch')

        # Decomposition metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Trend Range", f"{result.trend.max() - result.trend.min():.0f}")
        col_m2.metric("Seasonal Amplitude", f"{result.seasonal.max() - result.seasonal.min():.0f}")
        col_m3.metric("Residual Std", f"{result.resid.std():.0f}")

        st.info("""
        💡 **Interpretation:**
        - **Trend**: Shows the overall direction of bike rentals over time
        - **Seasonal**: Captures repeating patterns (weekly/monthly cycles)
        - **Residual**: The unexplained variation after removing trend and seasonality
        """)
    else:
        st.warning(f"Need at least {period * 2} days of data for decomposition with period={period}")

except ImportError:
    st.error("statsmodels not installed. Run: pip install statsmodels")

# ==============================================================================
# 4. AUTOCORRELATION ANALYSIS
# ==============================================================================
st.markdown("---")
st.header("🔄 Autocorrelation Analysis")
st.markdown("Identify temporal dependencies and lag patterns in the data")

try:
    from statsmodels.tsa.stattools import acf, pacf

    # User options
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        max_lags = st.slider("Maximum Lags", min_value=24, max_value=168, value=48,
                            help="Number of time lags to analyze")
    with col_a2:
        data_source = st.radio(
            "Data Granularity",
            ["Hourly", "Daily"],
            horizontal=True
        )

    # Prepare data based on selection
    if data_source == "Hourly":
        series = df['cnt'].dropna()
        lag_label = "hours"
    else:
        series = df_daily.dropna()
        lag_label = "days"

    # Calculate ACF and PACF
    acf_values = acf(series, nlags=max_lags)
    pacf_values = pacf(series, nlags=max_lags, method='ols')

    # Confidence interval (95%)
    conf_int = 1.96 / np.sqrt(len(series))

    # Create side-by-side plots
    col_acf, col_pacf = st.columns(2)

    with col_acf:
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=list(range(max_lags + 1)),
            y=acf_values,
            name='ACF',
            marker_color='#636EFA'
        ))
        # Confidence bounds
        fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
        fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
        fig_acf.add_hline(y=0, line_color="black", line_width=0.5)

        fig_acf.update_layout(
            title=f"ACF - Autocorrelation Function ({lag_label})",
            xaxis_title=f"Lag ({lag_label})",
            yaxis_title="Correlation",
            template="plotly_white",
            yaxis_range=[-0.5, 1.1]
        )
        st.plotly_chart(fig_acf, width='stretch')

    with col_pacf:
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(
            x=list(range(max_lags + 1)),
            y=pacf_values,
            name='PACF',
            marker_color='#EF553B'
        ))
        # Confidence bounds
        fig_pacf.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
        fig_pacf.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
        fig_pacf.add_hline(y=0, line_color="black", line_width=0.5)

        fig_pacf.update_layout(
            title=f"PACF - Partial Autocorrelation Function ({lag_label})",
            xaxis_title=f"Lag ({lag_label})",
            yaxis_title="Partial Correlation",
            template="plotly_white",
            yaxis_range=[-0.5, 1.1]
        )
        st.plotly_chart(fig_pacf, width='stretch')

    # Identify significant lags
    significant_acf = [i for i, v in enumerate(acf_values) if abs(v) > conf_int and i > 0]
    significant_pacf = [i for i, v in enumerate(pacf_values) if abs(v) > conf_int and i > 0]

    st.markdown("#### 📊 Significant Lags")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.write(f"**ACF Significant Lags:** {significant_acf[:10]}..." if len(significant_acf) > 10 else f"**ACF Significant Lags:** {significant_acf}")

    with col_s2:
        st.write(f"**PACF Significant Lags:** {significant_pacf[:10]}..." if len(significant_pacf) > 10 else f"**PACF Significant Lags:** {significant_pacf}")

    st.info("""
    💡 **Key Findings:**
    - **Lag 1**: Strong immediate correlation (yesterday/last hour affects today/this hour)
    - **Lag 24** (hourly): Daily cycle - same hour yesterday correlates strongly
    - **Lag 168** (hourly) / **Lag 7** (daily): Weekly pattern - same day last week

    *Red dashed lines indicate 95% confidence bounds. Bars outside these bounds are statistically significant.*
    """)

except ImportError:
    st.error("statsmodels not installed. Run: pip install statsmodels")

# ==============================================================================
# 5. PCA ANALYSIS
# ==============================================================================
st.markdown("---")
st.header("🎯 Principal Component Analysis")
st.markdown("Reduce dimensionality and identify key drivers of variance")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Select numeric columns for PCA
    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed']
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) >= 2:
        st.write(f"**Features for PCA:** {available_cols}")

        # Prepare data
        X_pca = df[available_cols].dropna()

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pca)

        # Fit PCA
        pca = PCA()
        pca.fit(X_scaled)

        # Results
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Scree plot
        fig_pca = go.Figure()

        # Individual variance bars
        fig_pca.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(explained_var))],
            y=explained_var * 100,
            name='Individual',
            marker_color='#636EFA',
            text=[f'{v*100:.1f}%' for v in explained_var],
            textposition='auto'
        ))

        # Cumulative variance line
        fig_pca.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumulative_var))],
            y=cumulative_var * 100,
            name='Cumulative',
            mode='lines+markers',
            line=dict(color='#EF553B', width=3),
            marker=dict(size=10)
        ))

        fig_pca.update_layout(
            title="PCA Explained Variance",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained (%)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_pca, width='stretch')

        # Metrics matching Slide 15
        col_pca1, col_pca2, col_pca3, col_pca4 = st.columns(4)
        col_pca1.metric("PC1 Variance", f"{explained_var[0]*100:.1f}%")
        col_pca2.metric("PC2 Variance", f"{explained_var[1]*100:.1f}%" if len(explained_var) > 1 else "N/A")
        col_pca3.metric("PC3 Variance", f"{explained_var[2]*100:.1f}%" if len(explained_var) > 2 else "N/A")
        col_pca4.metric("Cumulative (3 PCs)", f"{cumulative_var[min(2, len(cumulative_var)-1)]*100:.1f}%")

        # Feature loadings
        st.markdown("#### 📊 Feature Loadings (Component Contributions)")

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(explained_var))],
            index=available_cols
        )

        # Heatmap of loadings
        fig_loadings = px.imshow(
            loadings.values,
            x=loadings.columns.tolist(),
            y=loadings.index.tolist(),
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        fig_loadings.update_layout(
            title="Feature Contributions to Principal Components",
            template="plotly_white"
        )
        st.plotly_chart(fig_loadings, width='stretch')

        # Prepare values for display
        pc1_var = f"{explained_var[0]*100:.1f}%"
        pc2_var = f"{explained_var[1]*100:.1f}%" if len(explained_var) > 1 else "N/A"
        pc3_var = f"{explained_var[2]*100:.1f}%" if len(explained_var) > 2 else "N/A"
        cum_var = f"{cumulative_var[min(2, len(cumulative_var)-1)]*100:.1f}%"

        st.info(f"""
        💡 **Interpretation (matching Slide 15):**
        - **PC1** ({pc1_var}): Primarily captures temperature variations (temp, atemp)
        - **PC2** ({pc2_var}): Captures humidity and wind patterns
        - **PC3** ({pc3_var}): Remaining weather variation

        *With 3 components, we explain **{cum_var}** of variance in weather features.*
        """)

    else:
        st.warning(f"Need at least 2 numeric columns for PCA. Found: {available_cols}")

except ImportError:
    st.error("scikit-learn not installed. Run: pip install scikit-learn")

# ==============================================================================
# 6. STATIONARITY TEST
# ==============================================================================
st.markdown("---")
st.header("📉 Stationarity Test")

try:
    from statsmodels.tsa.stattools import adfuller

    # Test on daily data
    adf_result = adfuller(df_daily.dropna())

    col_adf1, col_adf2, col_adf3 = st.columns(3)
    col_adf1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
    col_adf2.metric("P-Value", f"{adf_result[1]:.4f}")
    col_adf3.metric("Critical (5%)", f"{adf_result[4]['5%']:.4f}")

    if adf_result[1] < 0.05:
        st.success("✅ **Result: STATIONARY** - The null hypothesis (unit root) is rejected at 5% significance level.")
    else:
        st.warning("⚠️ **Result: NON-STATIONARY** - The series has a unit root (trend/seasonality present).")

    st.info("""
    💡 **Augmented Dickey-Fuller Test:**
    - Tests the null hypothesis that a unit root is present (non-stationary)
    - P-value < 0.05: Reject null → Series is stationary
    - P-value > 0.05: Fail to reject → Series is non-stationary

    *As noted in Slide 8: Working with non-stationary data can be acceptable for forecasting,
    but stationarity helps isolate seasonality and trends.*
    """)

except ImportError:
    st.error("statsmodels not installed. Run: pip install statsmodels")
