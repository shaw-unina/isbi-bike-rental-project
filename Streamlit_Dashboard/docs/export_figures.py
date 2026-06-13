"""
Export all dashboard figures for LaTeX report.

This script generates all visualizations from the Streamlit dashboard
and saves them as high-quality images for use in reports.

Requirements:
    pip install kaleido  # For Plotly image export

Usage:
    python export_figures.py

Output:
    All figures saved to ./figures/ directory
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import joblib

# Try to import kaleido for image export
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("⚠️  kaleido not installed. Install with: pip install -U kaleido")
    print("   Figures will be saved as interactive HTML instead of PNG.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Dataset_Bike.csv")

# Image settings
IMG_WIDTH = 1200
IMG_HEIGHT = 600
IMG_SCALE = 2  # For higher resolution

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================
print("=" * 60)
print("FIGURE EXPORT FOR LATEX REPORT")
print("=" * 60)

print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df.index = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]
print(f"   Loaded {len(df)} records")

# Daily aggregation
df_daily = df.groupby(df.index.date)[['registered', 'casual', 'cnt']].sum()
df_daily.index = pd.to_datetime(df_daily.index)
df_daily['kpi_efficiency_rate'] = (df_daily['registered'] / df_daily['cnt']) * 100

# ==============================================================================
# HELPER FUNCTION
# ==============================================================================
def save_figure(fig, filename, width=IMG_WIDTH, height=IMG_HEIGHT):
    """Save figure as PNG (if kaleido available) or HTML."""
    filepath_png = os.path.join(FIGURES_DIR, f"{filename}.png")
    filepath_html = os.path.join(FIGURES_DIR, f"{filename}.html")

    # Update layout for export
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=14),
        template="plotly_white"
    )

    if KALEIDO_AVAILABLE:
        fig.write_image(filepath_png, scale=IMG_SCALE)
        print(f"   ✅ Saved: {filename}.png")
    else:
        fig.write_html(filepath_html)
        print(f"   ✅ Saved: {filename}.html")

    return filepath_png if KALEIDO_AVAILABLE else filepath_html

# ==============================================================================
# FIGURE 1: EFFICIENCY TREND
# ==============================================================================
print("\n📊 Generating Figure 1: Efficiency Trend...")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily['kpi_efficiency_rate'],
    mode='lines',
    name='% Registered',
    line=dict(color='#00CC96', width=2)
))
fig1.add_hline(y=df_daily['kpi_efficiency_rate'].mean(), line_dash="dot",
               annotation_text="Average", line_color="red")
fig1.update_layout(
    title="Business Efficiency Trend (Daily % Registered Users)",
    xaxis_title="Date",
    yaxis_title="% Registered"
)
save_figure(fig1, "01_efficiency_trend")

# ==============================================================================
# FIGURE 2: SUBSCRIBERS VS CASUAL - STACKED AREA
# ==============================================================================
print("📊 Generating Figure 2: Subscribers vs Casual (Stacked Area)...")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily['registered'],
    fill='tozeroy',
    name='Registered',
    line=dict(color='#636EFA'),
    fillcolor='rgba(99, 110, 250, 0.6)'
))
fig2.add_trace(go.Scatter(
    x=df_daily.index,
    y=df_daily['registered'] + df_daily['casual'],
    fill='tonexty',
    name='Casual',
    line=dict(color='#EF553B'),
    fillcolor='rgba(239, 85, 59, 0.6)'
))
fig2.update_layout(
    title="Daily User Composition: Registered vs Casual",
    xaxis_title="Date",
    yaxis_title="Number of Rentals"
)
save_figure(fig2, "02_subscribers_vs_casual_area")

# ==============================================================================
# FIGURE 3: MONTHLY USER DISTRIBUTION
# ==============================================================================
print("📊 Generating Figure 3: Monthly User Distribution...")

df_monthly = df.copy()
df_monthly['month'] = df_monthly.index.to_period('M').astype(str)
monthly_users = df_monthly.groupby('month')[['registered', 'casual']].sum().reset_index()

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=monthly_users['month'], y=monthly_users['registered'],
                      name='Registered', marker_color='#636EFA'))
fig3.add_trace(go.Bar(x=monthly_users['month'], y=monthly_users['casual'],
                      name='Casual', marker_color='#EF553B'))
fig3.update_layout(
    title="Monthly User Distribution",
    barmode='group',
    xaxis_title="Month",
    yaxis_title="Total Rentals"
)
save_figure(fig3, "03_monthly_user_distribution")

# ==============================================================================
# FIGURE 4: WEATHER IMPACT BY YEAR
# ==============================================================================
print("📊 Generating Figure 4: Weather Impact (2011 vs 2012)...")

# Prepare weather data
cols_meteo_ohe = ['weathersit_1.0', 'weathersit_2.0', 'weathersit_3.0', 'weathersit_4.0']
df_meteo = df.copy()
df_meteo['Year'] = df_meteo.index.year
df_meteo['Meteo_Raw'] = df_meteo[cols_meteo_ohe].idxmax(axis=1)
mapping_nomi = {
    'weathersit_1.0': '1. Sunny/Clear',
    'weathersit_2.0': '2. Cloudy',
    'weathersit_3.0': '3. Light Rain',
    'weathersit_4.0': '4. Storm'
}
df_meteo['Meteo_Label'] = df_meteo['Meteo_Raw'].map(mapping_nomi)
df_weather_comp = df_meteo.groupby(['Year', 'Meteo_Label'])['cnt'].mean().reset_index()
df_2011 = df_weather_comp[df_weather_comp['Year'] == 2011]
df_2012 = df_weather_comp[df_weather_comp['Year'] == 2012]

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=df_2011['Meteo_Label'], y=df_2011['cnt'], name='2011',
                      marker_color='rgba(99, 110, 250, 0.7)'))
fig4.add_trace(go.Bar(x=df_2012['Meteo_Label'], y=df_2012['cnt'], name='2012',
                      marker_color='rgba(239, 85, 59, 0.7)'))
fig4.update_layout(
    title="Weather Impact on Bike Rentals: 2011 vs 2012",
    xaxis_title="Weather Condition",
    yaxis_title="Average Rentals",
    barmode='group'
)
save_figure(fig4, "04_weather_impact_yearly")

# ==============================================================================
# FIGURE 5: CLIMATE IMPACT BY USER TYPE
# ==============================================================================
print("📊 Generating Figure 5: Climate Impact by User Type...")

df_weather_users = df_meteo.groupby('Meteo_Label')[['registered', 'casual']].mean().reset_index()
sunny_reg = df_weather_users[df_weather_users['Meteo_Label'].str.contains('1.')]['registered'].values[0]
storm_reg = df_weather_users[df_weather_users['Meteo_Label'].str.contains('4.')]['registered'].values[0]
sunny_cas = df_weather_users[df_weather_users['Meteo_Label'].str.contains('1.')]['casual'].values[0]
storm_cas = df_weather_users[df_weather_users['Meteo_Label'].str.contains('4.')]['casual'].values[0]

fig5 = go.Figure()
fig5.add_trace(go.Bar(x=['Sunny', 'Storm'], y=[sunny_reg, storm_reg], name='Registered',
                      marker_color='#636EFA', text=[f'{sunny_reg:.0f}', f'{storm_reg:.0f}'], textposition='auto'))
fig5.add_trace(go.Bar(x=['Sunny', 'Storm'], y=[sunny_cas, storm_cas], name='Casual',
                      marker_color='#EF553B', text=[f'{sunny_cas:.0f}', f'{storm_cas:.0f}'], textposition='auto'))
fig5.update_layout(
    title="Weather Impact: Registered vs Casual Users",
    xaxis_title="Weather Condition",
    yaxis_title="Average Rentals",
    barmode='group'
)
save_figure(fig5, "05_climate_impact_user_type")

# ==============================================================================
# FIGURE 6: PEAK HOURS HEATMAP
# ==============================================================================
print("📊 Generating Figure 6: Peak Hours Heatmap...")

df_heatmap = df.copy()
df_heatmap['dayofweek'] = df_heatmap.index.dayofweek
heatmap_data = df_heatmap.pivot_table(values='cnt', index='hr', columns='dayofweek', aggfunc='mean')
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig6 = px.imshow(
    heatmap_data,
    labels=dict(x="Day of Week", y="Hour", color="Avg Rentals"),
    x=day_labels,
    y=list(range(24)),
    aspect="auto",
    color_continuous_scale='YlOrRd'
)
fig6.update_layout(title="Average Bike Rentals: Hour vs Day of Week")
save_figure(fig6, "06_peak_hours_heatmap", height=700)

# ==============================================================================
# FIGURE 7: WORKING DAY VS NON-WORKING DAY
# ==============================================================================
print("📊 Generating Figure 7: Working Day vs Non-Working Day Comparison...")

df_working = df[df['workingday'] == 1].groupby('hr')['cnt'].mean()
df_nonworking = df[df['workingday'] == 0].groupby('hr')['cnt'].mean()

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=df_working.index, y=df_working.values, name='Working Days',
                          mode='lines+markers', line=dict(color='#636EFA', width=3), marker=dict(size=8)))
fig7.add_trace(go.Scatter(x=df_nonworking.index, y=df_nonworking.values, name='Non-Working Days',
                          mode='lines+markers', line=dict(color='#EF553B', width=3), marker=dict(size=8)))
fig7.add_vrect(x0=6, x1=9, fillcolor="rgba(0,100,80,0.1)", layer="below", line_width=0,
               annotation_text="Morning Rush", annotation_position="top left")
fig7.add_vrect(x0=16, x1=19, fillcolor="rgba(0,100,80,0.1)", layer="below", line_width=0,
               annotation_text="Evening Rush", annotation_position="top left")
fig7.update_layout(
    title="Hourly Rental Patterns: Working vs Non-Working Days",
    xaxis_title="Hour of Day",
    yaxis_title="Average Rentals",
    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
)
save_figure(fig7, "07_working_vs_nonworking")

# ==============================================================================
# FIGURE 8: BOX PLOT BY HOUR
# ==============================================================================
print("📊 Generating Figure 8: Box Plot by Hour...")

fig8 = px.box(df, x='hr', y='cnt', color='hr', title="Distribution of Rentals by Hour")
fig8.update_layout(showlegend=False, xaxis_title="Hour", yaxis_title="Rentals (cnt)")
save_figure(fig8, "08_boxplot_hour", height=500)

# ==============================================================================
# FIGURE 9: SCATTER TEMP VS CNT
# ==============================================================================
print("📊 Generating Figure 9: Temperature vs Rentals...")

df_temp_avg = df.groupby(pd.cut(df['temp'], bins=20))['cnt'].mean().reset_index()
df_temp_avg.columns = ['temp_bin', 'avg_cnt']
df_temp_avg['temp_mid'] = df_temp_avg['temp_bin'].apply(lambda x: x.mid if pd.notna(x) else None)

fig9 = px.scatter(df.sample(5000, random_state=42), x='temp', y='cnt', color='cnt',
                  opacity=0.4, title="Temperature vs Bike Rentals")
fig9.update_layout(xaxis_title="Normalized Temperature", yaxis_title="Rentals (cnt)")
save_figure(fig9, "09_temp_vs_rentals")

# ==============================================================================
# FIGURE 10: TIME SERIES DECOMPOSITION
# ==============================================================================
print("📊 Generating Figure 10: Time Series Decomposition...")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(df_daily['cnt'], model='additive', period=7)

    fig10 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                          vertical_spacing=0.08)
    fig10.add_trace(go.Scatter(x=df_daily.index, y=result.observed, name='Original',
                               line=dict(color='#636EFA')), row=1, col=1)
    fig10.add_trace(go.Scatter(x=df_daily.index, y=result.trend, name='Trend',
                               line=dict(color='#EF553B')), row=2, col=1)
    fig10.add_trace(go.Scatter(x=df_daily.index, y=result.seasonal, name='Seasonal',
                               line=dict(color='#00CC96')), row=3, col=1)
    fig10.add_trace(go.Scatter(x=df_daily.index, y=result.resid, name='Residual',
                               mode='markers', marker=dict(color='#AB63FA', size=3)), row=4, col=1)
    fig10.update_layout(title="Time Series Decomposition (Additive, Period=7 days)",
                        showlegend=False, height=800)
    save_figure(fig10, "10_decomposition", height=800)
except ImportError:
    print("   ⚠️ statsmodels not available for decomposition")

# ==============================================================================
# FIGURE 11: ACF PLOT
# ==============================================================================
print("📊 Generating Figure 11: ACF Plot...")

try:
    from statsmodels.tsa.stattools import acf, pacf

    max_lags = 48
    acf_values = acf(df_daily['cnt'].dropna(), nlags=max_lags)
    conf_int = 1.96 / np.sqrt(len(df_daily))

    fig11 = go.Figure()
    fig11.add_trace(go.Bar(x=list(range(max_lags + 1)), y=acf_values, name='ACF', marker_color='#636EFA'))
    fig11.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
    fig11.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
    fig11.add_hline(y=0, line_color="black", line_width=0.5)
    fig11.update_layout(
        title="Autocorrelation Function (ACF) - Daily Data",
        xaxis_title="Lag (days)",
        yaxis_title="Correlation",
        yaxis_range=[-0.5, 1.1]
    )
    save_figure(fig11, "11_acf_plot")
except ImportError:
    print("   ⚠️ statsmodels not available for ACF")

# ==============================================================================
# FIGURE 12: PACF PLOT
# ==============================================================================
print("📊 Generating Figure 12: PACF Plot...")

try:
    pacf_values = pacf(df_daily['cnt'].dropna(), nlags=max_lags, method='ols')

    fig12 = go.Figure()
    fig12.add_trace(go.Bar(x=list(range(max_lags + 1)), y=pacf_values, name='PACF', marker_color='#EF553B'))
    fig12.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
    fig12.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
    fig12.add_hline(y=0, line_color="black", line_width=0.5)
    fig12.update_layout(
        title="Partial Autocorrelation Function (PACF) - Daily Data",
        xaxis_title="Lag (days)",
        yaxis_title="Partial Correlation",
        yaxis_range=[-0.5, 1.1]
    )
    save_figure(fig12, "12_pacf_plot")
except ImportError:
    print("   ⚠️ statsmodels not available for PACF")

# ==============================================================================
# FIGURE 13: PCA SCREE PLOT
# ==============================================================================
print("📊 Generating Figure 13: PCA Scree Plot...")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    numeric_cols = ['temp', 'atemp', 'hum', 'windspeed']
    X_pca = df[numeric_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    pca = PCA()
    pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig13 = go.Figure()
    fig13.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))],
                           y=explained_var * 100, name='Individual', marker_color='#636EFA',
                           text=[f'{v*100:.1f}%' for v in explained_var], textposition='auto'))
    fig13.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(cumulative_var))],
                               y=cumulative_var * 100, name='Cumulative', mode='lines+markers',
                               line=dict(color='#EF553B', width=3), marker=dict(size=10)))
    fig13.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)"
    )
    save_figure(fig13, "13_pca_scree_plot")
except ImportError:
    print("   ⚠️ sklearn not available for PCA")

# ==============================================================================
# FIGURE 14: FEATURE IMPORTANCE (REALISTIC MODEL)
# ==============================================================================
print("📊 Generating Figure 14: Feature Importance...")

# Feature importance data from realistic model training
feature_importance = {
    'hr': 0.4877,
    'temp': 0.1209,
    'workingday': 0.0972,
    'Orario_punta': 0.0884,
    'yr': 0.0881,
    'hum': 0.0264,
    'season': 0.0230,
    'atemp': 0.0184,
    'weathersit_3.0': 0.0134,
    'windspeed': 0.0073
}

fig14 = go.Figure()
fig14.add_trace(go.Bar(
    x=list(feature_importance.values()),
    y=list(feature_importance.keys()),
    orientation='h',
    marker_color='#636EFA',
    text=[f'{v*100:.1f}%' for v in feature_importance.values()],
    textposition='auto'
))
fig14.update_layout(
    title="Random Forest Feature Importance (Realistic Model)",
    xaxis_title="Importance",
    yaxis_title="Feature",
    yaxis=dict(autorange="reversed")
)
save_figure(fig14, "14_feature_importance")

# ==============================================================================
# FIGURE 15: MODEL COMPARISON
# ==============================================================================
print("📊 Generating Figure 15: Model Comparison...")

model_data = {
    'Model': ['With Leakage', 'Realistic'],
    'R²': [0.9999, 0.9456],
    'RMSE': [1.21, 41.01],
    'MAPE': [0.27, 30.65]
}

fig15 = make_subplots(rows=1, cols=3, subplot_titles=['R² Score', 'RMSE', 'MAPE (%)'])

colors = ['#636EFA', '#EF553B']

fig15.add_trace(go.Bar(x=model_data['Model'], y=model_data['R²'], marker_color=colors,
                       text=[f'{v:.4f}' for v in model_data['R²']], textposition='auto'), row=1, col=1)
fig15.add_trace(go.Bar(x=model_data['Model'], y=model_data['RMSE'], marker_color=colors,
                       text=[f'{v:.2f}' for v in model_data['RMSE']], textposition='auto'), row=1, col=2)
fig15.add_trace(go.Bar(x=model_data['Model'], y=model_data['MAPE'], marker_color=colors,
                       text=[f'{v:.2f}%' for v in model_data['MAPE']], textposition='auto'), row=1, col=3)

fig15.update_layout(title="Model Performance Comparison: Leaky vs Realistic", showlegend=False)
save_figure(fig15, "15_model_comparison", width=1400)

# ==============================================================================
# FIGURE 16: FORECAST SIMULATION
# ==============================================================================
print("📊 Generating Figure 16: Forecast Simulation...")

try:
    # Load the realistic model
    model_path = os.path.join(SCRIPT_DIR, "..", "models", "random_forest_realistic.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)

        # Filter to December 2012
        df_forecast = pd.read_csv(DATA_PATH)
        col_data = df_forecast.columns[0]
        df_forecast[col_data] = pd.to_datetime(df_forecast[col_data])

        start_date = "2012-12-01"
        end_date = "2013-01-01"
        mask = (df_forecast[col_data] >= start_date) & (df_forecast[col_data] < end_date)
        df_test = df_forecast.loc[mask].sort_values(by=col_data).copy()

        if not df_test.empty and hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_

            # Check which features are available
            available_features = [f for f in feature_cols if f in df_test.columns]

            if len(available_features) == len(feature_cols):
                # Generate predictions for all December 2012
                X_test = df_test[feature_cols]
                predictions = model.predict(X_test)

                # Select a specific point for visualization (mid-December)
                mid_point = len(df_test) // 2
                selected_date = df_test[col_data].iloc[mid_point]

                fig16 = go.Figure()

                # Historical line (actual values)
                fig16.add_trace(go.Scatter(
                    x=df_test[col_data],
                    y=df_test['cnt'],
                    mode='lines',
                    name='Actual',
                    line=dict(color='rgba(0,100,250, 0.7)', width=2)
                ))

                # Predictions line
                fig16.add_trace(go.Scatter(
                    x=df_test[col_data],
                    y=predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='rgba(239, 85, 59, 0.7)', width=2, dash='dot')
                ))

                # Highlight current prediction point
                fig16.add_trace(go.Scatter(
                    x=[selected_date],
                    y=[predictions[mid_point]],
                    mode='markers',
                    name='Current Forecast',
                    marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black'))
                ))

                fig16.update_layout(
                    title="Forecast Simulation: December 2012 (Realistic Model)",
                    xaxis_title="Date",
                    yaxis_title="Bike Rentals (cnt)",
                    xaxis_range=[pd.to_datetime(start_date), pd.to_datetime(end_date)],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                save_figure(fig16, "16_forecast_simulation")
            else:
                print(f"   ⚠️ Model requires features not in test data")
        else:
            print("   ⚠️ Test data empty or model has no feature names")
    else:
        print(f"   ⚠️ Model not found at {model_path}")

        # Create a simpler forecast figure using the v2 model (with all features)
        model_path_v2 = os.path.join(SCRIPT_DIR, "..", "models", "random_forest_model_v2.joblib")
        if os.path.exists(model_path_v2):
            model = joblib.load(model_path_v2)

            # Reload fresh data
            df_forecast = pd.read_csv(DATA_PATH)
            col_data = df_forecast.columns[0]
            df_forecast[col_data] = pd.to_datetime(df_forecast[col_data])

            mask = (df_forecast[col_data] >= "2012-12-01") & (df_forecast[col_data] < "2013-01-01")
            df_test = df_forecast.loc[mask].sort_values(by=col_data).reset_index(drop=True)

            if hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_
                available = [f for f in feature_cols if f in df_test.columns]

                if len(available) == len(feature_cols):
                    X_test = df_test[feature_cols]
                    predictions = model.predict(X_test)
                    mid_point = len(df_test) // 2
                    selected_date = df_test[col_data].iloc[mid_point]

                    fig16 = go.Figure()
                    fig16.add_trace(go.Scatter(x=df_test[col_data], y=df_test['cnt'],
                                              mode='lines', name='Actual',
                                              line=dict(color='rgba(0,100,250, 0.7)', width=2)))
                    fig16.add_trace(go.Scatter(x=df_test[col_data], y=predictions,
                                              mode='lines', name='Predicted',
                                              line=dict(color='rgba(239, 85, 59, 0.7)', width=2, dash='dot')))
                    fig16.add_trace(go.Scatter(x=[selected_date], y=[predictions[mid_point]],
                                              mode='markers', name='Current Forecast',
                                              marker=dict(color='red', size=15, symbol='star',
                                                         line=dict(width=2, color='black'))))
                    fig16.update_layout(
                        title="Forecast Simulation: December 2012",
                        xaxis_title="Date",
                        yaxis_title="Bike Rentals (cnt)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    save_figure(fig16, "16_forecast_simulation")

except Exception as e:
    print(f"   ⚠️ Error generating forecast figure: {e}")

# ==============================================================================
# EXPORT METRICS TO JSON
# ==============================================================================
print("\n📊 Exporting metrics to JSON...")

metrics = {
    "efficiency_kpi": {
        "average": round(df_daily['kpi_efficiency_rate'].mean(), 1),
        "last_day": round(df_daily['kpi_efficiency_rate'].iloc[-1], 1)
    },
    "user_composition": {
        "avg_pct_registered": round((df_daily['registered'] / df_daily['cnt']).mean() * 100, 1),
        "avg_pct_casual": round((df_daily['casual'] / df_daily['cnt']).mean() * 100, 1),
        "total_registered": int(df_daily['registered'].sum()),
        "total_casual": int(df_daily['casual'].sum())
    },
    "weather_impact": {
        "registered_drop_storm": round(((sunny_reg - storm_reg) / sunny_reg) * 100, 1),
        "casual_drop_storm": round(((sunny_cas - storm_cas) / sunny_cas) * 100, 1)
    },
    "peak_hours": {
        "peak_hour": int(df.groupby('hr')['cnt'].mean().idxmax()),
        "peak_value": round(df.groupby('hr')['cnt'].mean().max(), 0),
        "off_peak_hour": int(df.groupby('hr')['cnt'].mean().idxmin()),
        "off_peak_value": round(df.groupby('hr')['cnt'].mean().min(), 0)
    },
    "model_performance": {
        "leaky_model": {"R2": 0.9999, "RMSE": 1.21, "MAE": 0.52, "MAPE": 0.27},
        "realistic_model": {"R2": 0.9456, "RMSE": 41.01, "MAE": 25.05, "MAPE": 30.65}
    },
    "pca_variance": {
        "PC1": round(explained_var[0] * 100, 1),
        "PC2": round(explained_var[1] * 100, 1),
        "PC3": round(explained_var[2] * 100, 1),
        "cumulative_3": round(cumulative_var[2] * 100, 1)
    },
    "dataset_info": {
        "total_records": len(df),
        "total_days": len(df_daily),
        "date_range": f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    }
}

metrics_path = os.path.join(FIGURES_DIR, "metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✅ Saved: metrics.json")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("EXPORT COMPLETE")
print("=" * 60)

figures_list = [f for f in os.listdir(FIGURES_DIR) if f.endswith(('.png', '.html'))]
print(f"\n📁 Output directory: {FIGURES_DIR}")
print(f"📊 Figures exported: {len(figures_list)}")
print(f"📋 Metrics file: metrics.json")

print("\n📝 Figures list:")
for f in sorted(figures_list):
    print(f"   - {f}")

if not KALEIDO_AVAILABLE:
    print("\n⚠️  To export PNG images, install kaleido:")
    print("   pip install -U kaleido")
    print("   Then run this script again.")
