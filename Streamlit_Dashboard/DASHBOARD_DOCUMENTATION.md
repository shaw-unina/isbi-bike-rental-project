# Streamlit Dashboard Documentation

A comprehensive multi-page Streamlit dashboard for **Bike Rental Time Series Analysis and Forecasting**. This project demonstrates data analysis, visualization, machine learning model training, and interactive forecasting capabilities.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Installation and Running](#installation-and-running)
4. [Core Concepts](#core-concepts)
5. [File Descriptions](#file-descriptions)
   - [Main Application](#main-application)
   - [Dashboard Pages](#dashboard-pages)
   - [Models Directory](#models-directory)
   - [Documentation Directory](#documentation-directory)
   - [Mock Application](#mock-application)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Key Metrics and KPIs](#key-metrics-and-kpis)
8. [Glossary of Terms](#glossary-of-terms)

---

## Project Overview

This Streamlit dashboard provides an interactive platform for analyzing bike rental data from a bike-sharing system. The application includes:

- **Data Exploration**: Upload CSV data and explore statistics, user composition, and temporal patterns
- **KPI Monitoring**: Track business efficiency metrics (registered vs casual users)
- **Weather Impact Analysis**: Analyze how weather conditions affect bike rentals
- **Time Series Analysis**: Decomposition, autocorrelation, and stationarity tests
- **ML Forecasting**: Real-time prediction simulation using trained Random Forest models
- **Model Comparison**: Compare multiple ML models using standard regression metrics

---

## Directory Structure

```
Streamlit_Dashboard/
├── App_progetto.py              # Main entry point (Home page)
├── Dataset_Bike.csv             # Bike rental dataset (17,379 hourly records)
├── requirements.txt             # Python dependencies
├── README.md                    # Basic readme
│
├── pages/                       # Streamlit multi-page app pages
│   ├── 1AnalisiDati.py          # Data Analysis page
│   ├── 2Forecast.py             # Forecast Simulation page
│   ├── 3ModelComparison.py      # Model Comparison page
│   └── 4AdvancedAnalysis.py     # Advanced Analysis page
│
├── models/                      # Machine learning models and training scripts
│   ├── random_forest_model.joblib        # Original model (with data leakage)
│   ├── random_forest_model_v2.joblib     # V2 model (with data leakage)
│   ├── random_forest_no_lags.joblib      # Model without lag features
│   ├── random_forest_realistic.joblib    # Realistic model (no leakage)
│   ├── retrain_model.py                  # Original training script
│   ├── retrain_model_no_lags.py          # Training without lags
│   └── retrain_model_realistic.py        # Realistic training script
│
├── docs/                        # Documentation and exports
│   ├── export_figures.py        # Script to export figures for LaTeX
│   ├── figures/                 # Generated PNG figures (16 files)
│   ├── bike_rental_report.tex   # LaTeX report
│   └── bike_rental_report.pdf   # Compiled PDF report
│
└── mock_application/            # Tutorial application for Streamlit concepts
    ├── app.py                   # Mock app entry point
    └── pages/
        ├── 1_session_state.py   # Session state demo
        ├── 2_connection.py      # Database connection demo
        ├── 3_caching.py         # Caching demo
        └── 4_pages.py           # Multi-page app demo
```

---

## Installation and Running

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Run main dashboard
streamlit run App_progetto.py

# Run mock/tutorial application (from mock_application directory)
streamlit run mock_application/app.py
```

### Dependencies (requirements.txt)

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `plotly` | Interactive visualizations |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | Machine learning algorithms |
| `statsmodels` | Statistical models (decomposition, ACF/PACF) |
| `xgboost` | Gradient boosting models |
| `prophet` | Time series forecasting |
| `tensorflow` | Deep learning |
| `joblib` | Model serialization |

---

## Core Concepts

### Session State

Streamlit's `st.session_state` is a dictionary-like object that persists data across page reruns and navigation. This dashboard uses it to share the loaded DataFrame between pages:

```python
# Store data in session state
st.session_state['df_condiviso'] = df

# Access data from any page
if 'df_condiviso' in st.session_state:
    df = st.session_state['df_condiviso']
```

**Key variable**: `df_condiviso` (Italian for "shared dataframe") - contains the uploaded bike rental data.

### Multi-Page Architecture

Streamlit automatically creates a multi-page app when Python files are placed in a `pages/` folder. Files are ordered alphabetically, so numeric prefixes control the order:

- `1AnalisiDati.py` → First page in sidebar
- `2Forecast.py` → Second page
- `3ModelComparison.py` → Third page
- `4AdvancedAnalysis.py` → Fourth page

### Plotly Interactive Charts

All visualizations use Plotly for interactive features:
- Hover tooltips
- Zoom and pan
- Download as PNG
- Dynamic filtering

### Joblib Model Serialization

Trained models are saved using `joblib` and include the `feature_names_in_` attribute for input validation:

```python
# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')

# Get expected features
feature_cols = model.feature_names_in_
```

---

## File Descriptions

### Main Application

#### `App_progetto.py`
**Purpose**: Entry point and home page of the dashboard.

**Location Called From**: Run directly with `streamlit run App_progetto.py`

**Key Components**:
- Sets page configuration (`st.set_page_config`)
- Displays welcome message and navigation instructions
- Simple landing page directing users to sidebar navigation

```python
st.set_page_config(page_title="Data Portal", layout="wide")
st.title("🏠 Home Page")
```

---

### Dashboard Pages

#### `pages/1AnalisiDati.py`

**Purpose**: Main data analysis page with KPIs, weather impact, and temporal pattern analysis.

**Called From**: Automatically loaded by Streamlit from `pages/` directory; appears as "1AnalisiDati" in sidebar.

**Key Features**:

| Section | Description |
|---------|-------------|
| Data Loading | CSV upload via sidebar, stores in `st.session_state['df_condiviso']` |
| Overview | Row count, columns, basic statistics (mean, min, max, std) |
| KPI Dashboard | Efficiency rate = registered/total users (%) |
| User Composition | Stacked area chart of registered vs casual users |
| Weather Impact | Year-over-year comparison with weather conditions |
| Peak Hours | Heatmap of hour × day-of-week, working vs non-working day patterns |
| Correlation Analysis | Scatter plots and box plots by hour/season/temperature |

**Key Concepts**:

1. **Efficiency Rate KPI**:
   ```python
   kpi_efficiency_rate = (registered / cnt) * 100
   ```
   Measures the percentage of registered (subscription) users vs total rentals.

2. **One-Hot Encoded Weather**: Dataset uses columns `weathersit_1.0` through `weathersit_4.0`:
   - `weathersit_1.0`: Sunny/Clear
   - `weathersit_2.0`: Cloudy
   - `weathersit_3.0`: Light Rain
   - `weathersit_4.0`: Storm

3. **Daily Aggregation**: Hourly data is grouped by date for trend analysis:
   ```python
   df_daily = df.groupby(df.index.date)[['registered', 'cnt']].sum()
   ```

---

#### `pages/2Forecast.py`

**Purpose**: Real-time forecasting simulation page with interactive time slider.

**Called From**: Sidebar navigation; requires data loaded in `1AnalisiDati.py` first.

**Prerequisites**:
- Data must be loaded via session state
- Model file (`.joblib`) must be uploaded via sidebar

**Key Features**:
- Time slider for December 2012 test period
- Dynamic prediction display
- Historical vs predicted visualization
- Uses model's `feature_names_in_` for input validation

**How It Works**:
```python
# 1. Load model from sidebar upload
model = joblib.load(model_file)

# 2. Filter to December 2012 test set
df_test = df[(df[col_data] >= "2012-12-01") & (df[col_data] < "2013-01-01")]

# 3. Generate prediction for selected timestamp
X_input = row[model.feature_names_in_]
prediction = model.predict(X_input)[0]
```

---

#### `pages/3ModelComparison.py`

**Purpose**: Compare multiple ML models using standard regression metrics.

**Called From**: Sidebar navigation; requires data loaded in `1AnalisiDati.py` first.

**Key Features**:

| Feature | Description |
|---------|-------------|
| Multi-model upload | Upload multiple `.joblib` files simultaneously |
| Metrics calculation | RMSE, MAE, R², MAPE for each model |
| Visual comparison | Grouped bar charts, prediction overlay, error distribution |
| Best model identification | Highlights best performer with green background |

**Metrics Calculated**:

```python
def calculate_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE (%)': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
```

**Metric Definitions**:
- **RMSE (Root Mean Square Error)**: Square root of average squared differences. Penalizes large errors. Lower is better.
- **MAE (Mean Absolute Error)**: Average of absolute differences. More robust to outliers. Lower is better.
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model. Range 0-1, higher is better.
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Scale-independent. Lower is better.

---

#### `pages/4AdvancedAnalysis.py`

**Purpose**: Advanced time series analysis including decomposition, autocorrelation, and PCA.

**Called From**: Sidebar navigation; requires data loaded in `1AnalisiDati.py` first.

**Key Features**:

| Analysis | Description |
|----------|-------------|
| Time Series Decomposition | Separates data into Trend, Seasonal, and Residual components |
| Autocorrelation (ACF) | Measures correlation of series with lagged versions of itself |
| Partial Autocorrelation (PACF) | Correlation after removing intermediate lag effects |
| PCA | Reduces dimensionality of weather features |
| Stationarity Test (ADF) | Tests if series has unit root (non-stationary) |

**Key Concepts**:

1. **Time Series Decomposition**:
   ```python
   result = seasonal_decompose(series, model='additive', period=7)
   # Components: result.observed, result.trend, result.seasonal, result.resid
   ```
   - **Additive**: Y = Trend + Seasonal + Residual (when seasonal amplitude is constant)
   - **Multiplicative**: Y = Trend × Seasonal × Residual (when seasonal amplitude varies)

2. **Autocorrelation Function (ACF)**:
   ```python
   acf_values = acf(series, nlags=48)
   ```
   - Measures correlation between observations at different time lags
   - Values outside confidence bounds (±1.96/√n) are statistically significant
   - Key lags: 1 (immediate), 24 (daily cycle), 168 (weekly cycle)

3. **Partial Autocorrelation Function (PACF)**:
   ```python
   pacf_values = pacf(series, nlags=48, method='ols')
   ```
   - Measures direct correlation at lag k, removing effects of intermediate lags
   - Helps determine AR order in ARIMA models

4. **Principal Component Analysis (PCA)**:
   ```python
   pca = PCA()
   pca.fit(StandardScaler().fit_transform(X))
   explained_variance = pca.explained_variance_ratio_
   ```
   - Reduces dimensionality of weather features (temp, atemp, hum, windspeed)
   - PC1 captures ~50% variance (temperature-related)
   - PC2 captures ~27% variance (humidity/wind patterns)

5. **Augmented Dickey-Fuller Test**:
   ```python
   adf_result = adfuller(series)
   # adf_result[1] is p-value; < 0.05 means stationary
   ```
   - Tests null hypothesis that a unit root is present (non-stationary)
   - P-value < 0.05: Reject null → Series is stationary
   - Stationary data has constant mean and variance over time

---

### Models Directory

#### `models/random_forest_realistic.joblib`

**Purpose**: Production-ready Random Forest model trained WITHOUT data leakage.

**Training Script**: `retrain_model_realistic.py`

**Features Used** (20 features):
- Time: `hr`, `season`, `yr`, `holiday`, `workingday`, `Orario_punta`
- Weather: `temp`, `atemp`, `hum`, `windspeed`
- Month one-hot: `mnth_1` through `mnth_12`
- Day of week one-hot: `weekday_0` through `weekday_6`
- Weather one-hot: `weathersit_1.0` through `weathersit_4.0`

**Performance**:
| Metric | Value |
|--------|-------|
| R² | 0.9456 |
| RMSE | 41.01 |
| MAE | 25.05 |
| MAPE | 30.65% |

**Why "Realistic"**: Excludes features that cause data leakage:
- `registered` + `casual` = `cnt` (trivial prediction)
- Lag features (future information in test set)

---

#### `models/random_forest_model_v2.joblib`

**Purpose**: Model with ALL features including lag features.

**Warning**: This model exhibits data leakage and shows unrealistic performance (R² = 0.9999).

**Features Include**: All columns except `cnt`, including:
- `cnt_lag_1` through `cnt_lag_24` (target variable lag features)
- `registered`, `casual` (sum to target)

---

#### `models/retrain_model_realistic.py`

**Purpose**: Training script for the realistic Random Forest model.

**Called From**: Run manually from command line:
```bash
cd models
python retrain_model_realistic.py
```

**Key Logic**:
```python
# Features to EXCLUDE (leakage or trivial)
exclude_features = [
    'cnt',           # Target
    'registered',    # Part of target (registered + casual = cnt)
    'casual',        # Part of target
] + lag_columns      # All cnt_lag_* columns

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

---

### Documentation Directory

#### `docs/export_figures.py`

**Purpose**: Export all dashboard visualizations as PNG images for LaTeX reports.

**Called From**: Run manually:
```bash
cd docs
python export_figures.py
```

**Requirements**:
```bash
pip install kaleido  # For Plotly image export
```

**Outputs** (16 figures):
1. `01_efficiency_trend.png` - Daily efficiency KPI trend
2. `02_subscribers_vs_casual_area.png` - Stacked area chart of user types
3. `03_monthly_user_distribution.png` - Monthly bar chart
4. `04_weather_impact_yearly.png` - 2011 vs 2012 weather comparison
5. `05_climate_impact_user_type.png` - Weather impact on registered vs casual
6. `06_peak_hours_heatmap.png` - Hour × Day heatmap
7. `07_working_vs_nonworking.png` - Hourly patterns comparison
8. `08_boxplot_hour.png` - Distribution by hour
9. `09_temp_vs_rentals.png` - Temperature scatter plot
10. `10_decomposition.png` - Time series decomposition
11. `11_acf_plot.png` - Autocorrelation function
12. `12_pacf_plot.png` - Partial autocorrelation function
13. `13_pca_scree_plot.png` - PCA variance explained
14. `14_feature_importance.png` - Random Forest feature importances
15. `15_model_comparison.png` - Leaky vs Realistic model comparison
16. `16_forecast_simulation.png` - December 2012 predictions

**Also Exports**: `metrics.json` with key statistics for report integration.

---

### Mock Application

#### `mock_application/app.py`

**Purpose**: Tutorial application demonstrating core Streamlit concepts.

**Called From**: Run with `streamlit run mock_application/app.py`

**Topics Covered**:
1. Session State management
2. Database connections (simulated)
3. Caching strategies
4. Multi-page app architecture

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      App_progetto.py (Home)                      │
│                     - Welcome message                            │
│                     - Navigation instructions                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1AnalisiDati.py (Data Analysis)               │
│                                                                  │
│   [Sidebar: CSV Upload] ─────────────────────────────┐          │
│                                                       │          │
│   1. Load CSV file                                    │          │
│   2. Convert first column to datetime index           │          │
│   3. Store in st.session_state['df_condiviso']  ◄────┘          │
│   4. Display KPIs, charts, weather analysis                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    st.session_state['df_condiviso']
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 2Forecast.py    │   │ 3ModelComparison│   │ 4AdvancedAnalysis│
│                 │   │                 │   │                  │
│ - Load model    │   │ - Load models   │   │ - Decomposition  │
│ - Filter Dec'12 │   │ - Calculate     │   │ - ACF/PACF       │
│ - Time slider   │   │   metrics       │   │ - PCA            │
│ - Predictions   │   │ - Compare       │   │ - ADF Test       │
└─────────────────┘   └─────────────────┘   └──────────────────┘
```

---

## Key Metrics and KPIs

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Efficiency Rate | `registered / cnt × 100` | Higher = more committed users |
| Weather Sensitivity | `(sunny - storm) / sunny × 100` | Higher = more weather-dependent |
| Peak Hour | `argmax(hourly_avg)` | Hour with highest average rentals |
| Commute Pattern | Working day 8AM + 5PM peaks | Indicates commuter usage |
| R² (Model) | `1 - (SS_res / SS_tot)` | Variance explained (0-1) |

---

## Glossary of Terms

| Term | Definition |
|------|------------|
| **cnt** | Total count of bike rentals (target variable) |
| **registered** | Number of registered (subscription) users |
| **casual** | Number of casual (non-subscription) users |
| **weathersit** | Weather situation (1=Clear, 2=Cloudy, 3=Rain, 4=Storm) |
| **workingday** | Binary flag (1=workday, 0=weekend/holiday) |
| **hr** | Hour of the day (0-23) |
| **temp/atemp** | Normalized temperature / "feels like" temperature |
| **hum** | Normalized humidity |
| **windspeed** | Normalized wind speed |
| **season** | Season (1=Spring, 2=Summer, 3=Fall, 4=Winter) |
| **Orario_punta** | Italian for "peak hour" - binary flag for rush hours |
| **df_condiviso** | Italian for "shared dataframe" - the session state variable |
| **Data Leakage** | When training data contains information about the target that wouldn't be available at prediction time |
| **Lag Features** | Past values of the target variable used as features |
| **Stationarity** | Statistical property where mean and variance are constant over time |
| **ACF** | Autocorrelation Function - correlation at different lags |
| **PACF** | Partial ACF - direct correlation removing intermediate effects |
| **PCA** | Principal Component Analysis - dimensionality reduction technique |
| **ADF Test** | Augmented Dickey-Fuller test for stationarity |

---

## Dataset Schema

The `Dataset_Bike.csv` file contains 17,379 hourly records with 60 columns:

### Core Columns
| Column | Type | Description |
|--------|------|-------------|
| (first column) | datetime | Timestamp (YYYY-MM-DD HH:MM:SS) |
| season | float | Season (1-4) |
| yr | int | Year (2011, 2012) |
| hr | int | Hour (0-23) |
| holiday | float | Holiday flag (0/1) |
| workingday | float | Working day flag (0/1) |

### Weather Columns
| Column | Type | Description |
|--------|------|-------------|
| temp | float | Normalized temperature (0-1) |
| atemp | float | Normalized "feels like" temp (0-1) |
| hum | float | Normalized humidity (0-1) |
| windspeed | float | Normalized wind speed (0-1) |
| weathersit_1.0 | int | One-hot: Clear/Sunny |
| weathersit_2.0 | int | One-hot: Cloudy |
| weathersit_3.0 | int | One-hot: Light Rain |
| weathersit_4.0 | int | One-hot: Storm |

### Target Columns
| Column | Type | Description |
|--------|------|-------------|
| casual | float | Casual user count |
| registered | float | Registered user count |
| cnt | float | Total rentals (casual + registered) |

### Engineered Features
| Column | Type | Description |
|--------|------|-------------|
| Orario_punta | int | Peak hour indicator |
| mnth_1 to mnth_12 | int | One-hot month encoding |
| weekday_0 to weekday_6 | int | One-hot day-of-week encoding |
| cnt_lag_1 to cnt_lag_24 | float | Lag features (past values of cnt) |

---

*Documentation generated January 2025*
