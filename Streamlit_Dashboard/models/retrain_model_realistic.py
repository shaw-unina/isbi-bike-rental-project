"""
Retrain Random Forest model with REALISTIC features only.
Excludes:
  - Lag features (data leakage from past target values)
  - registered/casual (they SUM to cnt - trivial prediction)

This gives TRUE predictive performance based only on:
  - Time features (hour, day, month, season)
  - Weather features (temp, humidity, wind, weather condition)
  - Calendar features (working day, holiday)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Dataset_Bike.csv')
df = pd.read_csv(data_path)

# First column is datetime
df.index = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]

# Target
target = 'cnt'

# Features to EXCLUDE (leakage or trivial)
exclude_features = [
    'cnt',           # Target
    'registered',    # Part of target (registered + casual = cnt)
    'casual',        # Part of target
]

# Also exclude lag features
lag_columns = [col for col in df.columns if 'lag' in col.lower()]
exclude_features.extend(lag_columns)

print("="*60)
print("REALISTIC MODEL TRAINING")
print("="*60)
print(f"\nExcluding {len(exclude_features)} features (leakage/trivial):")
print(f"  - cnt (target)")
print(f"  - registered, casual (sum to target)")
print(f"  - {len(lag_columns)} lag features")

# Select REALISTIC features only
features = [col for col in df.columns if col not in exclude_features]
print(f"\nUsing {len(features)} REALISTIC features:")
for f in features:
    print(f"  - {f}")

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Random Forest
print("\n" + "="*60)
print("Training Random Forest (realistic features only)...")
print("="*60)

model = RandomForestRegressor(
    n_estimators=150,      # Slightly more trees
    max_depth=20,          # Deeper trees for complex patterns
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")

    return {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}

train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# Feature importance
print("\n" + "="*60)
print("Top 15 Feature Importances:")
print("="*60)
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    bar = "█" * int(row['Importance'] * 50)
    print(f"  {row['Feature']:20s}: {row['Importance']:.4f} {bar}")

# Save model
output_path = os.path.join(script_dir, 'random_forest_realistic.joblib')
joblib.dump(model, output_path)
print(f"\n✅ Model saved to: {output_path}")
print(f"   Features expected: {len(model.feature_names_in_)} columns")

# Summary
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON                         │
├──────────────────────┬──────────┬──────────┬───────────────┤
│ Model                │ R²       │ RMSE     │ MAPE          │
├──────────────────────┼──────────┼──────────┼───────────────┤
│ With ALL features    │ 0.9999   │ 1.21     │ 0.27%         │
│ (lag + reg + cas)    │ (leaky)  │          │ (unrealistic) │
├──────────────────────┼──────────┼──────────┼───────────────┤
│ REALISTIC features   │ {r2:.4f}   │ {rmse:.2f}    │ {mape:.2f}%          │
│ (weather + time)     │ (true)   │          │ (realistic)   │
└──────────────────────┴──────────┴──────────┴───────────────┘

The REALISTIC model predicts bike demand using only information
that would be available BEFORE the rental period:
  - Weather forecast (temp, humidity, wind, conditions)
  - Calendar (day of week, month, season, holiday)
  - Time of day (hour, peak hour indicator)

This is what you'd use in a real deployment scenario.
""".format(r2=test_metrics['R²'], rmse=test_metrics['RMSE'], mape=test_metrics['MAPE']))
