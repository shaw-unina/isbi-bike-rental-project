"""
Retrain Random Forest model WITHOUT lag features for realistic evaluation.
This avoids data leakage and provides true predictive performance metrics.
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

# First column is datetime (unnamed), use it as index
df.index = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]  # Remove first column

# Target
target = 'cnt'

# Identify and EXCLUDE lag features
lag_columns = [col for col in df.columns if 'lag' in col.lower()]
print(f"Excluding {len(lag_columns)} lag features: {lag_columns[:5]}... (and more)")

# Select features WITHOUT lags
features = [col for col in df.columns if col != target and col not in lag_columns]
print(f"\nUsing {len(features)} features:")
print(features)

X = df[features]
y = df[target]

# Split data (time-based would be better, but using random for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Random Forest (same hyperparameters as before)
print("\nTraining Random Forest model (without lag features)...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
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
    # MAPE (avoiding division by zero)
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
print("\n" + "="*50)
print("Top 10 Feature Importances:")
print("="*50)
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")

# Save model
output_path = os.path.join(script_dir, 'random_forest_no_lags.joblib')
joblib.dump(model, output_path)
print(f"\n✅ Model saved to: {output_path}")
print(f"   Features expected: {len(model.feature_names_in_)} columns")

# Comparison summary
print("\n" + "="*50)
print("COMPARISON: With Lags vs Without Lags")
print("="*50)
print("""
Model WITH lag features (previous):
  - R² ≈ 0.9999 (near perfect - data leakage)
  - MAPE ≈ 0.27%

Model WITHOUT lag features (this model):
  - R² = {:.4f} (realistic)
  - MAPE = {:.2f}%

The difference shows how much the lag features were 'cheating'
by providing almost-direct access to the target value.
""".format(test_metrics['R²'], test_metrics['MAPE']))
