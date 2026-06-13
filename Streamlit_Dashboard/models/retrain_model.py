"""
Retrain Random Forest model with current scikit-learn version.
Run this script to create a compatible model file.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Dataset_Bike.csv')
df = pd.read_csv(data_path)

# First column is datetime (unnamed), use it as index
df.index = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]  # Remove first column

# Target and features
target = 'cnt'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.4f}")
print(f"Test R²:  {test_score:.4f}")

# Save model
output_path = os.path.join(script_dir, 'random_forest_model_v2.joblib')
joblib.dump(model, output_path)
print(f"\nModel saved to: {output_path}")
print(f"Features expected by model: {len(model.feature_names_in_)} columns")
