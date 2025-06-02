import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your dataset
df = pd.read_csv("diabetes.csv")
df = df.dropna()

# Separate features and label
y = df["diabetes"]
X = df.drop("diabetes", axis=1)

# Convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
