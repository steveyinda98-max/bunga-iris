import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("data/iris.csv")  # pastikan folder 'data' ada

# Shuffle dataset
iris_df = iris_df.sample(frac=1, random_state=seed)

# Selecting features and target
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']   # ← perbaikan: Series, bukan DataFrame

# Split dataset (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=seed)

# Train model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(clf, "rf_model.sav")
