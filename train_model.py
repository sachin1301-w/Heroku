import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv("dataset.csv").fillna("")

X = data[["url", "title", "html"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("url", TfidfVectorizer(analyzer="char", ngram_range=(3, 5)), "url"),
        ("title", TfidfVectorizer(analyzer="word", ngram_range=(1, 2)), "title"),
        ("html", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000), "html"),
    ]
)

model = Pipeline([
    ("features", preprocessor),
    ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
])

# -----------------------------
# Train Model
# -----------------------------
print("🔹 Training model...")
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "spot_the_fake_model.pkl")
print("📁 Model saved as spot_the_fake_model.pkl")
