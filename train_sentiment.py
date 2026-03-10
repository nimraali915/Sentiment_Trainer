import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# loading
data = pd.read_csv("imbd_clean.csv", encoding="utf-8")

X = data["Review"]
y = data["Sentiment"]   # here label 0 for negative and  1 for Positive

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build pipeline
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))

# 4. Train
print("Training model...")
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
print("\nEvaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save sklearn model
joblib.dump(pipeline, "sentiment_model.pkl")

# 7. Retrain on FULL dataset
pipeline.fit(X, y)
joblib.dump(pipeline, "sentiment_model_final.pkl")

# 8. Convert to ONNX
initial_type = [("input", StringTensorType([None]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

with open("sentiment_model_final.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(" Model exported: sentiment_model_final.onnx")
