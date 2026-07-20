import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import pickle
import os

# Load data
df = pd.read_csv('data/DravidianCodeMix/tamil_offensive_full.csv', header=None)
df.columns = ['text', 'label']

print("Dataset shape:", df.shape)
print("Label distribution:\n", df['label'].value_counts())

# Split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train with MLflow tracking
with mlflow.start_run():
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    report = classification_report(y_test, preds, output_dict=True)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("accuracy", report['accuracy'])
    mlflow.sklearn.log_model(model, "model")

    print(classification_report(y_test, preds))

# Save model
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
print("Model saved to models/")
