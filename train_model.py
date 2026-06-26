import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# Load Sentiment Dataset
# =====================================

amazon = pd.read_csv(
    "amazon_cells_labelled.txt",
    sep="\t",
    names=["review", "sentiment"]
)

imdb = pd.read_csv(
    "imdb_labelled.txt",
    sep="\t",
    names=["review", "sentiment"]
)

yelp = pd.read_csv(
    "yelp_labelled.txt",
    sep="\t",
    names=["review", "sentiment"]
)

reviews = pd.concat(
    [amazon, imdb, yelp],
    ignore_index=True
)

print("Total Reviews :", len(reviews))

# =====================================
# Train Test Split
# =====================================

X = reviews["review"]
y = reviews["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# =====================================
# Logistic Regression
# =====================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", round(accuracy*100,2), "%")

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

# =====================================
# Save Model
# =====================================

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully")