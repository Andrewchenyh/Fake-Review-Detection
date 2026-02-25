import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('../data/data_for_regression.csv')
df.head()

X_text = df["text"]
y = df["label"]

print(df.shape)
print(y.value_counts())

# Train test val split
from sklearn.model_selection import train_test_split

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)



tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2),
    sublinear_tf=True
)


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
X_val_tfidf = tfidf_vectorizer.transform(X_val_text)


def get_features_tfidf():
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test
