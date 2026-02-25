from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from feature_engineering import get_features_tfidf, tfidf_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test = get_features_tfidf()


lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
svm_model = LinearSVC(dual=False) 
# class 0 → 'CG'; class 1 → 'OR'
# Logistic regression
print("Training Logistic Regression...")
lr_model.fit(X_train_tfidf, y_train)

y_val_pred_lr = lr_model.predict(X_val_tfidf)

print("Logistic Regression (Validation)")
print(classification_report(y_val, y_val_pred_lr))

# SVM
print("Training Linear SVM...")
svm_model.fit(X_train_tfidf, y_train)

y_val_pred_svm = svm_model.predict(X_val_tfidf)

print("Linear SVM (Validation)")
print(classification_report(y_val, y_val_pred_svm))


# Get feature names from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients
coeffs = lr_model.coef_[0]

# Create an importance df
importance_df = pd.DataFrame({
    "word": feature_names,
    "coefficient": lr_model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\nTop 10 words predicting REAL (OR) reviews:")
print(importance_df.head(10)) 

print("\nTop 10 words predicting FAKE (CG) reviews:")
print(importance_df.tail(10).iloc[::-1])  

import matplotlib.pyplot as plt

# Top 10 REAL
plt.figure(figsize=(10,5))
plt.barh(importance_df.head(10)['word'], importance_df.head(10)['coefficient'], color='green')
plt.title("Top Words Predicting REAL (OR) Reviews")
plt.xlabel("Coefficient")
plt.gca().invert_yaxis()
plt.show()

# Top 10 FAKE
plt.figure(figsize=(10,5))
plt.barh(importance_df.tail(10).iloc[::-1]['word'], importance_df.tail(10).iloc[::-1]['coefficient'], color='red')
plt.title("Top Words Predicting FAKE (CG) Reviews")
plt.xlabel("Coefficient")
plt.gca().invert_yaxis()
plt.show()