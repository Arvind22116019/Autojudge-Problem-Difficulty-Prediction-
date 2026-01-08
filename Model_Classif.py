#Classification
clf = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)

print("Classification Accuracy:",
      accuracy_score(y_class_test, y_class_pred))

print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_class_test, y_class_pred))



