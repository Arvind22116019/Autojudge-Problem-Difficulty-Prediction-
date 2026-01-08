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

#Score Prediction
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train_dense, y_score_train)

y_score_pred = reg.predict(X_test_dense)

mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))

print("Regression MAE:", mae)
print("Regression RMSE:", rmse)

