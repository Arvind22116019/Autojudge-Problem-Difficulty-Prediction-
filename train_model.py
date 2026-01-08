from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ===== IMPORTANT: convert to dense ONLY for regression =====
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# ===== Train Regressor =====
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train_dense, y_score_train)

# ===== Predict =====
y_score_pred = reg.predict(X_test_dense)

# ===== Evaluate =====
mae = mean_absolute_error(y_score_test, y_score_pred)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))

print("Regression MAE:", mae)
print("Regression RMSE:", rmse)
