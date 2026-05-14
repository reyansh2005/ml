from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
# Create synthetic dataset
X,y=make_regression(n_samples=100,n_features=5,random_state=42)
# Split dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2
,random_state=42)
# Create Gradient Boosting model
model=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Gradient Boosting Regressor:")
print("mse:",mean_squared_error(y_test,y_pred))
print("r2 score:",r2_score(y_test,y_pred))
# Create AdaBoost model
model=AdaBoostRegressor(n_estimators=100,learning_rate=0.1,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("AdaBoost Regressor:")
print("mse:",mean_squared_error(y_test,y_pred))
print("r2 score:",r2_score(y_test,y_pred))
