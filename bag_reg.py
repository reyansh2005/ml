from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
# Create synthetic dataset
X,y=make_regression(n_samples=100,n_features=5,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("mse:",mean_squared_error(y_test,y_pred))
print("r2 score:",r2_score(y_test,y_pred))