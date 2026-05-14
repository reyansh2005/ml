from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,r2_score
# Create synthetic dataset
X,y=make_regression(n_samples=100,n_features=5,random_state=42)
# Split dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Create KNN model
model=KNeighborsRegressor(n_neighbors=3,weights='uniform')
# Fit the model
model.fit(X_train,y_train)
# Predict on test set
y_pred=model.predict(X_test)
# Print classification report
#print(classification_report(y_test,y_pred))
print("mse:",mean_squared_error(y_test,y_pred))
print("r2 score:",r2_score(y_test,y_pred))