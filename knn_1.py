from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Create synthetic dataset
X,y=make_classification(n_samples=100,n_features=5,n_classes=2,random_state=42)
# Split dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Create KNN model
model=KNeighborsClassifier(n_neighbors=3)
# Fit the model
model.fit(X_train,y_train)
# Predict on test set
y_pred=model.predict(X_test)
# Print classification report
print(classification_report(y_test,y_pred))