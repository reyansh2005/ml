from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
# Create synthetic dataset
X,y=make_classification(n_samples=200,n_features=10,n_classes=2,random_state=42)
model=SVC(kernel="rbf",gamma='scale')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))
print("accuracy:",accuracy_score(y_test,y_pred))