from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
X,y=make_blobs(n_samples=300,n_features=2,centers=4,random_state=42)
model=GaussianMixture(n_components=4,random_state=42)
model.fit(X)
y_pred=model.predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred,cmap='viridis')
plt.scatter(model.means_[:,0],model.means_[:,1],c='red',marker='X',s=200)
plt.show()