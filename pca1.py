from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X=load_iris().data
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
plt.scatter(X_pca[:,0],X_pca[:,1],c=load_iris().target,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
