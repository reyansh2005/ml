from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
X,y=load_iris(return_X_y=True)
U,S,Vt=np.linalg.svd(X)
X_svd=U[:, :2]
plt.scatter(X_svd[:,0],X_svd[:,1],c=y,cmap='viridis')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')   
plt.title('SVD of Iris Dataset')
plt.show()