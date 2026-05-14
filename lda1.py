from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
X,y=load_iris(return_X_y=True)
lda=LinearDiscriminantAnalysis(n_components=2)
X_lda=lda.fit_transform(X,y)
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='viridis')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA of Iris Dataset')
plt.show()