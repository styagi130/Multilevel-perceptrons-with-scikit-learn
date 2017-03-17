from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import time
np.random.seed(0)
X, y = make_moons(20000, noise=1) #generating data 200 is the number of data points noise defines randomness
#change the noise and number of points to see how the accuracy and traning time varies

#plotting datapoints
plt.scatter(X[:,0], X[:,1],s=20,c=y,cmap=plt.cm.Spectral)
plt.title("Data points")
plt.show()

t0=time.time()

#defining object for Neural network
clf = MLPClassifier(activation="tanh",hidden_layer_sizes=(6,1 ),alpha=0.0001, max_iter=2000000,solver="lbfgs")

#fitting the data
clf.fit(X, y)
t1=time.time()

c=clf.predict([[1,1.5]])
print "Loss after final iteration: ",clf.loss_,"\n","No of iterations: ",clf.n_iter_,"No of layers: ", clf.n_layers_,"accuracy: ",clf.score(X, y),"\n","Time: ",t1-t0,"\n","Print Pred: ",c

#plotting the decision boundary
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h=0.01

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y,s=20, cmap=plt.cm.Spectral)
plt.title("neural_network")
plt.show()
