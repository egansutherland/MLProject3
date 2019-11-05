# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-11-03 18:39:25
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-11-04 22:50:13
import neural_network as nnet
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
# import neural_network

def plot_decision_boundary(pred_func, X, y):
	# Set min and max values and give it some padding
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
	# Generate a gride of points with distance h between them
	xx, yy = np.meshgrid(np.arrange(x_min, x_max, h), np.arrange(y_min, y_max, h))
	# Predict the function value for the whole grid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

np.random.seed(0)
X, y = make_moons(200, noise = 0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
	plt.subplot(5,2,i+1)
	plt.title('HiddenLayerSize%d' % nn_hdim)
	# print(X)
	# print(X.shape)
	# print(X[0])
	# print(X[0].shape)
	# print(y)
	
	model = nnet.build_model(X, y, nn_hdim)
	plot_decision_boudary(lambda x: predict(model,x),X,y)
plt.show()

