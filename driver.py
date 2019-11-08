# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-11-03 18:39:25
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-11-07 19:07:09
from neural_network import *
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Fancy comment for commit message
# import neural_network

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


np.random.seed(5)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim, print_loss=True)
    print('model', model)
    plot_decision_boundary(lambda X: np.array([predict(model, x) for x in X]), X, y)
    # plot_decision_boundary(lambda x:  predict(model,x),X,y)
plt.show()
