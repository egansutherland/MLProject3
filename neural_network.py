import numpy as np
import matplotlib.pyplot as plt
import math
learning_rate = 0.01

# Helper function to predict an output (0 or 1)
# Model is the current version of the model {'W1':W1, 'b1’:b1, 'W2':W2, 'b2',b2}. It's a dictionary
# x is one sample (without the label)
def predict(model, x):
	a = np.add(np.matmul(x,model["W1"]), model["b1"])
	h = np.tanh(a)
	z = np.add(np.matmul(h,model["W2"]), model["b2"])
	z = np.exp(z)
	z = np.true_divide(z,np.sum(z))
	return z	

# Helper function to evaluate the total loss on the dataset
# model is the current version of the model {'W1':W1, 'b1’:b1, 'W2':W2, 'b2',b2}. It's a dictionary
# X is all the training data
# y is the training labels
def calculate_loss(model, X, y):
	n = len(X[0])
	y_hat = np.array()
	for sample in X:
		y_hat.append(predict(model, sample))
	y_hat = np.log(y_hat)
	intermediate = np.matmul(y,y_hat)
	return_value = np.sum(intermediate)*(-1/n)
	return return_value


# This function learns parameters for the neural network and returns the model.
# - X is the training data
# - y is the training labels
#  nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
#- print_loss: If True, print the loss every 100 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
	pass

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