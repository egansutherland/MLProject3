import numpy as np
import matplotlib.pyplot as plt
import math
import copy
learning_rate = 0.01


# Helper function to predict an output (0 or 1)
# Model is the current version of the model {'W1':W1, 'b1’:b1, 'W2':W2, 'b2',b2}. It's a dictionary
# x is one sample (without the label)
def predict(model, x):
	a = np.add(np.matmul(x, model["W1"]), model["b1"])
	h = np.tanh(a)
	z = np.add(np.matmul(h, model["W2"]), model["b2"])
	z = np.exp(z)
	z = np.true_divide(z, np.sum(z))
	return np.argmax(z)


# Helper function to evaluate the total loss on the dataset
# model is the current version of the model {'W1':W1, 'b1’:b1, 'W2':W2, 'b2',b2}. It's a dictionary
# X is all the training data
# y is the training labels
def calculate_loss(model, X, y):
	n = len(X[0])
	loss = 0
	for sample, label in X, y:
		# Forward propagation. Same as predict, but keeps y_hat as a 2d array
		a = np.add(np.matmul(sample, model['W1']), model['b1'])
		h = np.tanh(a)
		z = np.add(np.matmul(h, model['W2']), model['b2'])
		z = np.exp(z)
		y_hat = np.true_divide(z, np.sum(z))
		y_label =[]
		if label == 0:
			y_label = np.array([1, 0])
		else:
			y_label = np.array([0, 1])
		loss = loss + (y_label[0]*np.log(y_hat[0])) + (y_label[1]*np.log(y_hat[1]))
	return_value = loss*(-1)/n
	return return_value


# This function learns parameters for the neural network and returns the model.
# - X is the training data
# - y is the training labels
#  nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
#- print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
	global learning_rate
	# Initialize weights randomly
	W1 = np.random.rand(len(X[0]), nn_hdim)
	W2 = np.random.rand(nn_hdim, 2)
	b1 = np.random.rand(nn_hdim)
	b2 = np.random.rand(2)
	print('w1', W1)
	print('w1 shape', W1.shape)
	print('w2', W2)
	print('w2 shape', W2.shape)
	# Gradient variables for back-prop
	grad_W1 = np.zeros((len(X[0]), nn_hdim))
	grad_W2 = np.zeros((nn_hdim, 2))
	grad_b1 = np.zeros(nn_hdim)
	grad_b2 = np.zeros(2)
	# Setup model variable
	model = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

	for i in range(0,num_passes):
		for j in range(0, len(X)):


			# Forward propagation. Same as predict, but keeps y_hat as a 2d array
			a = np.add(np.matmul(X[j], W1), b1)
			h = np.tanh(a)
			z = np.add(np.matmul(h, W2), b2)
			z = np.exp(z)
			y_hat = np.true_divide(z, np.sum(z))

			# print the loss every 1000 epochs
			if print_loss and i == 1000:
				loss = calculate_loss(model, X, y)
				print(loss)

			# Some set up for the back propagation
			
			# Numpy representation vectors is troublesome for multiplication:
			# We know:    b2 = [ b1 b2]
			# and the transpose is:    [ b1
			#                            b2 ]
			# Where the original is a row vector and the other is a column vector
			# This distinction in shape is important for matrix multiplication
			# but since it's one dimensional the numpy view seemingly makes no
			# distinction. It just returns an iterator. To fix this, you have
			# to force it into a 2D array simply by enclosing the single array
			# into another array. Like so: [ [b1, b2] ]
			h = np.array([h])
			y_hat = np.array([y_hat])

			sample = np.array([X[j]])


			# Since labels is a list of classes (0 or 1), have to convert to a probability distribution
			yj = []
			if y[j] == 0:
				yj = np.array([1,0])
			else:
				yj = np.array([0,1])

			#Back propagation	
			dl_dy = np.subtract(y_hat, yj)
			dl_da = np.multiply(1 - np.square(h), np.matmul(dl_dy, np.transpose(W2)))
			dl_da.reshape((1, nn_hdim))
			dl_w2 = np.matmul(np.transpose(h), dl_dy)
			dl_w1 = np.matmul(np.transpose(sample), dl_da) 
			dl_b1 = copy.deepcopy(dl_da)
			dl_b2 = copy.deepcopy(dl_dy)

			grad_W1 = grad_W1 + dl_w1
			grad_W2 = grad_W2 + dl_w2
			grad_b1 = grad_b1 + dl_b1
			grad_b2 = grad_b2 + dl_b2

		## finished one epoch

		# Fix shape change
		grad_b1 = np.reshape(grad_b1, b1.shape)
		grad_b2 = np.reshape(grad_b2, b2.shape)

		# Get average gradients
		grad_W1 = grad_W1/len(X)
		grad_W2 = grad_W2/len(X)
		grad_b1 = grad_b1/len(X)
		grad_b2 = grad_b2/len(X)

		# update weights and biases
		W1 = np.subtract(W1, grad_W1 * learning_rate)
		W2 = np.subtract(W2, grad_W2 * learning_rate)
		b1 = np.subtract(b1, grad_b1 * learning_rate)
		b2 = np.subtract(b2, grad_b2 * learning_rate)
		model["W1"] = W1
		model["W2"] = W2
		model["b1"] = b1
		model["b2"] = b2

		# Clear average gradient counters and keep going
		grad_W1 = np.zeros((len(X[0]),nn_hdim))
		grad_W2 = np.zeros((nn_hdim, 2))
		grad_b1 = np.zeros(nn_hdim)
		grad_b2 = np.zeros(2)

	return model
			

	

