import numpy as np
import matplotlib.pyplot as plt
import math
import copy
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
	return_value = np.dot(y,y_hat)*(-1)*math.pow(n,-1)
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
	b1 = np.random.rand((1,nn_hdim))
	b2 = np.random.rand((1,len(X[0])))
	# Gradient variables for back-prop
	grad_W1 = np.zeros((2,nn_hdim))
	grad_W2 = np.zeros((nn_hdim, 2))
	grad_b1 = np.zeros((1,nn_hdim))
	grad_b2 = np.zeros((1,len(X[0])))

	model = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

	for i in range(0,num_passes):
		for j in range(0, len(X)):
			y_hat = predict(model, X[j])
			# print the loss every 1000 epochs
			if print_loss and i == 1000:
				loss = calculate_loss(model, X, y)
				print(loss)
			dl_dy = y_hat - y[j]
			a = np.add(np.matmul(X[j],W1), b1)
			dl_da = np.matmul(np.matmul((1 - (math.pow(np.tanh(a,2)))),dl_dy), np.transpose(W2))
			h = np.tanh(a)
			dl_w2 = np.matmul(np.transpose(h),dl_dy)
			dl_w1 = np.matmul(np.transepose(X[j]), dl_da) 
			dl_b1 = copy.deepcopy(dl_da)
			dl_b2 = copy.deepcopy(dl_dy)

			grad_W1 = grad_W1 + dl_w1
			grad_W2 = grad_W2 + dl_w2
			grad_b1 = grad_b1 + dl_b1
			grad_b2 = grad_b2 + dl_b2

		# finished one epoch, update weights and biases
		W1 = np.subtract(W1, np.multiply(grad_W1, learning_rate))
		W2 = np.subtract(W2, np.multiply(grad_W2, learning_rate))
		b1 = np.subtract(b1, np.multiply(grad_b1, learning_rate))
		b2 = np.subtract(b2, np.multiply(grad_b2, learning_rate))
		model["W1"] = W1
		model["W2"] = W2
		model["b1"] = b1
		model["b2"] = b2

		#keep going

	return model
			

	

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