import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def plot_3d(X, Y, Z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(X, Y, Z)
	ax.text2D(0.05, 0.95, "Plot for cost vs w1 and w2", transform=ax.transAxes)
	ax.set_xlabel('w1')
	ax.set_ylabel('w2')
	ax.set_zlabel('Cost')
	plt.show()

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Plot for cost vs iterations') 
    plt.savefig('2_plt_1.png')
    plt.close()

def loss_test_data(X, y, theta):
	y_pred = X.dot(theta)
	loss = (0.5/(y.size))*sum((y-y_pred)**2)
	return loss

def gradient_descent(x, y, theta, alpha, n):
	i = 0
	iterations = []
	losses = []
	w1 = []
	w2 = []
	while(i<n):
		i += 1
		prediction = x.dot(theta)
		loss = (0.5/(y.size))*sum((y-prediction)**2)
		for j in range(len(prediction)):
			error = prediction[j] - y[j]
			# print(x[j].shape)
			theta = theta - (alpha * np.dot(x[j], error))
		iterations.append(i)
		losses.append(loss)
		w1.append(theta[1])
		w2.append(theta[2])
	return theta, losses, iterations, w1, w2


# Importing the dataset
dataset = pd.read_excel('data.xlsx')

X_train = dataset.iloc[:, 0:2].values
X_train = np.insert(X_train, 0, np.ones(349), 1)
Y_train = dataset.iloc[:, 2].values

# Gradient Descent Method
np.random.seed(10)
theta = np.random.rand(3)
theta, losses, iterations, w1, w2 = gradient_descent(X_train, Y_train, theta, 0.000000001, 100)
loss = loss_test_data(X_train, Y_train, theta)
print("Stochiastic Gradient Descent")
print("Weights : ", end='')
print(theta)
print("Loss on test data : ", end='')
print(round(loss,3))

# Plots
plot(iterations, losses)
plot_3d(w1, w2, losses)