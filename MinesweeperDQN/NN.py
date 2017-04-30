import numpy as np

class MLPClassifier(object):
	weights = []
	hidden_layer_sizes = []
	epochs = 100
	epsilon = 0.1

	def __init__(self, hidden_layer_sizes, max_iter=100, epsilon=0.1):
		self.hidden_layer_sizes = list(hidden_layer_sizes)
		self.epochs = max_iter
		self.epsilon = epsilon

	def initializeWeights(self, input_layer_size, output_layer_size):
		np.random.seed(0)
		w = [input_layer_size] + self.hidden_layer_sizes + [output_layer_size]
		for i in range(len(w) - 1):
			self.weights.append(np.random.random((w[i], w[i + 1])) - 1)

	def fit(self, X, y):
		print X
		self.initializeWeights(len(X), 1)
		# print self.weights[2].shape
		for j in range(self.epochs):
			print "Epoch:", j
			l, l_error, l_delta = [], [], []
			l0 = X
			for i in range(len(self.weights)):
				l1 = self.sigmoid(np.dot(l0, self.weights[i]))
				l0 = l1
				l.append(l1)

			# Output layer error
			l_error.insert(0, y - l[len(l) - 1])
			l_delta.insert(0, l_error[len(l_error) - 1] * self.sigmoid(l[len(l) - 1], deriv=True))
			for i in range(len(self.weights) - 1):
				l_error.insert(0, l_delta[0].dot(self.weights[len(self.weights) - i - 1].T))
				l_delta.insert(0, l_error[0] * self.sigmoid(l[len(l) - i - 2], deriv=True))

			# print X.shape
			self.weights[0] += 0.01 * X.T.dot(l_delta[0])
			for i in range(len(self.weights) - 1):
				self.weights[i + 1] += 0.01 * l[i].T.dot(l_delta[i + 1])


			# return self.weights[0], self.weights[1]

	def predict(self, X):
		l0 = X
		for i in range(len(self.weights)):
			l1 = self.sigmoid(np.dot(l0, self.weights[i]))
			l0 = l1
		return np.argmax(l1)

	def sigmoid(self, x, deriv=False):
		if (deriv == True):
			return self.sigmoid(x) * (1 - self.sigmoid(x))
		return 1.0 / (1.0 + np.exp(-x))
