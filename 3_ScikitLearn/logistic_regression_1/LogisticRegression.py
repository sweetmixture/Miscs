#
#
#
import numpy as np

class LogisticRegressionGD:

	'''
		* parameters

		eta : float learning rate [0:1]

		n_iter : int epoch

		random_state : int

		* attributes

		w_ : 1d-array length .eq. n_features

		b_ : scalar

		losses_ : list

	'''

	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self,X,y):

		'''
			* parameter

			X : array-like, shape = [n_samples, n_features]

			y : array-like, shape = [n_samples]

			* return self
		'''

		rgen = np.random.RandomState(self.random_state)

		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
		self.b_ = np.float_(0.)
		self.losses_ = []

		for i in range(self.n_iter):

			net_input = self.net_input(X)			# do (net_input)  z = wTx + b
			output    = self.activation(net_input)	# do (output) = sigmoid(z)

			errors    = (y - output) # this is not the same with 'loss'

			# update 
			self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]	# here X.T include row (sample) column (features)
			self.b_ += self.eta * 2.0 * errors.mean()

			loss = ( -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output)) )/X.shape[0]

			self.losses_.append(loss)

		return self

	def net_input(self, X):
		return np.dot(X, self.w_) + self.b_

	def activation(self, z):
		''' logistic sigmoid type activation function '''
		return 1./(1. + np.exp(-np.clip(z,-250,250)))

	def predict(self, X):
		''' after training '''
		return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

if __name__=='__main__':

	None
