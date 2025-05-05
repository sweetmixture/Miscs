#
# 05.05.2025
#
import numpy as np

class AdalineGD:
	'''
		Adaptive type neuron classifier

		* parameters

		eta : float [0:1.0] learning rate

		n_iter : int training iteration number

		random_state : int

		* attributes

		w_ : 1d-array weight vector (trained)

		b_ : scalar

		losses_ : list averaged RMSD for each epoch

	'''

	def __init__(self, eta=0.01, n_iter=50, random_state=1):

		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self,X,y):
		'''
			training data set

			* parameters

			X : {array-like}, shape = [n_samples,n_features] row: samples, columns: features

			y : array-like, shape = [n_samples] target

			* return

			self
		'''

		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # note. w_ size is equal to the size of features

		self.b_ = np.float_(0.)
		self.losses_ = []

		for i in range(self.n_iter):

			net_input = self.net_input(X)
			output = self.activation(net_input)

			errors = (y - output)
			
			# update
			self.w_ += self.eta * 2.0 * (X.T).dot(errors) / X.shape[0]
			self.b_ += self.eta * 2.0 * errors.mean()
			#self.b_ += self.eta * 2.0 * errors / X.shape[0] : this is not same with the line right above

			loss = (errors**2.).mean()
			self.losses_.append(loss)

		return self

	def net_input(self, X):
		# simply do operation : w.X + b
		return np.dot(X, self.w_) + self.b_

	def activation(self, X):
		# simply return the raw input with any processing
		return X

	def predict(self, X):
		# return class label through step function
		return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0.)

class AdalineSGD:
	'''
		Adaptive type neuron classifier

		* parameters

		eta : float [0:1.0] learning rate

		n_iter : int training iteration number

		random_state : int

		(NEW)
		shuffle : bool (default = True) True > mixes training data on each epoch

		* attributes

		w_ : 1d-array weight vector (trained)

		b_ : scalar

		losses_ : list averaged RMSD for each epoch
	'''
	def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):

		self.eta = eta
		self.n_iter = n_iter

		self.w_initialised = False
		self.shuffle = shuffle
		self.random_state = random_state

	def fit(self,X,y):
		'''
			training data set

			* parameters

			X : {array-like}, shape = [n_samples,n_features] row: samples, columns: features

			y : array-like, shape = [n_samples] target

			* return

			self
		'''
	
		self._initialise_weights(X.shape[1]) # init : self.w_ & self.b_ // set self.w_initialised = True
		self.losses_ = []

		for i in range(self.n_iter):

			if self.shuffle:
				X, y = self._shuffle(X,y) # shuffle (re-ordering randomly)

			losses = []
			for xi, target in zip(X,y): # 'i' indicate dataset '(i)'
				losses.append(self._update_weights(xi,target))
			avg_loss = np.mean(losses)
			self.losses_.append(avg_loss)

		return self

	def partial_fit(self, X, y):
		# without weight initialisation
		if not self.w_initialised:
			self._initialise_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi, target in zip(X,y):
				self._update_weights(xi,target)
		else:
			self._update_weights(X,y)

		return self

	def _shuffle(self,X,y):
		
		r = self.rgen.permutation(len(y))
		return X[r], y[r]

	def _initialise_weights(self,m):
		self.rgen = np.random.RandomState(self.random_state)
		self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)

		self.b_ = np.float_(0.)
		self.w_initialised = True

	def _update_weights(self, xi, target):

		output = self.activation(self.net_input(xi))
		error = (target - output)
		self.w_ += self.eta * 2.0 * xi * (error)
		self.b_ += self.eta * 2.0 * error
		loss = error**2.
		return loss

	def net_input(self,X):
		return np.dot(X,self.w_) + self.b_

	def activation(self,X):
		return X

	def predict(self,X):
		return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

if __name__ == '__main__':
	None













