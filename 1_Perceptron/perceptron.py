import numpy as np

class Perceptron:
	
	# eta			: <float>	learning-rate
	# n_iter		: <int>		iteration
	# random_state	: <int>		random_number

	# ---
	# attributes
	# ---

	# w_			: 1d-array	weight
	# b_			: scalar	intercept
	# errors_		: list		error at each iteration

	def __init__(self, eta=0.01, n_iter=50, random_state=1):

		# eta          : learning rate
		# n_iter       :
		# random_state : 
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self,X,y):
		# ---
		# data training
		# ---
		# X			: array-like, shape = [n_samples,n_features] : Not DataFrame
		#
		# y			: array-like, shape = [n_samples]
		#             target_samples
		#
		# return
		# self.object

		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]) # length of w vector == number of features : X.shape[1] | X.shape[0] > number of data set
		self.b_ = np.float_(0.)
		self.errors_ = []
		# epoch - iteration
		for _ in range(self.n_iter):
			errors = 0
			# looping feature xi in X (row-wise)
			for xi, target in zip(X,y):
				update = self.eta * (target - self.predict(xi))	# <<< predict()
				self.w_ += update * xi
				self.b_ += update
				errors += int(update != 0.)
			self.errors_.append(errors)
		return self
				
	def net_input(self,X):
		# calculate input
		return np.dot(X, self.w_) + self.b_

	def predict(self,X):
		# 'step-function'
		return np.where(self.net_input(X) >= 0.0, 1, 0.)        # <<< net_input()

if __name__=='__main__':
	None
