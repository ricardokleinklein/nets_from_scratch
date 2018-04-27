from sklearn import datasets
from sklearn.preprocessing  import normalize, scale
from numpy.random import shuffle


class Dataset():
	"""Baseline class for datasets."""
	def __init__(self):
		self.X = None
		self.Y = None
		self.is_scale = False

		self.batch = None

	def __getitem__(self, idx):
		return self.X[idx]

	def _get_sample(self, idx):
		return self.X[idx], self.Y[idx]

	def _get_n_samples(self, init, last):
		return self.X[init:last], self.Y[init:last]

	def _get_num_instances(self):
		return len(self.X)

	def _get_num_variables(self):
		return self.X.size[1]

	def _normalize(self):
		return scale(self.X)

	def set_batches(self):
		raise NotImplementedError

	def next_batch(self):
		raise NotImplementedError


class Iris(Dataset):
	"""Iris dataset."""
	def __init__(self):
		iris = datasets.load_iris()
		self.X = iris.data
		self.X = self._normalize()
		self.Y = iris.target

	def set_batches(self, batch_size):
		idx_list = [i for i in range(self._get_num_instances())]
		shuffle(idx_list)
		self.X = self.X[idx_list]
		self.Y = self.Y[idx_list]

		n_batches = self._get_num_instances() // batch_size
		self.batch = [self._get_n_samples(i * batch_size, i * batch_size + batch_size) 
			for i in range(n_batches)]

	def next_batch(self, step):
		return self.batch[step]

	
