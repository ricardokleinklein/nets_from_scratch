from sklearn import datasets
from sklearn.preprocessing  import normalize, scale
import numpy as np
from numpy.random import shuffle


class Dataset():
	"""Baseline class for datasets."""
	def __init__(self):
		self.X = None
		self.Y = None

		self.phase = {'train': None, 'test': None}
		self.batch = None

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]

	def _normalize(self):
		return scale(self.X)

	def _to_one_hot(self):
		size = len(self.X)
		n_values = np.max(self.Y) + 1
		one_hot = np.zeros((size, n_values))
		for i in range(size):
			one_hot[i, self.Y[i]] = 1
		return one_hot

	def split_train_test(self, test_size):
		test_size = len(self.X) * test_size // 100
		train_size = len(self.X) - test_size
		idx_list = [i for i in range(len(self.X))]
		shuffle(idx_list)

		X = self._normalize()[idx_list]
		label = self._to_one_hot()[idx_list]

		size = len(self.X)
		self.phase['train'] = [(X[i], label[i]) for i in range(train_size)]
		self.phase['test'] = [(X[train_size + i], label[train_size + i]) 
			for i in range(test_size)]

	def set_batches(self, data, batch_size):
		raise NotImplementedError

	def next_batch(self, step):
		raise NotImplementedError



class Iris(Dataset):
	"""Iris flowers dataset."""
	def __init__(self):
		self.X = datasets.load_iris().data
		self.Y = datasets.load_iris().target

		self.phase = {'train': None, 'test': None}

	def set_batches(self, data, batch_size):
		n_batches = len(data) // batch_size
		shuffle(data)
		self.batch = [data[i * batch_size: i * batch_size + batch_size]
			for i in range(n_batches)]
		
	def next_batch(self, step):
		return self.batch[step]



	
