from sklearn import datasets
from sklearn.preprocessing  import normalize, scale
import numpy as np
from numpy.random import shuffle


class Dataset:
	"""Baseline class for datasets."""
	def __init__(self):
		self.X = None
		self.Y = None
		self.is_scaled = False

		self.phase = {'train': None, 'test': None}
		self.batch = None

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]

	def _normalize(self):
		return scale(self.X)

	def _to_one_hot(self):
		num_vals = np.max(self.Y) + 1
		target = np.zeros((self.Y.shape[0], num_vals))
		target[np.arange(len(target)), self.Y] += 1
		return target

	def split_train_test(self, test_size):
		test_size = len(self.X) * test_size // 100
		train_size = len(self.X) - test_size
		idx_list = [i for i in range(len(self.X))]
		shuffle(idx_list)

		if self.is_scaled:
			X = self._normalize()[idx_list]
		else:
			X = self.X[idx_list]
		target = self._to_one_hot()[idx_list]

		train_idx = np.random.choice(idx_list, train_size, replace=False)
		self.phase['train'] = np.asarray([[X[idx], target[idx]] for idx in train_idx])
		test_idx = [i for i in idx_list if i not in train_idx]
		self.phase['test'] = np.asarray([[X[idx], target[idx]] for idx in test_idx])

	def set_batches(self, data, batch_size):
		raise NotImplementedError

	def next_batch(self, step):
		raise NotImplementedError


class smallMNIST(Dataset):
	"""smallMNIST dataset."""
	def __init__(self):
		self.X = datasets.load_digits().images
		self.X = np.asarray([x.flatten() for x in self.X])
		self.Y = datasets.load_digits().target
		self.is_scaled = True

		self.phase = {'train': None, 'test': None}

	def set_batches(self, data, batch_size):
		n_batches = len(data) // batch_size
		shuffle(data)
		batch = [data[i * batch_size: i * batch_size + batch_size]
			for i in range(n_batches)]
		self.batch = np.asarray(batch)

	def next_batch(self, step):
		batch = self.batch[step]
		x, target = batch[:, 0], batch[:, 1]
		x = np.asarray([np.asarray(i) for i in x])
		target = np.asarray([np.asarray(i) for i in target])
		return x, target


class Iris(Dataset):
	"""Iris flowers dataset."""
	def __init__(self):
		self.X = datasets.load_iris().data
		self.Y = datasets.load_iris().target
		self.is_scaled = True

		self.phase = {'train': None, 'test': None}

	def set_batches(self, data, batch_size):
		n_batches = len(data) // batch_size
		shuffle(data)
		batch = [data[i * batch_size: i * batch_size + batch_size]
			for i in range(n_batches)]
		self.batch = np.asarray(batch)
		
	def next_batch(self, step):
		batch = self.batch[step]
		x, target = batch[:, 0], batch[:, 1]
		x = np.asarray([np.asarray(i) for i in x])
		target = np.asarray([np.asarray(i) for i in target])
		return x, target



	
