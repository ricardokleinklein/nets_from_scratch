# coding: utf-8

import collections
import numpy as np

from itertools import chain
from utils import *

np.random.seed(1)


class Layer(object):
	"""Base class for different layers."""

	def get_params(self):
		"""Return an iterator over the parameters if any."""
		return []

	def get_params_grad(self, x, grad):
		"""Return gradients over the parameters.
		x: input to the layer.
		grad: gradient at the output of the layer.
		"""
		return []

	def _forward(self, x, is_train=False):
		"""Forward step. Computes the output of the layer.
		x: input to the layer.
		is_train: whether it is training phase or test.
		"""
		raise NotImplementedError

	def _backwards(self, pred, grad=None, target=None):
		"""Return gradient at the inputs of the layer.
		grad: gradient at the output of the layer.
		target: target labels to compute gradient from.
		"""
		raise NotImplementedError


class Linear(Layer):
	"""Linear transformation."""
	def __init__(self, input_size, output_size):
		self.W = random_mat(input_size, output_size)
		self.b = random_mat(1, output_size)

	def get_params(self):
		return chain(np.nditer(self.W, op_flags=['readwrite']),
			np.nditer(self.b, op_flags=['readwrite']))

	def get_params_grad(self, x, grad):
		Jw = x.T.dot(grad)
		Jb = np.sum(grad, axis=0)
		return [g for g in chain(np.nditer(Jw), np.nditer(Jb))]

	def _forward(self, x, is_train):
		return x.dot(self.W) + self.b

	def _backwards(self, pred, grad):
		return grad.dot(self.W.T)


class Sigmoid(Layer):

	def _forward(self, x, is_train):
		return sigmoid(x)

	def _backwards(self, pred, grad):
		return np.multiply(sigmoid(pred, is_deriv=True), grad)


class Softmax(Layer):

	def _forward(self, x, is_train):
		return softmax(x)

	def _backwards(self, pred, target):
		return (pred - target) / pred.shape[0]

	def get_cost(self, pred, target):
		return cross_entropy_loss(pred, target)


class Relu(Layer):

	def _forward(self, x, is_train):
		return relu(x)

	def _backwards(self, pred, grad):
		return np.multiply(relu(pred, is_deriv=True), grad)


class Dropout(Layer):

	def __init__(self, dropout_rate):
		self.dist = None
		self.p = dropout_rate

	def _forward(self, x, is_train):
		if not is_train:
			return x
		self.dist = dropout(x, self.p)
		return self.dist * x

	def _backwards(self, pred, grad):
		return self.dist * grad

