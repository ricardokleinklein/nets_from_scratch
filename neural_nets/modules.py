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
		self.W = random_mat((input_size, output_size))
		self.b = random_mat((1, output_size))

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


class Conv2d(Layer):
	"""2D Convolutional layer."""
	def __init__(self, input_channels, output_channels, kernel,
		stride=1, padding=1):
		self.W = random_mat((kernel, kernel, input_channels, output_channels))
		self.b = random_mat((1, 1, 1, output_channels))
		self.stride = stride
		self.pad = int(padding)
		self.f = kernel

	def get_params(self):
		return chain(np.nditer(self.W, op_flags=['readwrite']),
			np.nditer(self.b, op_flags=['readwrite']))

	def get_params_grad(self, x, grad):
		(f, f, C_prev, C) = self.W.shape
		(m, H, W, C) = grad.shape
		dW = np.zeros(grad.shape)
		db = np.zeros((1, 1, 1, C))
		A_prev_grad = zero_pad2d(x, self.pad)
		dA_prev_grad = zero_pad2d(grad, self.pad)

		for i in range(m):
			a_prev_pad = A_prev_grad[i]
			da_prev_pad = dA_prev_grad[i]
			for h in range(H):
				for w in range(W):
					for c in range(C):
						vert_start = h * self.stride
						vert_end = h * self.stride + self.f
						horiz_start = w * self.stride
						horiz_end = w * self.stride + self.f

						a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
						dW[:, :, :, c] += a_slice * grad[i, h, w, c]
						db[:, :, :, c] += grad[i, h, w, c]
		return [g for g in chain(np.nditer(dW), np.nditer(db))]


	def _forward_partial(self, x, c):
		return np.sum(np.multiply(x, self.W[:, :, :, c]) + float(self.b[:, :, :, c]))

	def _forward(self, x, is_train):
		(m, n_H_prev, n_W_prev, n_C_prev) = x.shape
		(f, f, n_C_prev, n_C) = self.W.shape

		n_H = int((n_H_prev - f + 2 * self.pad) / self.stride + 1)
		n_W = int((n_W_prev - f + 2 * self.pad) / self.stride + 1)
		
		Z = np.zeros((m, n_H, n_W, n_C))
		X_prev_pad = zero_pad2d(x, self.pad)
		for i in range(m):                               
			x_prev_pad = X_prev_pad[i]                      
			for h in range(n_H):                        
				for w in range(n_W):                     
					for c in range(n_C):                   

						vert_start = h * self.stride
						vert_end = h * self.stride + f
						horiz_start = w * self.stride
						horiz_end = w * self.stride + f
						
						x_slice_prev = x_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
						# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
						Z[i, h, w, c] = self._forward_partial(x_slice_prev, c)

		assert(Z.shape == (m, n_H, n_W, n_C))
		return Z

	def _backwards(self, pred, grad):
		(m, H_prev, W_prev, C_prev) = pred.shape
		(m, H, W, C) = grad.shape
		dA_prev = np.zeros((m, H_prev, W_prev, C_prev))
		dA_prev_pad = zero_pad2d(dA_prev, self.pad)
		for i in range(m):
			da_prev_pad = dA_prev_pad[i]
			for h in range(H):
				for w in range(W):
					for c in range(C):

						vert_start = h * self.stride
						vert_end = h * self.stride + self.f
						horiz_start = w * self.stride
						horiz_end = w * self.stride + self.f
						print(self.W[:, :, :, c].shape, grad[i, h, w, c].shape)
						da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, :, c] * grad[i, h, w, c]
			dA_prev[i, :, :, :] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
		assert(dA_prev.shape == (m, H_prev, W_prev, C_prev))
		return dA_prev


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


class Pooling(Layer):

	def __init__(self, kernel, stride, mode):
		self.f = kernel
		self.s = stride
		self.mode = mode

	def _forward(self, x, is_train):
		(m, n_H_prev, n_W_prev, n_C_prev) = x.shape
		n_H = int(1 + (n_H_prev - self.f) / self.s)
		n_W = int(1 + (n_W_prev - self.f) / self.s)
		n_C = n_C_prev

		A = np.zeros((m, n_H, n_W, n_C))
		for i in range(m):
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):

						vert_start = h * self.s
						vert_end = h * self.s + self.f
						horiz_start = w * self.s
						horiz_end = w * self.s + self.f

						x_prev_slice = x[i, vert_start:vert_end, horiz_start:horiz_end, c]
						if self.mode == "max":
							A[i, h, w, c] = np.max(x_prev_slice)
						elif self.mode == "average":
							A[i, h, w, c] = np.mean(x_prev_slice)
		assert(A.shape == (m, n_H, n_W, n_C))
		return A

	def _backwards(self, pred, grad):
		m, H_prev, W_prev, C_prev = pred.shape
		m, H, W, C = grad.shape
		grad_prev = np.zeros(pred.shape)

		for i in range(m):
			pred_prev = pred[i]
			for h in range(H):
				for w in range(W):
					for c in range(C):
						vert_start = h * self.s
						vert_end = h * self.s + self.f
						horiz_start = w * self.s
						horiz.end = w * self.s + self.f

						if self.mode == 'max':
							pred_prev_slice = pred_prev[vert_start:vert_end, horiz_start:horiz_end, c]
							mask = create_mask(pred_prev_slice)
							grad_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * grad[i, h, w, c]
						elif self.mode == 'average':
							dg = grad[i, h, w, c]
							shape = (self.f, self.f)
							grad_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(dg, shape)
		assert grad_prev.shape == pred.shape
		return grad_prev


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

