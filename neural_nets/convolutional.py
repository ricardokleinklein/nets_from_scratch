# coding: utf-8

import numpy as np
from sklearn import metrics

from utils import *
from modules import *
from itertools import izip
from tqdm import tqdm

from baseline_net import NeuralNet

class ConvNet(NeuralNet):

	def __init__(self, hparams):
		self.name = 'ConvNet2d'
		self.epochs = hparams.epochs
		self.batch_size = hparams.batch_size
		self.lr = hparams.lr

		self.layers = []
		self.layers.append(Conv2d(3, 8, kernel=3))
		self.layers.append(Sigmoid())
		self.layers.append(Pooling(2, 2, "max"))
		self.layers.append(Conv2d(8, 32, kernel=4, padding=0))
		self.layers.append(Sigmoid())
		self.layers.append(Linear(32, 10))
		self.layers.append(Softmax())

		self.h = None
		self.grads = None

	def _forward(self, x, is_train=False):
		# Case 1 color channel data
		# B x H x W x C
		if len(x.shape) < 4:
			x = np.expand_dims(x, 3)
		assert len(x.shape) == 4

		self.h = [x]
		# Convolutional layers
		for layer in self.layers[:-2]:
			h = layer._forward(x, is_train=is_train)
			self.h.append(h)
			x = self.h[-1]
		h = np.squeeze(x)
		# Linear layers
		for layer in self.layers[-2:]:
			h = layer._forward(x, is_train=is_train)
			self.h.append(h)
			x = self.h[-1]
		return self.h

	def _backwards(self, target):
		param_grads = collections.deque()
		grad = None

		for layer in reversed(self.layers):
			pred = self.h.pop()
			if grad is None:
				in_grad = layer._backwards(pred, target)
			else:
				in_grad = layer._backwards(pred, grad)
			x = self.h[-1]
			grads = layer.get_params_grad(x, grad)
			param_grads.appendleft(grads)
			grad = in_grad
			self.grads = list(param_grads)
		return self.grads

	def _update_params(self):
		assert self.grads is not None
		for layer, layer_grad in zip(self.layers, self.grads):
			for param, grad in izip(layer.get_params(), layer_grad):
				param -= self.lr * grad

