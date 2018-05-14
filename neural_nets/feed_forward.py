# coding: utf-8

import numpy as np
from sklearn import metrics

from utils import *
from modules import *
from itertools import izip
from tqdm import tqdm

from baseline_net import NeuralNet


class Perceptron(NeuralNet):
	def __init__(self, hparams):
		self.name = 'Perceptron'
		self.epochs = hparams.epochs
		self.batch_size = hparams.batch_size
		self.lr = hparams.lr

		self.layers = []
		self.layers.append(Linear(hparams.input_size, hparams.output_size))
		self.layers.append(Softmax())

		self.h = None
		self.grads = None

	def _forward(self, x):
		self.h = [x]
		for layer in self.layers:
			h = layer._forward(x)
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
			self.grads =list(param_grads)
		return self.grads

	def _update_params(self):
		assert self.grads is not None
		for layer, layer_grad in zip(self.layers, self.grads):
			for param, grad in izip(layer.get_params(), layer_grad):
				param -= self.lr * grad


class MLP(NeuralNet):
	def __init__(self, hparams):
		self.name = 'MLP'
		self.epochs = hparams.epochs
		self.batch_size = hparams.batch_size
		self.lr = hparams.lr
		self.non_linear = hparams.non_linear
		self.dropout = hparams.dropout

		self.layers = self.__init__model(hparams.input_size, 
			hparams.hidden_size, hparams.output_size)

		self.h = None
		self.grads = None

	def __init__model(self, input_size, hidden_size, output_size):
		n_layers = len(hidden_size)
		assert n_layers > 0

		layers = []
		layers.append(Linear(input_size, hidden_size[0]))
		for i in range(n_layers):
			layers.append(Relu()) if self.non_linear == 1 else layers.append(Sigmoid())
			layers.append(Dropout(self.dropout))
			if i != (n_layers - 1):
				layers.append(Linear(hidden_size[i], hidden_size[i + 1]))
		layers.append(Linear(hidden_size[-1], output_size))
		layers.append(Softmax())
		return layers

	def _forward(self, x, is_train=False):
		self.h = [x]
		for layer in self.layers:
			h = layer._forward(x, is_train)
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
			self.grads =list(param_grads)
		return self.grads

	def _update_params(self):
		assert self.grads is not None
		for layer, layer_grad in zip(self.layers, self.grads):
			for param, grad in izip(layer.get_params(), layer_grad):
				param -= self.lr * grad

	

	
		
		




	