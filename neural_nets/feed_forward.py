# coding: utf-8

import numpy as np
from sklearn import metrics

from utils import *
from modules import *
from itertools import izip
from tqdm import tqdm


class NeuralNet(object):

	def _forward(self, x):
		"""Forward pass to get the outputs of the model.
		x: input to the model
		"""
		raise NotImplementedError

	def _backwards(self, target):
		"""Backpropagation algorithm to update the network.
		target: one-hot encoding of the true labels.
		"""
		raise NotImplementedError

	def _update_params(self):
		"""SGD updating of the model's parameters."""
		raise NotImplementedError

	def train(self, data):
		"""Training stage for the network.
		data: complete task dataset."""
		for epoch in tqdm(range(self.epochs)):
			data.set_batches(data.phase['train'], self.batch_size)
			n_batches = len(data.batch)
			for step in range(n_batches):
				x, target = data.next_batch(step)
				act = self._forward(x)
				batch_cost = self.layers[-1].get_cost(act[-1], target)
				param_grads = self._backwards(target)
				self._update_params()

	def evaluate(self, data, get_confusion_matrix=True):
		"""Test stage for the network.
		data: complete task dataset."""
		data.set_batches(data.phase['test'], len(data.phase['test']))
		n_batches = len(data.batch)
		x, target = data.next_batch(0)
		self._forward(x)
		pred = np.argmax(self.h[-1], axis=1)
		y_true = np.argmax(target, axis=1)
		test_accuracy = metrics.accuracy_score(y_true, pred)
		print('The accuracy on the test set is {:.3f}'.format(test_accuracy))
		if get_confusion_matrix:
			self.get_confusion_matrix(y_true, pred)

	def get_confusion_matrix(self, y_true, pred):
		import matplotlib.pyplot as plt
		from matplotlib import figure
		from matplotlib.colors import colorConverter, ListedColormap
		conf_matrix = metrics.confusion_matrix(y_true, pred, labels=None) 
		n_vals = np.max(y_true)
		class_names = ['${:d}$'.format(x) for x in range(0, n_vals + 1)]  # class names
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.xaxis.tick_top()
		major_ticks = range(0,n_vals + 1)
		minor_ticks = [x + 0.5 for x in range(0, n_vals + 1)]
		ax.xaxis.set_ticks(major_ticks, minor=False)
		ax.yaxis.set_ticks(major_ticks, minor=False)
		ax.xaxis.set_ticks(minor_ticks, minor=True)
		ax.yaxis.set_ticks(minor_ticks, minor=True)
		ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
		ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
		ax.yaxis.set_label_position("right")
		ax.set_xlabel('Predicted label')
		ax.set_ylabel('True label')
		fig.suptitle('Confusion table', y=1.03, fontsize=15)
		ax.grid(b=True, which=u'minor')
		ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
		# Show the number of samples in each cell
		for x in xrange(conf_matrix.shape[0]):
			for y in xrange(conf_matrix.shape[1]):
				color = 'w' if x == y else 'k'
				ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       
		plt.show()


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
			if i != (n_layers - 1):
				layers.append(Linear(hidden_size[i], hidden_size[i + 1]))
		layers.append(Linear(hidden_size[-1], output_size))
		layers.append(Softmax())
		return layers

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

	

	
		
		




	