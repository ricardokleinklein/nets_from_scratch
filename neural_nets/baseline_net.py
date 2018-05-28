# coding: utf-8

import numpy as np
from sklearn import metrics

from utils import *
from modules import *
from itertools import izip
from tqdm import tqdm


class NeuralNet(object):

	def _forward(self, x, is_train):
		"""Forward pass to get the outputs of the model.
		x: input to the model.
		is_train: whether it is training phase or test.
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

	def train(self, data, display=False):
		"""Training stage for the network.
		data: complete task dataset."""
		train_cost = []
		validation_acc = []
		for epoch in tqdm(range(self.epochs)):
			data.set_batches(data.phase['train'], self.batch_size)
			n_batches = len(data.batch)
			minibatch_cost = 0
			for step in range(n_batches):
				x, target = data.next_batch(step)
				act = self._forward(x, is_train=True)
				minibatch_cost += self.layers[-1].get_cost(act[-1], target)
				param_grads = self._backwards(target)
				self._update_params()
			train_cost.append(minibatch_cost / n_batches)
			validation_acc.append(self.evaluate(data, is_train=True))
		if display:
			display_train(self.epochs, train_cost, validation_acc)

	def evaluate(self, data, is_train=False, get_confusion_matrix=True):
		"""Test stage for the network.
		data: complete task dataset."""
		if not is_train:
			data.set_batches(data.phase['test'], len(data.phase['test']))
		else:
			data.set_batches(data.phase['validation'], len(data.phase['validation']))

		n_batches = len(data.batch)
		x, target = data.next_batch(0)
		self._forward(x)
		pred = np.argmax(np.squeeze(self.h[-1]), axis=1)
		y_true = np.argmax(target, axis=1)
		test_accuracy = metrics.accuracy_score(y_true, pred)
		
		if not is_train:
			print('The accuracy on the test set is {:.3f}'.format(test_accuracy))
			if get_confusion_matrix:
				self.get_confusion_matrix(y_true, pred)
		return test_accuracy

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