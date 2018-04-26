# coding: utf-8

import numpy as np

class FFNN():
	"""Fully-connected Feed Forward Neural Network."""
	def __init__(self, hparams):
		self.n_layers = hparams.n_layers
		self.input_size = hparams.input_size
		self.output_size = hparams.output_size
		self.hidden_size = self._ensure_hidden_sizes(hparams)

		self.name = "FFNN w/ %i hidden layers" % hparams.n_layers

	def _ensure_hidden_sizes(self, hparams):
		assert len(hparams.hidden_size) == self.n_layers
		return hparams.hidden_size

	def _init_weights(self):
		pass

	def _loss_function(self):
		pass

	def _train_loop(self):
		pass

	def train(self):
		pass

	def evaluate(self):
		pass