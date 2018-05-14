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

		self.layers = []
		# self.layers.append(Conv2d(3, 8, 2))
		# self.layers.append(Softmax())
		self.layers.append(Pooling(3, 2, "max"))

	def _forward(self, x, is_train=False):
		for layer in self.layers:
			h = layer._forward(x, is_train)
		return h

	def train(self, x, display=False):
		return self._forward(x)


	def _backwards(self, target):
		pass

	def _update_params(self):
		pass