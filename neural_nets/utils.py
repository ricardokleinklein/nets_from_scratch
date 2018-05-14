# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, is_deriv=False):
		return 1 / (1 + np.exp(-x)) if not is_deriv else x * (1 - x)
		

def relu(x, is_deriv=False):
	return x * (x > 0) if not is_deriv else 1. * (x > 0)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def dropout(x, p):
	return np.random.binomial(1, p, size=x.shape)


def random_mat(size):
	n_dims = len(size)
	print(size)
	if n_dims == 2:
		return np.random.randn(size[0], size[1]) * np.sqrt(2.0/size[0])
	elif n_dims == 3:
		return np.random.randn(size[0], size[1], size[2]) * np.sqrt(2.0/size[0])
	elif n_dims == 4:
		return np.random.randn(size[0], size[1], size[2], size[3]) * np.sqrt(2.0/size[0])


def cross_entropy_loss(pred, target):
	return - np.multiply(target, np.log(pred)).sum() / pred.shape[0]


def display_train(epochs, train_cost, validation_acc):
	plt.subplot(2,1,1)
	plt.plot(range(epochs), train_cost)
	plt.xlabel('Epoch')
	plt.ylabel('Training cost')

	plt.subplot(2,1,2)
	plt.plot(range(epochs), validation_acc)
	plt.xlabel('Epoch')
	plt.ylabel('Validation accuracy')
	plt.show()


def zero_pad2d(x, pad):
	"""Pad with zeros the height and width of a batch of images.
	Args:
		x (array): input of shape (batch, height, width, channels)
		pad (int): amount of padding.

	Returns:
		x_pad (array): padded image of shape (batch, height + 2*pad, 
			width + 2*pad, channels)
	"""
	x_pad = np.pad(x, ((0,0), (pad, pad), (pad, pad), (0,0)), 'constant')
	return x_pad