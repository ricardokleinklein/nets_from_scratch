# coding: utf-8

import numpy as np



def sigmoid(x, is_deriv=False):
		return 1 / (1 + np.exp(-x)) if not is_deriv else x * (1 - x)
		

def relu(x, is_deriv=False):
	return x * (x > 0) if not is_deriv else 1. * (x > 0)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def dropout(x, p):
	return np.random.binomial(1, p, size=x.shape)


def random_mat(in_size, out_size):
	return np.random.randn(in_size, out_size) * np.sqrt(2.0/in_size)


def cross_entropy_loss(pred, target):
	return - np.multiply(target, np.log(pred)).sum() / pred.shape[0]

