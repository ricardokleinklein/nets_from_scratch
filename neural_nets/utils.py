# coding: utf-8

import numpy as np



def sigmoid(x, is_deriv=False):
		return 1 / (1 + np.exp(-x)) if not is_deriv else x * (1 - x)
		

def relu(x, is_deriv=False):
	return x * (x > 0) if not is_deriv else 1. * (x > 0)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def random_mat(size):
	return (2 * np.random.random(size=size) - 1) * 0.1


def cross_entropy_loss(pred, target):
	return - np.multiply(target, np.log(pred)).sum() / pred.shape[0]

