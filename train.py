# coding: utf-8
"""
Neural networks from scratch. Expect some fun!
Currently implemented Perceptron and MLP with dropout regularizer,
though it is optional.

Package expected to grow in the future, or not. That depends on
the amount of free time and my will power.

Usage:
	train.py [options] <task_id>

options:
	-h, --help	Show help text
	--verbatim	Display additional information [default: True]
	--metrics		Show additional training metrics	[default:True]
"""

from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple
from tqdm import tqdm
from docopt import docopt

from datasets import Iris, smallMNIST
from neural_nets import *

from hparams import hparams
import numpy as np

import matplotlib.pyplot as plt

def get_hparams(hparams):
	return namedtuple('hparams',hparams.keys())(*hparams.values())


def print_network(model):
	print(model.name + " = { ")
	for layer in model.layers:
		name = str(layer).split(" ")[0].split(".")[-2:]
		print("\t" + "/".join(name))
	print("}\n")


def print_hparams(hparams):
	print("hparams = { ")
	for key, value in vars(hparams).items():
		print("\t" + str(key) + " = " + str(value))
	print("}\n")


def get_data(task_id, test_size):
	if task_id == "digits":
		data = smallMNIST(flatten=False)
	elif task_id == "iris":
		data = Iris()
	else:
		raise IOError
	data.split_train_test(test_size)
	return data


if __name__ == "__main__":
	hparams = get_hparams(hparams)
	args = docopt(__doc__)
	task = args["<task_id>"]

	data = get_data(task, hparams.test_size)

	model = ConvNet(hparams)
	print("Initializing the neural network")
	print_hparams(hparams)
	print_network(model)
	

	print("Training stage")
	model.train(data, display=False)

	# print("Test stage")
	# model.evaluate(data, get_confusion_matrix=False)
