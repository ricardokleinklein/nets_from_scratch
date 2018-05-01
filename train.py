# coding: utf-8

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


if __name__ == "__main__":
	hparams = get_hparams(hparams)

	digits = smallMNIST()
	digits.split_train_test(hparams.test_size)

	iris = Iris()
	iris.split_train_test(hparams.test_size)

	model = MLP(hparams)
	print("Initializing the net")
	print_network(model)
	print_hparams(hparams)

	print("Training stage")
	model.train(digits)

	print("Test stage")
	model.evaluate(digits)
