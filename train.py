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


if __name__ == "__main__":
	hparams = get_hparams(hparams)

	digits = smallMNIST()
	digits.split_train_test(hparams.test_size)

	iris = Iris()
	iris.split_train_test(hparams.test_size)

	model = MLP(hparams)
	model.train(iris)
	model.evaluate(iris)
