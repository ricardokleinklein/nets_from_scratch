# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple
from tqdm import tqdm
from docopt import docopt

from datasets import Iris
from neural_nets import FFNN

from hparams import hparams


def get_hparams(hparams):
	return namedtuple('hparams',hparams.keys())(*hparams.values())


if __name__ == "__main__":
	X, labels = Iris().get_data()

	hparams = get_hparams(hparams)

	model = FFNN(hparams)
	print(model.hidden_size)
