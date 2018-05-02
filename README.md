# toy_nets

## Overview

General neural network framework from scratch. 
Currently only Vanilla Feedforward Neural net.

**Only tested on Linux**

## Requirements

- python = 2.7
- numpy >= 1.14.2
- matplotlib >= 1.3.1
- scikit-learn >= 0.19.1
- tqdm >= 4.23.1
- docopt >= 0.6.2

The recommended installation pipeline for the packages `scikit-learn`, `tqdm` and `docopt` is via `pip`. Thus,
```
pip install sklearn tqdm docopt
```

## Installation and Execution

If it's downloaded as a `zip` file, unzip it in an environment with the above mentioned requirements and type in:
```
python train.py <task_id>
```
Tasks available so far are `iris` and `digits`. E.g,
```
python train.py iris
```

In case this pipeline doesn't work for you and you need a downloadable and auto-installable package with the right environment and dependencies installed, please let me know and I'll send you further instructions.

## Design decisions

The whole package is set as a self-contained framework, only depending on `numpy` (and for plotting purposes, `matplotlib`). 

However, for the sake of simplicity, both datasets on their raw forms are loaded from `scikit-learn`. Nonetheless, the rest of the pipeline (train/validation/test setup, batch criteria) is built from scratch. The implementation can be found in `/datasets/datasets.py`.

Two neural network architectures are implemented so far: basic Perceptron and general MLP.
The output layer activation function is a **softmax layer**, for both available tasks so far are classification problems. Therefore, the loss function chosen is **cross-entropy loss**.
The activation layer for hidden layers can be set to be either **logistic sigmoid** or **relu**. Usually better results are achieved with **relu**. The weights of the net are initialized accordingly to [*Understanding the difficulty of training deep feedforward neural networks*, Glorot, X., Bengio, Y. (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).

**Dropout** regularization has been implemented. Depending on the size of the neural network its use can be not recommended, though. For very small nets dropout can worse performance.

The default hyperparameters of the model can be found in `hparams.py`, and can be freely modified. It is important to point out that `hparams.input_size` and `hparams.output_size` must be manually changed accordingly to the task to perform (`4, 3` for iris, `64, 10` for digits).

## Results attached

In the directory `EXP` included within the `zip` file you can find two files.
- `train_stats.png`: Displays the training cost along the training steps and the accuracy achieved when evaluating the model on the validation set after each training step.
- `confusion_matrix.png`: Shows the confusion matrix obtained when the trained model is evaluated over the test set.

## Contact

If you have any question or you think any aspect of the package is unclear, please do not hesitate to contact the author.

- e-mail: `ricardokleinklein@gmail.com` 

