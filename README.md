# NeuralVecmap (DNN-based embedding mapping)

This is an open source implementation of the nonlinear mapping between embedding sets used in this paper:

- D Newman-Griffis and A Zirikly, ["Embedding Transfer for Low-Resource Medical Named Entity
Recognition: A Case Study on Patient Mobility"](http://drgriffis.github.io/papers/2018-BioNLP.pdf). In _Proceedings of BioNLP 2018_, 2018.

The included `demo.sh` script will download two small sets of embeddings, learn a demonstration mapping between them, and calculate changes in nearest neighbors.

## Dependencies

External

- [Tensorflow](https://www.tensorflow.org/) (we used version 1.3.1)
- NumPy

Internal (frozen copies of all included in the `lib` directory)

- [pyemblib](https://github.com/drgriffis/pyemblib)
- [configlogger](https://github.com/drgriffis/configlogger)
- [custom logging utility](https://github.com/drgriffis/miscutils/blob/master/py/drgriffis/common/logging.py)

## Method description

This implementation learns a nonlinear mapping function from a _source_ set of embeddings to a _target_ set,
based on shared keys (pivots).  The embeddings do not have to be of the same dimensionality, but must have keys in common.

The process follows three steps:

### (1) Identification of pivots

Pivot terms used in the mapping process may be selected from the set of keys present in both the source
and target embeddings in one of two ways:

- <span style="text-decoration:underline">Frequent keys</span>: the top _N_ keys by frequency in the target corpus are used as pivots.
- <span style="text-decoration:underline">Random/all keys</span>: a random subset of _N_ shared keys (or all shared keys, if _N_ is unspecified) is used as pivots.

### (2) Learning k-fold projections

Pivot terms are divided into _k_ folds. For each fold, a nonlinear projection is learned as follows:

1. Construct a feed-forward DNN, taking source embeddings as input and generating output of the same size as target embeddings.  Model parameters include:
    - Number of layers
    - Activation function (tanh or ReLU)
    - Dimensionality of hidden layers (by default, same as target embedding size)
2. Use minibatch gradient descent to train over each shared key in the training set
    - Loss function is batch-wise MSE between output embeddings and reference target embeddings
    - Optimization with Adam
3. After each epoch (all shared keys in training set), evaluate MSE on held-out set
4. When held-out MSE stops decreasing, stop training and revert to previous best model parameters

### (3) Generating final transformation

Getting the final projection of source embeddings into target embedding space is a two-step process:

1. Take the projection function learned for each trained fold and project all source embeddings
2. Average all _k_ projections to yield final projection of source embeddings

## Nearest neighbor analysis

This repository also includes the code used to calculate changes in nearest neighbors after the learned mapping is applied, in `nn-analysis`.

- `nearest_neighbors.py` Tensorflow implementation of nearest neighbor calculation by cosine distance
- `nn_changes.py` script to calculate how often nearest neighbors change after the mapping is learned

## Reference

If you use this software in your own work, please cite the following paper:

```
@inproceedings{Newman-Griffis2018BioNLP,
  author = {Newman-Griffis, Denis and Zirikly, Ayah},
  title = {Embedding Transfer for Low-Resource Medical Named Entity Recognition: A Case Study on Patient Mobility},
  booktitle = {Proceedings of BioNLP 2018},
  year = {2018}
}
```
