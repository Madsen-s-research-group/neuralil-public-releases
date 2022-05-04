# NeuralIL

This repository contains more recent releases of the neural-network force field described in the paper

A Differentiable Neural-Network Force Field for Ionic Liquids

Hadrián Montes-Campos, Jesús Carrete, Sebastian Bichelmaier, Luis M. Varela & Georg K. H. Madsen

Journal of Chemical Information and Modeling 62 (2022) 1, 88–101

DOI: [10.1021/acs.jcim.1c01380](https://doi.org/10.1021/acs.jcim.1c01380)

The original version can be found in the supporting information of the article. These releases contain numerous improvements in terms of features and implementation. Releases used for specific publications will be tagged here.

## Quick start

Two example training scripts are provided in the `examples/` subdirectory:

- `train_an_nn.py`, which uses parameters similar to those in the article.

- `train_snn.py`, which removes the LayerNorm in favor of the principles of  [bidirectionally self-normalizing neural networks](https://arxiv.org/abs/2006.12169), keeping the weights matrices pseudo-orthogonal and changing the activation functions to be Gaussian-Poincaré normalized.

The set of parameters resulting from running the first script for 300 epochs is also included.

A third example script, `md_example`, shows how to use the ASE calculator interface for inference, in particular to run a simple molecular dynamics simulation.
