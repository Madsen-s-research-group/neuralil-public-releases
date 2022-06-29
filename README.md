# NeuralIL

This repository contains more recent releases of the neural-network force field described in the paper

A Differentiable Neural-Network Force Field for Ionic Liquids

Hadrián Montes-Campos, Jesús Carrete, Sebastian Bichelmaier, Luis M. Varela & Georg K. H. Madsen

Journal of Chemical Information and Modeling 62 (2022) 1, 88–101

DOI: [10.1021/acs.jcim.1c01380](https://doi.org/10.1021/acs.jcim.1c01380)

The original version can be found in the supporting information of the article. These releases contain numerous improvements in terms of features and implementation. Releases used for specific publications will be tagged here.

## Quick start

Three example scripts are provided in the `examples/` subdirectory:

- `train_an_nn.py`: train a model on selected SrTiO3(110) surface structures.
- `evaluate_a_poscar.py`: demonstrate how to evaluate the energy of a SrTiO3(110) surface structure provided as a POSCAR with a trained model.
- `evaluate_a_dataset.py`: demonstrate the evaluation of a dataset of structures, calculating energies and forces with a trained model and plotting the results.

The required data is available in subdirectories:

- `examples/datasets/` contains the training and validation data used in `training.py` and `evaluate_a_dataset.py`, separated into *.npy files.
- `examples/poscars/` contains the POSCAR of a SrTiO3(110) surface structure that is evaluated in `evaluate_a_poscar.py`.
- `examples/trained_models/` contains a model trained over 500 epochs as demonstrated in `train_an_nn.py`.
