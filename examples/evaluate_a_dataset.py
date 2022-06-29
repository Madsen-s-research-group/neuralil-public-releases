#!/usr/bin/env python
"""
    This example script reads in a prepared dataset, for which it
        - evaluates and plots the unit cell energies of all structures
        - evaluates and plots the force componenents of all atoms
"""

# imports
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'
          ] = 'false'  # less strain on the system memory

import jax
import jax.nn
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as jnp
import jax.flatten_util
import pathlib
import pickle

from neuralil.training import *

from neuralil.model import Core
from neuralil.model import NeuralIL as NeuralIL
from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator, get_max_number_of_neighbors
)

from ase.io import read

# path to the model(s)
saved_models_path = pathlib.Path(pathlib.Path.cwd() / "trained_models")

# unpickle the selected model
with open(saved_models_path / "model_nnff1.pickle", 'rb') as f:
    model_info = pickle.load(f)
    params = model_info.params

# Load dataset
DATA_DIR = pathlib.Path(
    pathlib.Path(pathlib.Path.cwd() / "datasets" / "validation_nnff1")
)
TEST_LABEL = str(DATA_DIR) + '/nnff1_val'
cells_test = jnp.load(TEST_LABEL + '_cells.npy')
positions_test = jnp.load(TEST_LABEL + '_positions.npy')
types_test = jnp.load(TEST_LABEL + '_types.npy')
energies_test = jnp.load(TEST_LABEL + '_energies.npy')
forces_test = jnp.load(TEST_LABEL + '_forces.npy')
n_test = positions_test.shape[0]

N_EVAL_BATCH = 32  # reduce this if you run into memory issues

# Compute the maximum number of neighbors across all configurations.
max_neighbors = max(
    [
        get_max_number_of_neighbors(p, t, model_info.r_cut, c) for p,
        t,
        c in zip(positions_test, types_test, cells_test)
    ]
)
print(f"{max_neighbors} max_neighbors need to be taken into account")

# initialize the descriptor generator
_pipeline = PowerSpectrumGenerator(
    model_info.n_max,
    model_info.r_cut,
    len(model_info.sorted_elements),
    max_neighbors
)
_core_model = Core(model_info.core_widths)

# setup the model
_model = NeuralIL(
    len(model_info.sorted_elements),
    model_info.embed_d,
    model_info.r_cut,
    _pipeline.process_data,
    _pipeline.process_some_data,
    _core_model
)


# jitted function for energy evaluation
def energy_calculator(params, p, t, c):
    return _model.apply(
        params, p, t, c, method=NeuralIL.calc_potential_energy
    ) * .5


# jitted function for force evaluation
def forces_calculator(params, p, t, c):
    return _model.apply(params, p, t, c, method=NeuralIL.calc_forces)


def calc_energies(params, p, t, c):
    index_from = 0
    index_to = N_EVAL_BATCH
    energies_list = []
    while (index_to < p.shape[0] or index_to == N_EVAL_BATCH):
        index_to = min(index_to + N_EVAL_BATCH, p.shape[0])
        p_batch = p[index_from:index_to, ...]
        t_batch = t[index_from:index_to, ...]
        c_batch = c[index_from:index_to, ...]
        contrib = jax.jit(jax.vmap(energy_calculator, in_axes=(None, 0, 0, 0))
                         )(params, p_batch, t_batch, c_batch)
        energies_list = [*energies_list, contrib]
        index_from = index_to

    energies = jnp.array(
        [e for el in energies_list for e in el]
    )  # sublists may contain a different number of elements

    return energies


def calc_forces(params, p, t, c):
    index_from = 0
    index_to = N_EVAL_BATCH
    forces_list = []
    while (index_to < p.shape[0] or index_to == N_EVAL_BATCH):
        index_to = min(index_to + N_EVAL_BATCH, p.shape[0])
        p_batch = p[index_from:index_to, ...]
        t_batch = t[index_from:index_to, ...]
        c_batch = c[index_from:index_to, ...]
        contrib = jax.jit(jax.vmap(forces_calculator, in_axes=(None, 0, 0, 0))
                         )(params, p_batch, t_batch, c_batch)
        forces_list = [*forces_list, contrib]
        index_from = index_to

    forces = onp.array(
        [f for fl in forces_list for f in fl]
    )  # sublists may contain a different number of elements

    return forces


# evaluate the unit cell energies
print(f"Calculating energies ...")
energies = calc_energies(params, positions_test, types_test, cells_test)
plt.figure()
plt.scatter(energies_test, energies)
plt.title(f"Energies")
plt.xlabel(r"true $E$ / eV")
plt.ylabel(r"predicted $E$ / eV")
plt.tight_layout()
plt.savefig(pathlib.Path(".") / "energies_eval.pdf", dpi=150)
plt.show()

# evaluate all force components
print(f"Calculating forces ...")
forces = calc_forces(params, positions_test, types_test, cells_test)
plt.figure()
plt.scatter(forces_test.flatten(), forces.flatten())
plt.title(f"Force components")
plt.xlabel(r"true $f$ / eV$\,$Å$^{-1}$")
plt.ylabel(r"predicted $f$ / eV$\,$Å$^{-1}$")
plt.tight_layout()
plt.savefig(pathlib.Path(".") / "forces_eval.pdf", dpi=150)
plt.show()
