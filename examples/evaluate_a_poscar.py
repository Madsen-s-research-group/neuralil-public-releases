#!/usr/bin/env python
"""
    This example script reads in a compatible POSCAR, for which it
        - evaluates and prints the unit cell energy
"""

# imports
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'
          ] = 'false'  # less strain on the system memory

import jax
import jax.nn
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

# path to the model
saved_models_path = pathlib.Path.cwd() / "trained_models"

# unpickle the selected model
with open(saved_models_path / "model_nnff1.pickle", 'rb') as f:
    model_info = pickle.load(f)
    params = model_info.params

# load from POSCAR
atoms = read(pathlib.Path.cwd() / "poscars" / "POSCAR_3x1a")
duplicated = atoms * (
    2, 1, 1
)  # the unit cell has to be duplicated to fit the cutoff radius
cells = duplicated.cell[...]
positions = duplicated.get_positions()
symbols = duplicated.get_chemical_symbols()
sorted_elements = sorted(["O", "Sr", "Ti"])
symbol_map = {s: i for i, s in enumerate(sorted_elements)}
n_types = len(symbol_map)
types = []
types = jnp.array([symbol_map[i] for i in symbols])

# Compute the maximum number of neighbors for the chosen configuration.
max_neighbors = get_max_number_of_neighbors(
    positions, types, model_info.r_cut, cells
)
print(
    f"Maximum numbers of neighbors is {max_neighbors} in the chosen structure."
)

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
@jax.jit
def full_energy(p, t, c):
    return _model.apply(params, p, t, c, method=NeuralIL.calc_potential_energy)


# evaluate the unit cell energy
e = full_energy(
    positions, types, cells
) * .5  # divided by 2 because of the duplicated unit cell
print(f"The energy of the unit cell is {e} eV")
