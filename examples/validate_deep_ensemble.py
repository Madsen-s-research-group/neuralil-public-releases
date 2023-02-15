#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import copy
import datetime
import json
import pathlib
import pickle
import sys

import ase.db
import flax
import flax.jax_utils
import flax.serialization
import jax
import jax.nn
import jax.numpy as jnp
import tqdm
import tqdm.auto

from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
    get_max_number_of_neighbors,
)
from neuralil.deep_ensembles.model import (
    DeepEnsemble,
    HeteroscedasticNeuralIL,
    HeteroscedasticNeuralILModelInfo,
    update_energy_offset,
)
from neuralil.deep_ensembles.training import *
from neuralil.training import (
    create_heteroscedastic_log_cosh,
    create_velo_minimizer,
)
from neuralil.utilities import *

# Predefined parameters for this run.
TRAINING_FRACTION = 0.9
R_CUT = 3.5
N_MAX = 4
EMBED_D = 2
WEIGHT_ENERGY = 0.5

N_EVAL_BATCH = 32

LOG_COSH_PARAMETER = 0.5

FEATURE_EXTRACTOR_WIDTHS = [64, 32, 16]
HEAD_WIDTHS = [16]

N_ENSEMBLE = 10

# Generate a random seed and the associated code
run_seed = 31337  # draw_urandom_int32()
run_seed_str = f"{run_seed:0X}"
rng = jax.random.PRNGKey(run_seed)

NN_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "auxiliary_files"
    / "EAN_params_deep_ensemble_8ABE86C9.pkl"
).resolve()

# Create a map from element names to integers. They must start at zero
# and be contiguous. They must also be consistent between the training
# and inference phases. Sorting the element names alphabetically is an
# easy and reasonable choice. Naturally, the set of elements can be
# extracted from the configurations themselves if this step is deferred.
sorted_elements = sorted(["C", "H", "O", "N"])
symbol_map = {s: i for i, s in enumerate(sorted_elements)}
n_types = len(symbol_map)

# The atom types are the same in every configuration and not stored in the file.
type_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
type_anion = ["N", "O", "O", "O"]

# Load the training and validation data.
cells_train = []
positions_train = []
energies_train = []
forces_train = []
print("- Reading the training JSON file")
with open(
    (
        pathlib.Path(__file__).parent
        / "original_training_and_validation"
        / "training.json"
    ).resolve(),
    "r",
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells_train.append(jnp.diag(jnp.array(json_data["Cell-Size"])))
        positions_train.append(json_data["Positions"])
        energies_train.append(json_data["Energy"])
        forces_train.append(json_data["Forces"])
print("- Done")
n_train = len(positions_train)
positions_train = jnp.asarray(positions_train)
cells_train = jnp.asarray(cells_train)
energies_train = jnp.asarray(energies_train)
forces_train = jnp.asarray(forces_train)

n_pair = len(positions_train[0]) // (len(type_anion) + len(type_cation))
types = n_pair * type_cation + n_pair * type_anion
unique_types = sorted(set(types))

types_train = jnp.asarray([[symbol_map[i] for i in types]] * n_train)

cells_validate = []
positions_validate = []
energies_validate = []
forces_validate = []
print("- Reading the validation JSON file")
with open(
    (
        pathlib.Path(__file__).parent
        / "original_training_and_validation"
        / "validation.json"
    ).resolve(),
    "r",
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells_validate.append(jnp.diag(jnp.array(json_data["Cell-Size"])))
        positions_validate.append(json_data["Positions"])
        energies_validate.append(json_data["Energy"])
        forces_validate.append(json_data["Forces"])
print("- Done")
n_validate = len(positions_validate)
positions_validate = jnp.asarray(positions_validate)
cells_validate = jnp.asarray(cells_validate)
energies_validate = jnp.asarray(energies_validate)
forces_validate = jnp.asarray(forces_validate)
types_validate = jnp.asarray([[symbol_map[i] for i in types]] * n_validate)

n_configurations = n_train + n_validate

print(f"- {n_configurations} configurations are available")
print(f"\t- {n_train} will be used for training")
print(f"\t- {n_validate} will be used for validation")


# Compute the maximum number of neighbors across all configurations.
max_neighbors = max(
    [
        get_max_number_of_neighbors(p, t, R_CUT, c)
        for p, t, c in zip(positions_train, types_train, cells_train)
    ]
)
max_neighbors = max(
    max_neighbors,
    max(
        [
            get_max_number_of_neighbors(p, t, R_CUT, c)
            for p, t, c in zip(
                positions_validate, types_validate, cells_validate
            )
        ]
    ),
)

print("- Maximum number of neighbors that must be considered: ", max_neighbors)

# Create the object that will generate descriptors for each configuration.
descriptor_generator = PowerSpectrumGenerator(
    N_MAX, R_CUT, n_types, max_neighbors
)

# Create the model. The number and types of the parameters is completely
# dependent on the kind of model used.
individual_model = HeteroscedasticNeuralIL(
    n_types,
    EMBED_D,
    R_CUT,
    descriptor_generator.process_data,
    FEATURE_EXTRACTOR_WIDTHS,
    HEAD_WIDTHS,
)
dynamics_model = DeepEnsemble(individual_model, N_ENSEMBLE)

# Load the parameters.
rng, init_rng = jax.random.split(rng)
template_params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=dynamics_model.calc_all_results,
)
model_info = pickle.load(open(NN_FILE, "rb"))
model_params = jax.tree_map(
    jnp.asarray,
    flax.serialization.from_state_dict(template_params, model_info.params),
)


def calc_averages(positions, types, cell):
    (
        pred_energy,
        pred_forces,
        sigma2_energy,
        sigma2_forces,
    ) = dynamics_model.apply(
        model_params,
        positions,
        types,
        cell,
        method=dynamics_model.calc_all_results,
    )
    average_energy = jnp.average(pred_energy, weights=1.0 / sigma2_energy)
    average_forces = (pred_forces / sigma2_forces[:, :, jnp.newaxis]).sum(
        axis=0
    ) / (1.0 / sigma2_forces).sum(axis=0)[:, jnp.newaxis]
    return average_energy, average_forces


def calc_aes(positions, types, cell, obs_energy, obs_forces):
    average_energy, average_forces = calc_averages(positions, types, cell)
    delta_energy = average_energy - obs_energy
    delta_forces = average_forces - obs_forces
    n_atoms = (types >= 0).sum()
    return jnp.abs(delta_energy) / n_atoms, jnp.abs(delta_forces).sum() / (
        3.0 * n_atoms
    )


calc_batched_aes = jax.jit(jax.vmap(calc_aes))

b_batch = 0
tae_energy = 0.0
tae_forces = 0.0
with tqdm.auto.tqdm(total=n_validate) as pbar:
    while b_batch < n_validate:
        e_batch = min(b_batch + N_EVAL_BATCH, n_validate)
        aes = calc_batched_aes(
            positions_validate[b_batch:e_batch],
            types_validate[b_batch:e_batch],
            cells_validate[b_batch:e_batch],
            energies_validate[b_batch:e_batch],
            forces_validate[b_batch:e_batch],
        )
        tae_energy += aes[0].sum()
        tae_forces += aes[1].sum()
        pbar.update(e_batch - pbar.n)
        b_batch = e_batch

print("Energy MAE:", tae_energy / n_validate, "eV / atom")
print("Force MAE:", tae_forces / n_validate, "eV / Å")
