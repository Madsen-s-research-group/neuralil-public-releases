#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import datetime
import json
import pathlib
import pickle
import sys

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
from neuralil.model import NeuralIL, NeuralILModelInfo, ResNetCore
from neuralil.training import *
from neuralil.utilities import *

# Parameters for this run.
USE_RANDOM_RUN_SEED = True
R_CUT = 3.5
N_MAX = 4
EMBED_D = 2
WEIGHT_ENERGY = 0.5

N_EPOCHS = 101
N_BATCH = 8
N_EVAL_BATCH = 32

LOG_COSH_PARAMETER_FORCES = 0.200  # In eV / angstrom
LOG_COSH_PARAMETER_ENERGY = 0.020  # In eV / atom

CORE_WIDTHS = [64, 32, 16]

ADAM_MINIMIZER = False

if ADAM_MINIMIZER:
    MAX_LEARNING_RATE = 1e-2
    MIN_LEARNING_RATE = 1e-3
    FIN_LEARNING_RATE = 1e-5

# Create a random seed or use the same one as in the manuscript.
if USE_RANDOM_RUN_SEED:
    run_seed = draw_urandom_int32()
else:
    run_seed = 31337
run_seed_str = f"{run_seed:0X}"
instance_code = f"{run_seed_str}"
rng = jax.random.PRNGKey(run_seed)

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
        pathlib.Path(__file__).resolve().parent.parent
        / "data_sets"
        / "EAN"
        / "training_original.json"
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
        pathlib.Path(__file__).resolve().parent.parent
        / "data_sets"
        / "EAN"
        / "validation_original.json"
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

# Create the model. The number and kind of parameters is completely dependent
# on the kind of model used.
core_model = ResNetCore(CORE_WIDTHS)

# dynamics_model = NeuralILwithMorse(
#     n_types,
#     EMBED_D,
#     R_CUT,
#     descriptor_generator.process_data,
#     descriptor_generator.process_some_data,
#     core_model,
# )

dynamics_model = NeuralIL(
    n_types,
    EMBED_D,
    R_CUT,
    descriptor_generator.process_data,
    descriptor_generator.process_some_data,
    core_model,
)


# Create the minimizer.
if ADAM_MINIMIZER:
    optimizer = create_one_cycle_minimizer(
        n_train // N_BATCH,
        MIN_LEARNING_RATE,
        MAX_LEARNING_RATE,
        FIN_LEARNING_RATE,
    )
else:
    optimizer = create_velo_minimizer(n_train // N_BATCH, N_EPOCHS)

# The model and the optimizer are stateless objects. Initialize the associated
# state for both. Note that the model initialization has a random component, but
# the optimizer initialization does not.
rng, init_rng = jax.random.split(rng)
model_params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=dynamics_model.calc_forces,
)
optimizer_state = optimizer.init(model_params)

# Create the function that will compute the contribution to the loss from a
# single data point.
log_cosh_energy = create_log_cosh(1.0 / LOG_COSH_PARAMETER_ENERGY)
log_cosh_forces = create_log_cosh(1.0 / LOG_COSH_PARAMETER_FORCES)


def calc_loss_contribution(
    pred_energy, pred_forces, obs_energy, obs_forces, types
):
    "Return the weighted log-cosh of the prediction deltas."
    delta_energy = (obs_energy - pred_energy) / (types >= 0).sum()
    energy_contribution = log_cosh_energy(delta_energy)
    delta_forces = jnp.sqrt(((obs_forces - pred_forces) ** 2).sum(axis=-1))
    forces_contribution = (
        log_cosh_forces(delta_forces).sum() / (types >= 0).sum()
    )
    return (
        WEIGHT_ENERGY * energy_contribution
        + (1.0 - WEIGHT_ENERGY) * forces_contribution
    )


# Create a driver for each training step.
training_step = create_training_step(
    dynamics_model, optimizer, calc_loss_contribution
)

# Create a driver for each full epoch.
rng, epoch_rng = jax.random.split(rng)
training_epoch = create_training_epoch(
    positions_train,
    types_train,
    cells_train,
    energies_train,
    forces_train,
    N_BATCH,
    training_step,
    epoch_rng,
)

# Create a dictionary of validation statistics that we want calculated.
validation_statistics = {
    "forces_MAE": fmae_validation_statistic,
    "energy_MAE": emae_validation_statistic,
    "energy_logcosh": create_elogcosh_validation_statistic(
        1.0 / LOG_COSH_PARAMETER_ENERGY
    ),
    "forces_RMSE": frmse_validation_statistic,
    "energy_RMSE": ermse_validation_statistic,
    "forces_logcosh": create_flogcosh_validation_statistic(
        1.0 / LOG_COSH_PARAMETER_FORCES
    ),
}
validation_units = {
    "forces_MAE": "eV / Å",
    "energy_MAE": "eV / atom",
    "forces_RMSE": "eV / Å",
    "energy_RMSE": "eV / atom",
    "forces_logcosh": "eV / Å",
    "energy_logcosh": "eV / atom",
}

# Create the driver for the validation step.
validation_step = create_validation_step(
    dynamics_model,
    validation_statistics,
    positions_validate,
    types_validate,
    cells_validate,
    energies_validate,
    forces_validate,
    N_EVAL_BATCH,
)

PICKLE_FILE = f"EAN_params_{instance_code}.pkl"

min_mae = jnp.inf
min_rmse = jnp.inf
for i_epoch in range(N_EPOCHS):
    if ADAM_MINIMIZER:
        optimizer_state = reset_one_cycle_minimizer(optimizer_state)
    # Run a full epoch.
    (optimizer_state, model_params) = training_epoch(
        optimizer_state, model_params
    )
    # Evaluate the results.
    statistics = validation_step(model_params)
    mae = statistics["forces_MAE"]
    rmse = statistics["forces_RMSE"]
    e_mae = statistics["energy_MAE"]
    e_rmse = statistics["energy_RMSE"]
    # Print the relevant statistics.
    print(
        f"VALIDATION: "
        f"F_RMSE = {rmse} {validation_units['forces_RMSE']}. "
        f"F_MAE = {mae} {validation_units['forces_MAE']}. "
        f"E_RMSE = {e_rmse} {validation_units['energy_RMSE']}. "
        f"E_MAE = {e_mae} {validation_units['energy_MAE']}."
    )

    # Create an object to hold all the information about the model.
    model_info = NeuralILModelInfo(
        model_name=dynamics_model.model_name,
        model_version=dynamics_model.model_version,
        timestamp=datetime.datetime.now(),
        r_cut=R_CUT,
        n_max=N_MAX,
        sorted_elements=sorted_elements,
        embed_d=EMBED_D,
        core_widths=CORE_WIDTHS,
        constructor_kwargs=dict(),
        random_seed=run_seed,
        params=flax.serialization.to_state_dict(model_params),
        specific_info=None,
    )

    # Save the state only if the validation RMSE is minimal.
    if mae < min_mae:
        min_mae = mae
    if rmse < min_rmse:
        with open(PICKLE_FILE, "wb") as f:
            print("- Saving the most recent state")
            pickle.dump(model_info, f, protocol=5)
        min_rmse = rmse
