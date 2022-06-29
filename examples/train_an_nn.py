#!/usr/bin/env python
"""
    This example script trains a NeuralIL model on prepared training and validation data.
"""

import datetime
import pathlib
import pickle
import sys

import ase
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
    PowerSpectrumGenerator, get_max_number_of_neighbors
)
from neuralil.model import Core, NeuralILModelInfo, NeuralIL
from neuralil.training import *
from neuralil.utilities import *

# Predefined parameters for this run.
R_CUT = 5.5
N_MAX = 5
EMBED_D = 2

N_EPOCHS = 5
N_BATCH = 8
N_EVAL_BATCH = 32

LOG_COSH_PARAMETER = 10e0  # In angstrom / eV

MAX_LEARNING_RATE = 3e-2
MIN_LEARNING_RATE = 3e-3
FIN_LEARNING_RATE = 3e-4

CORE_WIDTHS = [256, 128, 64, 32, 16, 16, 16]

# Generate a random seed and the associated
run_seed = draw_urandom_int32()
run_seed_str = f"{run_seed:0X}"
instance_code = f"OPTAX_{run_seed_str}"
rng = jax.random.PRNGKey(run_seed)

# Create a map from element names to integers. They must start at zero
# and be contiguous. They must also be consistent between the training
# and inference phases. Sorting the element names alphabetically is an
# easy and reasonable choice.
sorted_elements = sorted(["O", "Sr", "Ti"])
symbol_map = {s: i for i, s in enumerate(sorted_elements)}
n_types = len(symbol_map)

# Read the data from numpy files
positions = []
types = []
cells = []
energies = []
forces = []
print("- Reading from npy")

DATA_DIR = pathlib.Path(pathlib.Path(pathlib.Path.cwd() / "datasets"))
TRAIN_LABEL = str(DATA_DIR / "training_nnff1" / "nnff1_train")
cells_train = jnp.load(TRAIN_LABEL + '_cells.npy')
positions_train = jnp.load(TRAIN_LABEL + '_positions.npy')
types_train = jnp.load(TRAIN_LABEL + '_types.npy')
energies_train = jnp.load(TRAIN_LABEL + '_energies.npy') * 2.
forces_train = jnp.load(TRAIN_LABEL + '_forces.npy')
N_TRAIN = positions_train.shape[0]

VAL_LABEL = str(DATA_DIR / "validation_nnff1" / "nnff1_val")
cells_validate = jnp.load(VAL_LABEL + '_cells.npy')
positions_validate = jnp.load(VAL_LABEL + '_positions.npy')
types_validate = jnp.load(VAL_LABEL + '_types.npy')
energies_validate = jnp.load(VAL_LABEL + '_energies.npy') * 2.
forces_validate = jnp.load(VAL_LABEL + '_forces.npy')
N_VALIDATE = positions_validate.shape[0]

print("- Done")

# Shuffle the data.
n_configurations = positions_train.shape[0] + positions_validate.shape[0]
rng, shuffler_rng = jax.random.split(rng)
shuffle = create_array_shuffler(shuffler_rng)

print(f"- {n_configurations} configurations were loaded")
print(f"\t- {N_TRAIN} will be used for training")
print(f"\t- {N_VALIDATE} will be used for validation")

# Compute the maximum number of neighbors across all configurations.
positions = jnp.concatenate((positions_train, positions_validate), axis=0)
types = jnp.concatenate((types_train, types_validate), axis=0)
cells = jnp.concatenate((cells_train, cells_validate), axis=0)
max_neighbors = max(
    [
        get_max_number_of_neighbors(p, t, R_CUT, c) for p,
        t,
        c in zip(positions, types, cells)
    ]
)

print("- Maximum number of neighbors that must be considered: ", max_neighbors)

# Create the object that will generate descriptors for each configuration.
descriptor_generator = PowerSpectrumGenerator(
    N_MAX, R_CUT, n_types, max_neighbors
)

# Create the model. The number and kind of parameters is completely dependent
# on the kind of model used.
core_model = Core(CORE_WIDTHS)
dynamics_model = NeuralIL(
    n_types,
    EMBED_D,
    R_CUT,
    descriptor_generator.process_data,
    descriptor_generator.process_some_data,
    core_model
)

# Create the minimizer.
optimizer = create_one_cycle_minimizer(
    N_TRAIN // N_BATCH, MIN_LEARNING_RATE, MAX_LEARNING_RATE, FIN_LEARNING_RATE
)

# The model and the optimizer are stateless objects. Initialize the associated
# state for both. Note that the model initialization has a random component, but
# the optimizer initialization does not.
rng, init_rng = jax.random.split(rng)
model_params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=dynamics_model.calc_forces
)
optimizer_state = optimizer.init(model_params)

# Create the function that will compute the contribution to the loss from a
# single data point. In this case, our loss will not make use of the energies.
log_cosh = create_log_cosh(LOG_COSH_PARAMETER)


def calc_loss_contribution(pred_energy, pred_forces, obs_energy, obs_forces):
    "Return the log-cosh of the difference between predicted and actual forces."
    delta_forces = obs_forces - pred_forces
    return log_cosh(delta_forces).mean()


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
    epoch_rng
)

# Create a dictionary of validation statistics that we want calculated.
validation_statistics = {
    "force_MAE": fmae_validation_statistic,
    "energy_MAE": emae_validation_statistic,
    "force_RMSE": frmse_validation_statistic,
    "energy_RMSE": ermse_validation_statistic
}
validation_units = {
    "force_MAE": "eV / Å",
    "energy_MAE": "eV / atom",
    "force_RMSE": "eV / Å",
    "energy_RMSE": "eV / atom"
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
    N_EVAL_BATCH
)

PICKLE_FILE = f"model_params_{instance_code}.pickle"

min_mae = jnp.inf
min_rmse = jnp.inf
for i_epoch in range(N_EPOCHS):
    # Reset the training schedule.
    optimizer_state = reset_one_cycle_minimizer(optimizer_state)
    # Run a full epoch.
    optimizer_state, model_params = training_epoch(
        optimizer_state, model_params
    )
    # Evaluate the results.
    statistics = validation_step(model_params)
    mae = statistics["force_MAE"]
    rmse = statistics["force_RMSE"]
    # Print the relevant statistics.
    print(
        f"VALIDATION:"
        f"RMSE = {rmse} {validation_units['force_RMSE']}. "
        f"MAE = {mae} {validation_units['force_MAE']}."
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
        specific_info=None
    )

    # Save the state only if the validation MAE is minimal.
    if mae < min_mae:
        with open(PICKLE_FILE, "wb") as f:
            print("- Saving the most recent state")
            pickle.dump(model_info, f, protocol=5)
        min_mae = mae
    if rmse < min_rmse:
        min_rmse = rmse
