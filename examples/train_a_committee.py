#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import copy
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
from neuralil.model import (
    NeuralILModelInfo,
    NeuralILwithMorse,
    ResNetCore,
    update_energy_offset,
)
from neuralil.committees.model import CommitteewithMorse
from neuralil.committees.training import *
from neuralil.training import (
    create_elogcosh_validation_statistic,
    create_flogcosh_validation_statistic,
    create_log_cosh,
    create_velo_minimizer,
)
from neuralil.utilities import *

# Predefined parameters for this run.
TRAINING_FRACTION = 0.9
R_CUT = 3.5
N_MAX = 4
EMBED_D = 2
WEIGHT_ENERGY = 0.5

N_EPOCHS = 21
N_BATCH = 8
N_EVAL_BATCH = 32

LOG_COSH_PARAMETER_FORCES = 0.200  # In eV / angstrom
LOG_COSH_PARAMETER_ENERGY = 0.020  # In eV / atom

CORE_WIDTHS = [64, 32, 16]

N_ENSEMBLE = 10

# Generate a random seed and the associated code
run_seed = draw_urandom_int32()
run_seed_str = f"{run_seed:0X}"
instance_code = f"VELO_{run_seed_str}"
rng = jax.random.PRNGKey(run_seed)

# Create a map from element names to integers. They must start at zero
# and be contiguous. They must also be consistent between the training
# and inference phases. Sorting the element names alphabetically is an
# easy and reasonable choice. Naturally, the set of elements can be
# extracted from the configurations themselves if this step is deferred.
sorted_elements = sorted(["C", "H", "O", "N"])
symbol_map = {s: i for i, s in enumerate(sorted_elements)}
n_types = len(symbol_map)

# Read the data from a JSON file. In this case it is a custom design,
# but an ASE database is another common choice.
cells = []
positions = []
energies = []
forces = []
print("- Reading the JSON file")
with open(
    (pathlib.Path(__file__).parent / "configurations.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell-Size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")
n_configurations = len(positions)

# The atom types are the same in every configuration and not stored in the file.
type_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
type_anion = ["N", "O", "O", "O"]
n_pair = len(positions[0]) // (len(type_anion) + len(type_cation))
types = n_pair * type_cation + n_pair * type_anion
unique_types = sorted(set(types))
types = [[symbol_map[i] for i in types]] * n_configurations

# Shuffle the data.
rng, shuffler_rng = jax.random.split(rng)
shuffle = create_array_shuffler(shuffler_rng)
cells = shuffle(cells)
positions = shuffle(positions)
energies = shuffle(energies)
types = shuffle(types)
forces = shuffle(forces)

# Extract training and validation subsets.
n_train = round(n_configurations * TRAINING_FRACTION)
n_validate = n_configurations - n_train

print(f"- {n_configurations} configurations are available")
print(f"\t- {n_train} will be used for training")
print(f"\t- {n_validate} will be used for validation")


def split_array(in_array):
    "Split an array in training and validation sections."
    return jnp.split(in_array, (n_train, n_train + n_validate))[:2]


cells_train, cells_validate = split_array(cells)
positions_train, positions_validate = split_array(positions)
types_train, types_validate = split_array(types)
energies_train, energies_validate = split_array(energies)
forces_train, forces_validate = split_array(forces)

# Compute the maximum number of neighbors across all configurations.
max_neighbors = max(
    [
        get_max_number_of_neighbors(p, t, R_CUT, c)
        for p, t, c in zip(positions, types, cells)
    ]
)

print("- Maximum number of neighbors that must be considered: ", max_neighbors)

# Create the object that will generate descriptors for each configuration.
descriptor_generator = PowerSpectrumGenerator(
    N_MAX, R_CUT, n_types, max_neighbors
)

# Create the model. The number and kind of parameters is completely dependent
# on the kind of model used.
core_model = ResNetCore(CORE_WIDTHS)

individual_model = NeuralILwithMorse(
    n_types,
    EMBED_D,
    R_CUT,
    descriptor_generator,
    descriptor_generator.process_some_data,
    core_model,
)
dynamics_model = CommitteewithMorse(individual_model, N_ENSEMBLE)

# Create the minimizer.
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


# Initialize the bias of the linear layer to something vaguely reasonable to
# avoid wasting a few dozen epochs just centering the predicted energies.
@jax.jit
def individual_energy_calculator(params, positions, types, cell):
    return individual_model.apply(
        params,
        positions,
        types,
        cell,
        method=individual_model.calc_potential_energy,
    )


print("- Adjusting the offsets")
individual_params = unpack_params(model_params)
updated_params = []
for p in individual_params:
    energies = []
    for i in tqdm.auto.trange(positions_train.shape[0]):
        energies.append(
            individual_energy_calculator(
                p, positions_train[i], types_train[i], cells_train[i]
            )
        )
    energies = jnp.asarray(energies)
    offset = (
        -jnp.average(energies_train - energies) / positions_train[0].shape[0]
    )
    updated_params.append(update_energy_offset(p, offset))
model_params = pack_params(updated_params)
print("- Done")

optimizer_state = optimizer.init(model_params)

log_cosh_energy = create_log_cosh(1.0 / LOG_COSH_PARAMETER_ENERGY)
log_cosh_forces = create_log_cosh(1.0 / LOG_COSH_PARAMETER_FORCES)


def calc_loss_contribution(
    pred_energy, obs_energy, pred_forces, obs_forces, types
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

PICKLE_FILE = f"model_params_plain_ensemble_{instance_code}.pkl"


def params1_dominate_params2(statistics1, statistics2):
    if (statistics1 > statistics2).any():
        return False
    return (statistics1 < statistics2).any()


def are_fronts_equal(front1, front2):
    if len(front1) != len(front2):
        return False
    return all([p1[1] == p2[1] for (p1, p2) in zip(front1, front2)])


def reduce_pareto_front(pareto_front):
    distilled_pareto_front = []
    for p1 in pareto_front:
        for p2 in pareto_front:
            if params1_dominate_params2(
                jnp.asarray(p2[-2:]), jnp.asarray(p1[-2:])
            ):
                break
        else:
            distilled_pareto_front.append(p1)
    return distilled_pareto_front


initial_params = unpack_params(model_params)

pareto_fronts = [
    [(initial_params[i_model], -1, jnp.inf, jnp.inf)]
    for i_model in range(N_ENSEMBLE)
]

# In addition to saving based on the RMSE, we keep some info about the Pareto
# front.
for i_epoch in range(N_EPOCHS):
    # Run a full epoch.
    (optimizer_state, model_params) = training_epoch(
        optimizer_state, model_params
    )
    # Evaluate the results.
    statistics = validation_step(model_params)
    mae = statistics["forces_MAE"]
    rmse = statistics["forces_RMSE"]
    logcosh = statistics["forces_logcosh"]
    e_mae = statistics["energy_MAE"]
    e_rmse = statistics["energy_RMSE"]
    e_logcosh = statistics["energy_logcosh"]
    # Print the relevant statistics.
    candidates = []
    new_params = unpack_params(model_params)
    print("VALIDATION:")
    for i_model in range(N_ENSEMBLE):
        print(
            f"Model #{i_model+1}:"
            f"F_RMSE = {rmse[i_model]} {validation_units['forces_RMSE']}. "
            f"F_MAE = {mae[i_model]} {validation_units['forces_MAE']}. "
            f"F_logcosh = {logcosh[i_model]} {validation_units['forces_logcosh']}. "
            f"E_RMSE = {e_rmse[i_model]} {validation_units['energy_RMSE']}. "
            f"E_MAE = {e_mae[i_model]} {validation_units['energy_MAE']}. "
            f"E_logcosh = {e_logcosh[i_model]} {validation_units['energy_logcosh']}."
        )
        candidates.append(
            (
                new_params[i_model],
                i_epoch,
                float(logcosh[i_model]),
                float(e_logcosh[i_model]),
            )
        )

    old_pareto_fronts = copy.deepcopy(pareto_fronts)

    updated = []
    for i_model in range(N_ENSEMBLE):
        pareto_fronts[i_model].append(candidates[i_model])
        pareto_fronts[i_model] = reduce_pareto_front(pareto_fronts[i_model])
        if not are_fronts_equal(
            pareto_fronts[i_model], old_pareto_fronts[i_model]
        ):
            updated.append(i_model)

    if updated:
        n_elements = sum([len(p) for p in pareto_fronts])
        print("Pareto fronts updated")
        print("There are currently", n_elements, "elements")
        for p in pareto_fronts:
            print([t[1:] for t in p])
        print(
            "Only the element with the lowest force"
            " loss in each front will be saved."
        )
        to_be_saved = [min(p, key=lambda x: x[2]) for p in pareto_fronts]

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
            params=flax.serialization.to_state_dict(
                pack_params([m[0] for m in to_be_saved])
            ),
            specific_info=None,
        )

        with open(PICKLE_FILE, "wb") as f:
            print("- Saving the updated Pareto front")
            pickle.dump(model_info, f, protocol=5)
