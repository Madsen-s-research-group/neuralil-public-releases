#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pathlib
import pickle
from typing import Tuple

import ase
import ase.atoms
import ase.io
import ase.units
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

import tqdm
import tqdm.auto
from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
)
from neuralil.model import (
    NeuralILwithMorse,
    ResNetCore,
)
from neuralil.committees.model import CommitteewithMorse

from neuralil.utilities import *
from tqdm import tqdm

TARGET_TEMPERATURE = 500.0  # K
DISPLACEMENT_AMPLITUDE = 1e-2  # angstrom
ADVERSARIAL_LEARNING_RATE = 1e-3
MAX_NEIGHBORS = 62 
SIGMA = 0.1

PARAMETER_FILE = (
    pathlib.Path(__file__).parent / "STO_params_committee_E5618B21.pkl"
)
print(f"Utilizing {PARAMETER_FILE}.")

# Generate a random seed and construct a generator
run_seed = draw_urandom_int32()
run_seed_str = f"{run_seed:0X}"
instance_code = f"ADV_{run_seed_str}"
rng = jax.random.PRNGKey(run_seed)

model_info = pickle.load(open(PARAMETER_FILE, "rb"))
symbol_map = {s: i for i, s in enumerate(model_info.sorted_elements)}

BASE_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data_sets"
    / "STO"
    / "POSCAR_Founder_4x1"
).resolve()

base_struct = ase.io.read(BASE_FILE)


def split_atoms(
    atoms: ase.atoms.Atoms,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Get positions, types and cells (and doubled arrays) from atoms object."""

    duplicated = atoms * (2, 1, 1)

    types = [symbol_map[s] for s in atoms.symbols]
    dupl_types = [symbol_map[s] for s in duplicated.symbols]
    return (
        jnp.array(atoms.positions),
        jnp.array(types),
        jnp.array(atoms.cell[...]),
        jnp.array(dupl_types),
        jnp.array(duplicated.cell[...]),
    )


(
    base_positions,
    base_types,
    base_cell,
    dupl_types,
    dupl_cell,
) = split_atoms(base_struct)

offset_zeros = jnp.zeros_like(base_positions)
offset_ones = offset_zeros + jnp.asarray([base_cell[0, 0], 0.0, 0.0])
offset = jnp.concatenate([offset_zeros, offset_ones])

descriptor_generator = PowerSpectrumGenerator(
    model_info.n_max,
    model_info.r_cut,
    len(model_info.sorted_elements),
    MAX_NEIGHBORS,
)
core_model = ResNetCore(model_info.core_widths)

individual_model = NeuralILwithMorse(
    len(model_info.sorted_elements),
    model_info.embed_d,
    model_info.r_cut,
    descriptor_generator.process_data,
    descriptor_generator.process_some_data,
    core_model,
)
N_ENSEMBLE = 10
dynamics_model = CommitteewithMorse(individual_model, N_ENSEMBLE)


@jax.jit
def ensemble_calculator(positions, types, cell):
    """Call energy and forces calculation on model."""
    raw = dynamics_model.apply(
        model_info.params,
        positions,
        types,
        cell,
        method=dynamics_model.calc_potential_energy_and_forces,
    )
    return raw


def calc_adversarial_log_loss(positions, types, cell, T):
    """Adversarial loss for a given T"""
    predicted_energy, predicted_forces = ensemble_calculator(
        positions, types, cell
    )
    exponential_arg = predicted_energy.mean() / (ase.units.kB * T)

    loss = -(
        jnp.log(jnp.sqrt(predicted_forces.var(axis=0).sum())) - exponential_arg
    )

    return loss


def calc_adv_loss_from_pos(positions):
    """Wrapper function to calculate adv loss for changed positions."""
    dupl_positions = jnp.tile(positions.reshape((-1, 3)), (2, 1)) + offset

    return calc_adversarial_log_loss(
        dupl_positions,
        dupl_types,
        dupl_cell,
        TARGET_TEMPERATURE,
    )


def run(rng) -> None:
    """Adversarial loss optimization for one initialization."""
    rng, normal_rng = jax.random.split(rng)
    noise = (
        jax.random.normal(key=normal_rng, shape=base_positions.shape) * SIGMA
    )
    res = minimize(
        fun=calc_adv_loss_from_pos,
        x0=(base_positions + noise).flatten(),
        method="Powell",
        options={
            "maxiter": 50,
            "ftol": 1,
            "xtol": 1e-1,
            "disp": True,
        },
    )

    return res


if __name__ == "__main__":
    adv_losses = []
    # Generate 100 total structures
    for i in tqdm(range(100)):
        rng, run_rng = jax.random.split(rng)
        res = run(run_rng)
        # all results are saved to separate pickle files
        with open(f"res_{instance_code}_{str(i)}.pkl", "wb") as f:
            pickle.dump(obj=res.x, file=f, protocol=5)
        adv_losses.append(res.fun)
        np.savetxt(f"adv_losses_{instance_code}", np.asarray(adv_losses))
        print(f"Iteration done {res.success}")
