#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pathlib
import pickle
from typing import Tuple, Union

import flax
import h5py
import jax
import jax.numpy as jnp
import jax_md
import MDAnalysis
import numpy as onp
import scipy as sp
import scipy.constants
import tqdm

from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.committees.model import Committee, NeuralIL
from neuralil.model import ResNetCore

# Parameters of the NNFF
NTYPES = 4
CORE = ResNetCore
MAX_NEIGH = 25
NN_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "auxiliary_files"
    / "EAN_params_committee_1FEB3E08.pkl"
).resolve()


INITIAL_CONF = (
    pathlib.Path(__file__).resolve().parent.parent
    / "auxiliary_files"
    / "EAN_md_initial.gro"
).resolve()
OUT_TRAJ = "test_committee_nvt.trr"
OUT_LOG = "test_committee_nvt.log"
OUT_STD = "std_committee_values.h5"

TIMESTEP = 0.001  # Time step in ps
NSTEPS = 100000
WRITE_EVERY = (
    10  # Number of steps between logging of quantities and coordinates
)

TEMPERATURE = 298  # Temperature in K
TAU_T = 0.1  # Thermostat relaxation time (ps)
RNG_SEED = 71923  # Seed for the velocity generation

kb = sp.constants.k / sp.constants.e  # Boltzman constant in eV/K
kt = TEMPERATURE * kb  # Temperature conversion to eV


def load_model(path: str):
    """Load a Deep ensemble Neuralil Model"""

    with open(path, "rb") as datafile:
        data = pickle.load(datafile)

    pipeline = PowerSpectrumGenerator(
        data.n_max, data.r_cut, NTYPES, MAX_NEIGH
    )
    inner = NeuralIL(
        NTYPES,
        data.embed_d,
        data.r_cut,
        pipeline.process_data,
        pipeline.process_some_data,
        CORE(data.core_widths),
    )
    model = Committee(inner, 1)

    # We must run the model with a set of mock data so flax create the
    # parameters of the NNFF, they will be latter overwritten.
    pos_mock = jnp.zeros([2, 3])
    types_mock = jnp.zeros(2, dtype=jnp.int32)
    cell_mock = jnp.eye(3)

    # This RNG does not need to be random as it only randomize the palceholder
    # parameters than will be later overwritten.
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    # Create the placeholder parameters.
    params = model.init(
        init_rng, pos_mock, types_mock, cell_mock, method=model.calc_forces
    )

    # Substitute the placeholder parameters by the oners from the pickle file.
    params = jax.tree_map(
        jnp.asarray, flax.serialization.from_state_dict(params, data.params)
    )

    return model, params


def all_calculator(model, params):
    @jax.jit
    def calculate(positions, types, box):
        energy = model.apply(
            params, positions, types, box, method=model.calc_potential_energy
        )
        forces = model.apply(
            params, positions, types, box, method=model.calc_forces
        )
        return energy, forces

    return calculate


def force_calculator(model, params):
    @jax.jit
    def calculate_forces(positions, types, box):
        forces = model.apply(
            params, positions, types, box, method=model.calc_forces
        )
        mean_force = jnp.mean(forces, axis=0)
        return mean_force

    return calculate_forces


_e = sp.constants.e  # Electron Charge in C
_NA = sp.constants.N_A  # Avogadro's number
_CONVERSION = 10 / (
    _e * _NA
)  # Mass conversion from g/mol to units consistent with the other magnitudes


def load_system_gro(
    filename: Union[str, pathlib.Path]
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """Load the information for the simulation from a .gro file"""
    uni = MDAnalysis.Universe(filename)

    positions = jnp.array(uni.atoms.positions)

    types_sorted = sorted(list(set(uni.atoms.types)))
    type_translate = {typ: index for index, typ in enumerate(types_sorted)}

    types = jnp.array([type_translate[i] for i in uni.atoms.types])

    cell = jnp.array(uni.dimensions[:3])

    masses = jnp.array(uni.atoms.masses) * _CONVERSION

    return positions, types, masses, jnp.diag(cell)


# Load the initial positions
positions, types, masses, cell = load_system_gro(INITIAL_CONF)

print("Loading NNFF")
model, params = load_model(NN_FILE)
# Create function for calculating the force compatible with jax_md
potential = force_calculator(model, params)
calc_all = all_calculator(model, params)

# Create the position update function, taking into account the PBC
disp_fn, shift_fn = jax_md.space.periodic_general(
    cell, fractional_coordinates=False
)

# Couple a Nose Hoover thermostate to the simulation (By deafult it uses 5 chains)
init_fn, step_fn = jax_md.simulate.nvt_nose_hoover(
    potential, shift_fn, TIMESTEP, kt, tau=TAU_T
)

print("Generating initial state")
rng = jax.random.PRNGKey(RNG_SEED)
# Create the initial state with random velocities from a Maxwell distribution
state = init_fn(rng, positions, mass=masses, types=types, box=cell)


def simulate(step_fn, init_state, nsteps, types, cell):
    """Create a function that fuses several simulation steps into 1"""

    @jax.jit
    def do_one(i, state):
        return step_fn(state, types=types, box=cell)

    return jax.lax.fori_loop(0, nsteps, do_one, init_state)


# Set the values for the multiple-step update and JIT everything
simulation_fn = jax.jit(
    lambda init_state: simulate(step_fn, init_state, WRITE_EVERY, types, cell)
)

# Setup the MDAnalysis Universe for writting the trayectory
md_atoms = MDAnalysis.Universe(INITIAL_CONF, dt=TIMESTEP * WRITE_EVERY).atoms
n_steps = NSTEPS // WRITE_EVERY

# Setupt the variables to store the STD
energy_std_hos = onp.zeros(n_steps)
force_std_hos = onp.zeros((n_steps, len(md_atoms)))


print("Simulation Starting")
print(
    "Mind that the first iteration will be very slow due to the JIT compilation."
)

with MDAnalysis.Writer(
    OUT_TRAJ, len(md_atoms), dt=TIMESTEP * WRITE_EVERY
) as writer:
    with open(OUT_LOG, "w") as log_file:
        log_file.write("# Time (ps)  |  Temperature (K)\n")
        temp = (
            jax_md.quantity.temperature(
                velocity=state.velocity, mass=state.mass
            )
            / kb
        )
        log_file.write(f"{0:9d}    {temp: 8.3f}\n")
        log_file.flush()

        # SIMULATION LOOP
        for i in tqdm.tqdm(range(n_steps)):
            # Perform WRITE_EVERY steps
            state = simulation_fn(state)

            # Update the simulation log
            time = (i + 1) * WRITE_EVERY * TIMESTEP
            temp = (
                jax_md.quantity.temperature(
                    velocity=state.velocity, mass=state.mass
                )
                / kb
            )
            log_file.write(f"{time:9f}    {temp: 8.3f}\n")
            log_file.flush()

            # Add coordinates to the trajectory
            md_atoms.positions = state.position
            md_atoms.velocities = state.velocity
            writer.write(md_atoms)

            # Write the STD information
            ener, forces = calc_all(state.position, types, cell)
            energy_std_hos[i] = jnp.std(ener) / jnp.sqrt(ener.shape[0] - 1)
            forces_modulus = jnp.linalg.norm(forces, axis=-1)
            force_std_hos[i, :] = jnp.std(forces_modulus, axis=0) / jnp.sqrt(
                forces.shape[0] - 1
            )

print("Simulation finished writting STD values")

with h5py.File(OUT_STD, "w") as h5_file:
    h5_file.create_dataset("energy_std_hos", data=energy_std_hos)
    h5_file.create_dataset("force_std_hos", data=force_std_hos)
