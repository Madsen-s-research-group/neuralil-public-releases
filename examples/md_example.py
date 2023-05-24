#!/usr/bin/env python

import itertools
import json
import pathlib
import pickle
import re
from collections import defaultdict

import ase
import ase.io
import jax
import jax.random
import numpy as onp
import pymatgen.io.cif
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS

from neuralil.ase_integration import NeuralILASECalculator
from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.model import NeuralILwithMorse, ResNetCore

MIN_DIST = 1.0
TARGET_T = 1200
PICKLE_FILE = (
    pathlib.Path(__file__).parent / "model_params_neuralil_VELO_40584908.pkl"
)
MAX_NEIGHBORS = 35

# Load the parameters of the NeuralIL model we are going to use.
model_info = pickle.load(open(PICKLE_FILE, "rb"))
# Create the object used to compute descriptors...
descriptor_generator = PowerSpectrumGenerator(
    model_info.n_max,
    model_info.r_cut,
    len(model_info.sorted_elements),
    MAX_NEIGHBORS,
)
# ..., the immutable model object...
dynamics_model = NeuralILwithMorse(
    len(model_info.sorted_elements),
    model_info.embed_d,
    model_info.r_cut,
    descriptor_generator,
    descriptor_generator.process_some_data,
    ResNetCore(model_info.core_widths),
)

# ...and the ASE calculator.
calculator = NeuralILASECalculator(dynamics_model, model_info, MAX_NEIGHBORS)

# Load a starting configuration from a file.
with open(
    (pathlib.Path(__file__).parent / "configurations.json").resolve(), "r"
) as json_f:
    json_data = json.loads(next(json_f))
    cell = onp.diag(onp.array(json_data["Cell-Size"]))
    positions = json_data["Positions"]
symbols_cation = ["N", "H", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
symbols_anion = ["N", "O", "O", "O"]
n_pair = len(positions) // (len(symbols_anion) + len(symbols_cation))
symbols = n_pair * symbols_cation + n_pair * symbols_anion
atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell)

# Attach the calculator to the atoms object.
atoms.calc = calculator

# Run a very rough energy minimization to avoid high-energy local
# configurations.
optimizer = LBFGS(atoms)
optimizer.run(fmax=0.1)

# Initialize the velocities at random.
MaxwellBoltzmannDistribution(atoms, temperature_K=float(TARGET_T))

# Create a Langevin integrator to take the system to the desired temperature.
integrator = Langevin(
    atoms, 1.0 * units.fs, temperature_K=float(TARGET_T), friction=2e-2
)


# Set up a callback to print information about the evolution of the process.
def print_md_info(current_atoms=atoms, current_integrator=integrator):
    n_atoms = len(current_atoms)
    e_pot_per_atom = current_atoms.get_potential_energy() / n_atoms
    e_kin_per_atom = current_atoms.get_kinetic_energy() / n_atoms
    kinetic_temperature = e_kin_per_atom / (1.5 * units.kB)
    print(
        f"Step #{current_integrator.nsteps + 1}: "
        f"Epot = {e_pot_per_atom} eV / atom, T = {kinetic_temperature} K, "
        f"E = {e_pot_per_atom + e_kin_per_atom} eV / atom"
    )


# Attach the callback to the integrator, tell ASE to save the trajectory,
# and run a simulation for 10000 steps.
integrator.attach(print_md_info, interval=5)
trajectory = Trajectory(f"test_garnet_{TARGET_T}K.traj", "w", atoms)
integrator.attach(trajectory.write, interval=10)

integrator.run(10000)
