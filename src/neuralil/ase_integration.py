#!/usr/bin/env python
# Copyright 2019-2023 The NeuralIL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module implements a calculator intended to be run on multicore nodes.

import copy

import jax
import numpy as onp

# Since this module uses jax.pmap, we tell ASE to run on the CPU. We try to do
# it as early as possible.
# FIXME: Handle this from the command line.
jax.config.update("jax_platform_name", "cpu")
import flax.serialization
import jax.numpy as jnp
import jax.random
from ase.calculators.calculator import Calculator, compare_atoms

from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
    get_max_number_of_neighbors,
)
from neuralil.model import NeuralILModelInfo


# TODO: Add more docstrings to this class.
class NeuralILASECalculator(Calculator):
    """Basic ASE Calculator based on the NeuralIL force field.

    Args:
        model: The Flax model that calculates total energies and forces.
        model_info: A NeuralILModelInfo with all relevant information and the
            model parameters.
        max_neighbors: The maximum number of neighbors that the descriptor
            generator will be able to handle.
        n_devices: The number of shards to be used when paralelizing with
            jax.pmap(). If not specified, it will be taken from JAX.

    """

    implemented_properties = ("energy", "forces")
    excluded_properties = ("initial_charges", "initial_magmoms")

    _ADAMS_NUMBER = 42

    def __init__(self, model, model_info, max_neighbors, n_devices=None):
        self.calculator_results = dict()
        if not isinstance(model_info, NeuralILModelInfo):
            raise ValueError(
                "model_info must be a instance of NeuralILModelInfo"
            )
        if (
            model.model_name != model_info.model_name
            or model.model_version != model_info.model_version
        ):
            raise ValueError("model_info does not match the provided model")
        self.pipeline = PowerSpectrumGenerator(
            model_info.n_max,
            model_info.r_cut,
            len(model_info.sorted_elements),
            max_neighbors,
        )
        self.model = model
        self.model_info = copy.deepcopy(model_info)
        self.max_neighbors = max_neighbors
        if n_devices is None:
            self.n_devices = len(jax.local_devices())
        else:
            self.n_devices = n_devices
        # TODO: that this way of mapping elements to atom types may not match
        # the convention used in some highly custom scripts. Generalize it.
        self.symbol_map = {
            s: i for i, s in enumerate(model_info.sorted_elements)
        }
        # We can initialize the model parameters using dummy positions and
        # types because the number of atoms is immaterial.
        template_params = self.model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((type(self)._ADAMS_NUMBER, 3)),
            jnp.zeros(type(self)._ADAMS_NUMBER, dtype=jnp.asarray(1).dtype),
            jnp.eye(3),
            method=self.model.calc_forces,
        )
        # NOTE: I do not fully understand why this cast is necesary, or rather
        # why from_state_dict does not take care of it.
        self.params = jax.tree_map(
            jnp.asarray,
            flax.serialization.from_state_dict(
                template_params, model_info.params
            ),
        )

        # This could be implemented as an ordinary method, but there have
        # been bugs in JAX triggering repeated recompilation of arguments
        # to pmap, so it is better to play it safe.
        self._energy_worker = jax.pmap(
            self._serial_energy_worker, in_axes=(0, 0, None, None, None)
        )
        super().__init__()

    def _serial_energy_worker(
        self, partial_p, partial_t, positions, types, cell
    ):
        return self.model.apply(
            self.params,
            partial_p,
            partial_t,
            positions,
            types,
            cell,
            method=self.model.calc_some_atomic_energies,
        )

    def _calc_atomic_energies(self, p, t, c):
        p = jnp.asarray(p)
        t = jnp.asarray(t)
        c = jnp.asarray(c)
        n_atoms = p.shape[0]
        n_atoms_original = n_atoms
        padding = self.n_devices - n_atoms % self.n_devices
        padded_p = jnp.pad(p, ((0, padding), (0, 0)))
        padded_t = jnp.pad(t, (0, padding), "constant", constant_values=-1)
        n_atoms = padded_p.shape[0]

        padded_p = padded_p.reshape(
            (self.n_devices, n_atoms // self.n_devices, 3)
        )
        padded_t = padded_t.reshape(
            (self.n_devices, n_atoms // self.n_devices)
        )
        nruter = self._energy_worker(padded_p, padded_t, p, t, c)
        nruter = nruter.reshape(
            tuple([nruter.shape[0] * nruter.shape[1]] + list(nruter.shape[2:]))
        )
        return nruter[:n_atoms_original]

    def _calc_potential_energy(self, p, t, c):
        return self._calc_atomic_energies(p, t, c).sum()

    def _calc_forces(self, p, t, c):
        return -jax.grad(self._calc_potential_energy, argnums=0)(p, t, c)

    def get_potential_energy(self, atoms, *args, **kwargs):
        if not self.calculation_required(atoms, "energy"):
            return self.calculator_results["energy"]
        self.atoms = atoms
        jtypes = jnp.asarray([self.symbol_map[s] for s in atoms.symbols])
        n_neighbors = get_max_number_of_neighbors(
            jnp.asarray(self.atoms.positions),
            jtypes,
            self.model_info.r_cut,
            jnp.asarray(self.atoms.cell[...]),
        )
        if n_neighbors > self.max_neighbors:
            raise ValueError(
                f"{n_neighbors} exceed max_neighbors={self.max_neighbors}"
            )
        result = float(
            self._calc_potential_energy(
                self.atoms.positions, jtypes, self.atoms.cell[...]
            )
        )
        self.calculator_results["energy"] = result
        return result

    def get_forces(self, atoms, *args, **kwargs):
        if not self.calculation_required(atoms, "forces"):
            return self.calculator_results["forces"]
        self.atoms = atoms
        jtypes = jnp.asarray([self.symbol_map[s] for s in atoms.symbols])
        n_neighbors = get_max_number_of_neighbors(
            jnp.asarray(self.atoms.positions),
            jtypes,
            self.model_info.r_cut,
            jnp.asarray(self.atoms.cell[...]),
        )
        if n_neighbors > self.max_neighbors:
            raise ValueError(
                f"{n_neighbors} exceed max_neighbors={self.max_neighbors}"
            )
        result = onp.array(
            self._calc_forces(
                self.atoms.positions, jtypes, self.atoms.cell[...]
            )
        )
        self.calculator_results["forces"] = result
        return result

    def calculation_required(self, atoms, properties):
        for p in properties:
            if p not in self.implemented_properties:
                return True
        return (
            len(
                compare_atoms(
                    self.atoms,
                    atoms,
                    excluded_properties=self.excluded_properties,
                )
            )
            > 0
        )
