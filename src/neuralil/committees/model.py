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

import datetime
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

import flax.linen
import jax
import jax.numpy as jnp

from neuralil.model import NeuralIL, NeuralILwithMorse


class Committee(flax.linen.Module):
    """Ensemble of N NeuralIL models.

    The first argument is a NeuralIL object. The second one is the number
    of models in the ensemble. See the docstrings in the "model" module for
    details about the interface.
    """

    neuralil: NeuralIL
    n_models: int
    model_name: ClassVar[str] = "Committee"
    model_version: ClassVar[str] = "0.2"

    def setup(self):
        self.calc_atomic_energies_from_descriptors = flax.linen.vmap(
            NeuralIL.calc_atomic_energies_from_descriptors,
            in_axes=(None, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_models,
        )
        # Note the switch from gradient to Jacobian to account for the
        # ensemble axis.
        self._calc_jacobian = jax.jacrev(self.calc_potential_energy, argnums=0)

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.neuralil.descriptor_generator(
            positions, types, cell
        )
        # Note the expansion of "types" to account for the ensemble axis.
        return (types >= 0)[
            jnp.newaxis, :
        ] * self.calc_atomic_energies_from_descriptors(
            self.neuralil, descriptors, types
        )

    def calc_some_atomic_energies(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        some_descriptors = self.neuralil.partial_descriptor_generator(
            all_positions, all_types, some_positions, cell
        )
        return (some_types >= 0)[
            jnp.newaxis, :
        ] * self.calc_atomic_energies_from_descriptors(
            self.neuralil, some_descriptors, some_types
        )

    def calc_some_atomic_energies_and_average(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        return self.calc_some_atomic_energies(
            some_positions, some_types, all_positions, all_types, cell
        ).mean(axis=0)

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        # Note the change of axis to account for the prepending of the
        # ensemble axis.
        return jnp.squeeze(contributions.sum(axis=1))

    def calc_forces(self, positions, types, cell):
        # Note the switch from gradient to Jacobian to account for the
        # ensemble axis.
        return -self._calc_jacobian(positions, types, cell)

    def calc_potential_energy_and_forces(self, positions, types, cell):
        return (
            self.calc_potential_energy(positions, types, cell),
            self.calc_forces(positions, types, cell),
        )


class CommitteewithMorse(flax.linen.Module):
    """Ensemble of N NeuralILwithMorse models.

    The first argument is a NeuralIL object. The second one is the number
    of models in the ensemble. See the docstrings in the "model" module for
    details about the interface.
    """

    neuralil: NeuralILwithMorse
    n_models: int
    model_name: ClassVar[str] = "Committee"
    model_version: ClassVar[str] = "0.1"

    def setup(self):
        self.calc_atomic_energies_from_descriptors = flax.linen.vmap(
            NeuralILwithMorse.calc_atomic_energies_from_descriptors,
            in_axes=(None, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_models,
        )
        # Note the switch from gradient to Jacobian to account for the
        # ensemble axis.
        self.calc_atomic_morse_energies = flax.linen.vmap(
            NeuralILwithMorse.calc_morse_energies,
            in_axes=(None, None, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_models,
        )
        self.calc_some_morse_energies = flax.linen.vmap(
            NeuralILwithMorse.calc_some_morse_energies,
            in_axes=(None, None, None, None, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_models,
        )

        self._calc_jacobian = jax.jacrev(self.calc_potential_energy, argnums=0)

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.neuralil.descriptor_generator(
            positions, types, cell
        )
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            self.neuralil, descriptors, types
        )
        morse_contributions = self.calc_atomic_morse_energies(
            self.neuralil, positions, types, cell
        )
        # Note the expansion of "types" to account for the ensemble axis.
        return (types >= 0)[jnp.newaxis, :] * (
            nn_contributions + morse_contributions
        )

    def calc_some_atomic_energies(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        some_descriptors = self.neuralil.partial_descriptor_generator(
            all_positions, all_types, some_positions, cell
        )
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            self.neuralil, some_descriptors, some_types
        )
        morse_contributions = self.calc_some_morse_energies(
            self.neuralil,
            some_positions,
            some_types,
            all_positions,
            all_types,
            cell,
        )
        return (some_types >= 0)[jnp.newaxis, :] * (
            nn_contributions + morse_contributions
        )

    def calc_some_atomic_energies_and_average(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        return self.calc_some_atomic_energies(
            some_positions, some_types, all_positions, all_types, cell
        ).mean(axis=0)

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        # Note the change of axis to account for the prepending of the
        # ensemble axis.
        return jnp.squeeze(contributions.sum(axis=1))

    def calc_forces(self, positions, types, cell):
        # Note the switch from gradient to Jacobian to account for the
        # ensemble axis.
        return -self._calc_jacobian(positions, types, cell)

    def calc_potential_energy_and_forces(self, positions, types, cell):
        return (
            self.calc_potential_energy(positions, types, cell),
            self.calc_forces(positions, types, cell),
        )
