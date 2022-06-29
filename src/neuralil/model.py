# Copyright 2019-2021 The NeuralIL contributors
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
import jax.nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


@jax.custom_jvp
def _sqrt(x):
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    x, = primals
    xdot, = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0., 0., 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)


class Core(flax.linen.Module):
    """Multilayer perceptron with LayerNorm lying at the core of the model.

    This model takes the descriptors of each atom (Bessel + embedding,
    concatenated or otherwise combined) as inputs and calculates that atom's
    contribution to the potential energy. LayerNorm is applied at each layer
    except the first and the last ones.

    Args:
        layer_widths: The sequence of layer widths, excluding the output
            layer, which always has a width equal to one.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
    """
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(self.layer_widths[0])(descriptors)
        )
        for w in self.layer_widths[1:]:
            result = self.activation_function(
                flax.linen.LayerNorm()(
                    flax.linen.Dense(w, use_bias=False)(result)
                )
            )
        return self.activation_function(flax.linen.Dense(1)(result))


class NeuralIL(flax.linen.Module):
    """Wrapper model around the core layers to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atom types in the system.
        embed_d: The dimension of the embedding vector to be mixed with the
            descriptors.
        r_cut: The cutoff radius for the short-range part of the potential.
        partial_descriptor_generator: A function like descriptor_generator,
            but used to compute the descriptors for some atoms only.
        descriptor_generator: The function mapping the atomic coordinates,
            types and cell size to descriptors.
        core_model: The model that takes all the descriptors and returns an
            atomic contribution to the energy.
        mixer: The function that takes the Bessel descriptors and the
            embedding vectors and creates the input descriptors for the core
            model. The default choice just concatenates them.
    """
    n_types: int
    embed_d: int
    r_cut: float
    descriptor_generator: Callable
    partial_descriptor_generator: Callable
    core_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1
    )
    model_name: ClassVar[str] = "NeuralIL"
    model_version: ClassVar[str] = "0.3"

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = flax.linen.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._calc_gradient = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0)
        )
        self._calc_value_and_gradient = jax.checkpoint(
            jax.value_and_grad(self.calc_potential_energy, argnums=0)
        )

    def calc_combined_inputs(self, positions, types, cell):
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        return combined_inputs

    def calc_atomic_energies_from_descriptors(self, descriptors, types):
        """Compute the atomic contributions to the potential energy.

        Args:
            descriptors: The n_atoms vectors of descriptors, as a single
                tensor.
            types: The atom types, codified as integers from 0 to n_types - 1.

        Returns:
            The n_atoms contributions to the energy.
        """
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return (types >= 0) * jnp.squeeze(results)

    def calc_one_atomic_energy_from_descriptors(self, descriptors, one_type):
        """Compute one atomic contributions to the potential energy.

        Args:
            descriptors: The descriptors for that atom.
                tensor.
            types: The atom types, codified as an integers from 0 to
                n_types - 1.

        Returns:
            The atomic contributions to the energy.
        """
        embeddings = self.embed(one_type)
        combined_inputs = self.mixer(
            descriptors[jnp.newaxis, ...], embeddings[jnp.newaxis, ...]
        )
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return (one_type >= 0) * jnp.squeeze(results)

    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            The n_atoms contributions to the energy.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        return (types >= 0) * self.calc_atomic_energies_from_descriptors(
            descriptors, types
        )

    def calc_some_atomic_energies(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        """Compute some of the atomic contributions to the potential energy.

        Note that some_positions can also contain atoms that are not in
        all_positions, which is useful for padding and parallelizations
        but should be used with care.

        Args:
            some_positions: The (n_some_atoms, 3) vector with the Cartesian
                coordinates of each atom whose contribution to the energy
                should be computed.
            some_types: The atom types of the subset of atoms, codified as
                integers from 0 to n_types - 1. 
            all_positions: The positions of all atoms in the system.
            all_types: The atom types, codified as integers from 0 to
                n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            The n_atoms contributions to the energy.
        """
        some_descriptors = self.partial_descriptor_generator(
            all_positions, all_types, some_positions, cell
        )
        return (some_types >= 0) * self.calc_atomic_energies_from_descriptors(
            some_descriptors, some_types
        )

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._calc_gradient(positions, types, cell)

    def calc_potential_energy_and_forces(self, positions, types, cell):
        """Compute the total potential energy and all the forces.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            A two-element tuple. The first element is the sum of all atomic
            contributions to the potential energy. The second one is an
            (n_atoms, 3) vector containing all the forces.
        """
        energy, gradient = self._calc_value_and_gradient(positions, types, cell)
        return (energy, -gradient)


@dataclass
class NeuralILModelInfo:
    # A description of the general class of model
    model_name: str
    # A model version with an arbitrary factor
    model_version: str
    # A datetime object with the time of training
    timestamp: datetime.datetime
    # A cutoff radius for the descriptor generator
    r_cut: float
    # An upper bound to the radial index of the descriptors
    n_max: int
    # Alphabetical list of element symbols
    sorted_elements: list
    # Dimensionality of the element embedding
    embed_d: int
    # List of widths of the core layers
    core_widths: list
    # Dictionary of additional arguments to the model constructor
    constructor_kwargs: dict
    # Random seed used to create the RNG for training
    random_seed: int
    # Dictionary of model parameters created by flax
    params: FrozenDict
    # Any other information this kind of model requires
    specific_info: Any
