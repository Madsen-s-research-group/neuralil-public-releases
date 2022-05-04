# Copyright 2019-2022 The NeuralIL contributors
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
import flax.linen.initializers
import jax
import jax.nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from neuralil.bessel_descriptors import center_at_atoms, center_at_points


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


def _aux_function_f(t):
    "First auxiliary function used in the definition of the smooth bump."
    return jnp.where(t > 0., jnp.exp(-1. / jnp.where(t > 0., t, 1.)), 0.)


def _aux_function_g(t):
    "Second auxiliary function used in the definition of the smooth bump."
    f_of_t = _aux_function_f(t)
    return f_of_t / (f_of_t + _aux_function_f(1. - t))


def smooth_cutoff(r, r_switch, r_cut):
    """One-dimensional smooth cutoff function based on a smooth bump.

    Args:
        r: The radii at which the function must be evaluated.
        r_switch: The radius at which the function starts differing from 1.
        r_cut: The radius at which the function becomes exactly 0.
    """
    r_switch2 = r_switch * r_switch
    r_cut2 = r_cut * r_cut

    return 1. - _aux_function_g((r * r - r_switch2) / (r_cut2 - r_switch2))


def calc_morse_mixing_radii(radii, abd_probe, abd_source):
    """Compute the mixing radii between single-species Morse potentials.

    This radii are computed based on the repulsive part alone, as the solution
    of the equations

    r1 + r2 = radii
    phi_1'(2 * r1) = phi_2'(2 * r2)

    where phi' is the derivative of d * exp(-2 * a *(r - b)).
    Reference: J. Chem. Phys. 59 (1973) 2464.

    The function is intended to be used for two sets of atoms, all at once. We
    call the first set of atoms "probe" and the second one "source".

    Args:
        radii: The distances between atoms as an (n_probe, n_source) array.
        abd_probe: An (n_probe, 3) vector of Morse parameters, sorted as
            (a, b, d) in the notation of the equation above.
        abd_source: An (n_source, 3) vector of Morse parameters, sorted as
            (a, b, d) in the notation of the equation above.

    Returns:
        An (n_probe, n_source) array with the result.
    """
    a_probe = abd_probe[:, 0][:, jnp.newaxis]
    b_probe = abd_probe[:, 1][:, jnp.newaxis]
    d_probe = abd_probe[:, 2][:, jnp.newaxis]
    a_source = abd_source[:, 0][jnp.newaxis, :]
    b_source = abd_source[:, 1][jnp.newaxis, :]
    d_source = abd_source[:, 2][jnp.newaxis, :]
    return (
        a_source * radii +
        .25 * jnp.log(a_probe * d_probe / (a_source * d_source)) + .5 *
        (a_probe * b_probe - a_source * b_source)
    ) / (
        a_probe + a_source
    )


class RepulsiveMorseModel(flax.linen.Module):
    """Trainable repulsive Morse potential with per-element parameters.

    The potential function is parameterized as

    phi(r) = d * exp(-2 * a * (r - b))

    multiplied by a switching function with a fixed cutoff raduzs and a
    trainable switching radius to guarantee the smoothness of the forces.

    The parameters (a, b, d) are extracted from a three-component embedding
    vector v as
    a = a_min + v[0] ** 2
    b = b_min + v[1] ** 2
    d = d_min + v[2] ** 2
    to guarantee their positivity. The interaction between two atoms, of
    elements a and b, contributes a potential energy

    E_pot = .5 * (phi[v_a](2 * r1) + phi[v_b](2 * r2))

    For the meaning of r1 and r2, see the docstring of the function
    calc_morse_mixing_radius().

    Args:
        n_types: The number of atom types in the system.
        r_cut: The cutoff radius.
        a_min: The minimum possible value for the parameter a.
        b_min: The minimum possible value for the parameter b.
        d_min: The minimum possible value for the parameter d.
    """
    n_types: int
    r_cut: float
    a_min: float
    b_min: float
    d_min: float

    def setup(self):
        self.switch_param = self.param(
            "switch_param", flax.linen.initializers.lecun_normal(), (1, 1)
        )
        self.embed = flax.linen.Embed(self.n_types, 3)

    def calc_atomic_energies(self, radii, probe_types, source_types):
        """Compute the Morse repulsive contributions to the potential energy.

        Two groups of atoms can be specified: the "source" atoms that take part
        in the interacions and the "probe" atoms whose energies we compute.
        They can partially or completely overlap.

        Args:
            radii: The (n_probe, n_source) matrix of interatomic distances.
            probe_types: The atom types of the "probe" atoms, codified as
                integers from 0 to n_types - 1.
            source_types: The atom types of the "source" atoms, codified as
                integers from 0 to n_types - 1.

        Returns:
            The n_probe contributions to the energy from the "probe" atoms.
        """
        mask = jnp.logical_or(
            probe_types[:, jnp.newaxis] < 0, source_types[jnp.newaxis, :] < 0
        )
        radii = radii + 2. * mask * self.r_cut
        abd_probe = (
            jnp.array([self.a_min, self.b_min, self.d_min])[jnp.newaxis, :] +
            self.embed(probe_types)**2
        )
        abd_source = (
            jnp.array([self.a_min, self.b_min, self.d_min])[jnp.newaxis, :] +
            self.embed(source_types)**2
        )
        r_morse = calc_morse_mixing_radii(radii, abd_probe, abd_source)

        a = abd_probe[:, 0]
        b = abd_probe[:, 1]
        d = abd_probe[:, 2]
        contributions = (
            d[:, jnp.newaxis] * jnp.exp(
                -2. * a[:, jnp.newaxis] * (2. * r_morse - b[:, jnp.newaxis])
            )
        )
        # Zero out the contribution from the interaction of each atom with
        # itself. This trick only works with potentials that can be safely
        # evaluated at r=0.
        contributions *= jnp.logical_not(jnp.isclose(0., radii))
        # The switching radius will always lie between r_cut / 2. and r_cut, to
        # avoid nonsensical situations.
        r_switch = .5 * (
            1. + jax.nn.sigmoid(jnp.squeeze(self.switch_param))
        ) * self.r_cut
        cutoffs = smooth_cutoff(radii, r_switch, self.r_cut)
        return .5 * (cutoffs * contributions).sum(axis=1)


class TrivialCore(flax.linen.Module):
    """Basic multilayer perceptron.

    This model takes the descriptors of each atom (Bessel + embedding,
    concatenated or otherwise combined) as inputs and calculates that atom's
    contribution to the potential energy.

    Args:
        layer_widths: The sequence of layer widths, excluding the output
            layer, which always has a width equal to one.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
        use_intermediate_biases: If set to False, remove the biases from the
            intermediate layers.
        kernel_init: Initializer for the weight matrices (default:
            flax.linen.initializers.lecun_normal).
    """
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish
    use_intermediate_biases: bool = True
    kernel_init: Callable = flax.linen.initializers.lecun_normal()

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(
                self.layer_widths[0],
                use_bias=self.use_intermediate_biases,
                kernel_init=self.kernel_init,
                name="Inlet"
            )(descriptors)
        )
        for i_w, w in enumerate(self.layer_widths[1:]):
            result = self.activation_function(
                flax.linen.Dense(
                    w,
                    use_bias=self.use_intermediate_biases,
                    kernel_init=self.kernel_init,
                    name=f"Stage_{i_w + 1}"
                )(result)
            )
        return self.activation_function(
            flax.linen.Dense(1, kernel_init=self.kernel_init,
                             name="Outlet")(result)
        )


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
        use_intermediate_biases: If set to False, remove the biases from the
            layers wrapped in a LayerNorm.
        kernel_init: Initializer for the weight matrices (default:
            flax.linen.initializers.lecun_normal).
    """
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish
    use_intermediate_biases: bool = True
    kernel_init: Callable = flax.linen.initializers.lecun_normal()

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(
                self.layer_widths[0],
                kernel_init=self.kernel_init,
                name="Inlet"
            )(descriptors)
        )
        for i_w, w in enumerate(self.layer_widths[1:]):
            result = self.activation_function(
                flax.linen.LayerNorm()(
                    flax.linen.Dense(
                        w,
                        use_bias=self.use_intermediate_biases,
                        kernel_init=self.kernel_init,
                        name=f"Stage_{i_w + 1}"
                    )(result)
                )
            )
        return self.activation_function(
            flax.linen.Dense(1, kernel_init=self.kernel_init,
                             name="Outlet")(result)
        )


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
    model_version: ClassVar[str] = "0.5"

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


class NeuralILwithMorse(flax.linen.Module):
    """Wrapper model around the core layers to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both). This version includes a
    repulsive Morse contribution.

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
    model_name: ClassVar[str] = "NeuralIL+Morse"
    model_version: ClassVar[str] = "0.4"

    def setup(self):
        # TODO: Rethink the baseline a, b and d.
        # This model takes care of the repulsive part of the potential.
        self.morse = RepulsiveMorseModel(
            self.n_types, self.r_cut, 1e-2, 1e-2, 1e-2
        )
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

        Only the NN contribution is included.

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

        Only the NN contribution is included.

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
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            descriptors, types
        )
        morse_contributions = self.calc_morse_energies(positions, types, cell)
        return (types >= 0) * (nn_contributions + morse_contributions)

    def calc_morse_energies(self, positions, types, cell):
        """Compute the Morse part of the contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atom types, codified as integers from 0 to n_types - 1.
            cell: The (3, 3) matrix representing the simulation box if periodic
                boundary conditions are in effect. If it is not periodic along
                one or more directions, signal that fact with one of more zero
                vectors.

        Returns:
            The n_atoms contributions to the energy from the Morse part of the
            potential.
        """
        radii = center_at_atoms(positions, cell)[1]
        morse_contributions = self.morse.calc_atomic_energies(
            radii, types, types
        )
        return (types >= 0) * morse_contributions

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
        nn_contributions = self.calc_atomic_energies_from_descriptors(
            some_descriptors, some_types
        )
        morse_contributions = self.calc_some_morse_energies(
            some_positions, some_types, all_positions, all_types, cell
        )
        return (some_types >= 0) * (nn_contributions + morse_contributions)

    def calc_some_morse_energies(
        self, some_positions, some_types, all_positions, all_types, cell
    ):
        """Compute some of the Morse contributions to the potential energy.

        Note that some_positions can also contain atoms that are not in
        all_positions, which is useful for padding and parallelizations but
        should be used with care.

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
            The n_atoms contributions to the energy from the Morse part of the
            potential.
        """
        radii = center_at_points(all_positions, some_positions, cell)[1]
        morse_contributions = self.morse.calc_atomic_energies(
            radii, some_types, all_types
        )
        return (some_types >= 0) * morse_contributions

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
