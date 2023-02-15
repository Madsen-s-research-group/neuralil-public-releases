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
import itertools
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, List, Sequence

import flax
import flax.linen
import jax
import jax.nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from neuralil.model import ResNetDense, ResNetIdentity, pairwise

# Code for the "deep ensembles", i.e., ensembles of NeuralIL models that
# use heteroscedastic losses. To that end, two heads are added that
# generate estimates of the uncertainties on energies and forces. Therefore,
# the model is restructured into a feature extractor and several heads.
# The changes with respect to the base NeuralIL models are more extensive than
# in the case of the committee.
# Please see the docstrings for the homoscedastic version of each function for
# more information.


class FeatureExtractor(flax.linen.Module):
    "MLP that processes the descriptors into features."
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    def setup(self):
        identity_counter = 0
        dense_counter = 0
        res_layers = []
        for i_width, o_width in pairwise(self.layer_widths):
            if i_width == o_width:
                identity_counter += 1
                name = f"ResNetIdentity_{identity_counter}_{i_width}"
                res_layers.append(
                    ResNetIdentity(
                        i_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
            else:
                dense_counter += 1
                name = f"ResNetDense_{dense_counter}_{i_width}_to_{o_width}"
                res_layers.append(
                    ResNetDense(
                        i_width,
                        o_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
        self.res_layers = res_layers

    def __call__(self, descriptors):
        result = descriptors
        for layer in self.res_layers:
            result = layer(result)
        return result


class Head(flax.linen.Module):
    "MLP that processes features into a scalar output."
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    def setup(self):
        total_widths = self.layer_widths + (1,)
        identity_counter = 0
        dense_counter = 0
        res_layers = []
        for i_width, o_width in pairwise(total_widths):
            if i_width == o_width:
                identity_counter += 1
                name = f"ResNetIdentity_{identity_counter}_{i_width}"
                res_layers.append(
                    ResNetIdentity(
                        i_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
            else:
                dense_counter += 1
                name = f"ResNetDense_{dense_counter}_{i_width}_to_{o_width}"
                res_layers.append(
                    ResNetDense(
                        i_width,
                        o_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
        self.res_layers = res_layers

    def __call__(self, descriptors):
        result = descriptors
        for layer in self.res_layers:
            result = layer(result)
        return result


class HeteroscedasticNeuralIL(flax.linen.Module):
    "NeuralIL with three heads."
    n_types: int
    embed_d: int
    r_cut: float
    descriptor_generator: Callable
    extractor_widths: List
    head_widths: List
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1
    )
    model_name: ClassVar[str] = "HeteroscedasticNeuralIL"
    model_version: ClassVar[str] = "0.2"

    def setup(self):
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        self.feature_extractor = FeatureExtractor(
            self.extractor_widths, self.activation_function, self.kernel_init
        )
        self.energy_head = Head(
            self.head_widths, self.activation_function, self.kernel_init
        )
        self.energy_denormalizer = flax.linen.Dense(1)
        self.sigma2e_head = Head(
            self.head_widths, self.activation_function, self.kernel_init
        )
        self.sigma2e_denormalizer = flax.linen.Dense(1, use_bias=False)
        self.sigma2f_head = Head(
            self.head_widths, self.activation_function, self.kernel_init
        )
        self.sigma2f_denormalizer = flax.linen.Dense(1, use_bias=False)
        self._calc_grad = jax.grad(
            self.calc_potential_energy, argnums=0, has_aux=True
        )

    def calc_heads_from_descriptors(self, descriptors, types):
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        features = self.feature_extractor(combined_inputs)
        return (
            (types >= 0)
            * jnp.squeeze(
                self.energy_denormalizer(self.energy_head(features))
            ),
            (types >= 0)
            * jnp.squeeze(
                jax.nn.softplus(
                    self.sigma2e_denormalizer(self.sigma2e_head(features))
                )
            )
            + (types < 0),
            (types >= 0)
            * jnp.squeeze(
                jax.nn.softplus(
                    self.sigma2f_denormalizer(self.sigma2f_head(features))
                )
            )
            + (types < 0),
        )

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.descriptor_generator(positions, types, cell)
        heads = self.calc_heads_from_descriptors(descriptors, types)
        return heads

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        return (
            jnp.squeeze(contributions[0].sum(axis=0)),
            (
                jnp.squeeze(contributions[1].sum(axis=0)),
                jnp.squeeze(contributions[2]),
            ),
        )

    def calc_all_results(self, positions, types, cell):
        grad_and_aux = self._calc_grad(positions, types, cell)
        return (
            self.calc_potential_energy(positions, types, cell)[0],
            -grad_and_aux[0],
            grad_and_aux[1][0],
            grad_and_aux[1][1],
        )


class DeepEnsemble(flax.linen.Module):
    h_neuralil: HeteroscedasticNeuralIL
    n_models: int
    model_name: ClassVar[str] = "DeepEnsemble"
    model_version: ClassVar[str] = "0.2"

    def setup(self):
        self.calc_heads_from_descriptors = flax.linen.vmap(
            HeteroscedasticNeuralIL.calc_heads_from_descriptors,
            in_axes=(None, None),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.n_models,
        )
        # Note the switch from gradient to Jacobian to account for the
        # ensemble axis.
        self._calc_jacobian = jax.jacrev(
            self.calc_potential_energy, argnums=0, has_aux=True
        )

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.h_neuralil.descriptor_generator(
            positions, types, cell
        )
        heads = self.calc_heads_from_descriptors(
            self.h_neuralil, descriptors, types
        )
        return heads

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        # Note the change of axis to account for the prepending of the
        # ensemble axis.
        return (
            jnp.squeeze(contributions[0].sum(axis=1)),
            (
                jnp.squeeze(contributions[1].sum(axis=1)),
                jnp.squeeze(contributions[2]),
            ),
        )

    def calc_all_results(self, positions, types, cell):
        jacobian_and_aux = self._calc_jacobian(positions, types, cell)
        return (
            self.calc_potential_energy(positions, types, cell)[0],
            -jacobian_and_aux[0],
            jacobian_and_aux[1][0],
            jacobian_and_aux[1][1],
        )


@dataclass
class HeteroscedasticNeuralILModelInfo:
    model_name: str
    model_version: str
    timestamp: datetime.datetime
    r_cut: float
    n_max: int
    sorted_elements: list
    embed_d: int
    extractor_widths: list
    head_widths: list
    constructor_kwargs: dict
    random_seed: int
    params: FrozenDict
    specific_info: Any


def update_energy_offset(params, offset):
    unfrozen = flax.serialization.to_state_dict(params)
    flat_params = {
        "/".join(k): v
        for k, v in flax.traverse_util.flatten_dict(unfrozen).items()
    }
    flat_params["params/energy_denormalizer/bias"] -= offset
    unfrozen = flax.traverse_util.unflatten_dict(
        {tuple(k.split("/")): v for k, v in flat_params.items()}
    )
    return flax.serialization.from_state_dict(params, unfrozen)
