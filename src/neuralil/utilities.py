#!/usr/bin/env python
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

import os
import struct

import numpy as onp
import scipy as sp
import scipy.integrate
import scipy.stats

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import jax.random

# This module contains miscellaneous utilities used throughout the code.

__all__ = [
    "draw_urandom_int32", "create_array_shuffler", "create_gp_activation"
]


def draw_urandom_int32():
    "Generate a 32-bit random integer suitable for seeding the PRNG in JAX."
    return struct.unpack("I", os.urandom(4))[0]


def create_array_shuffler(rng):
    """Create a function able to reshuffle arrays in a consistent manner.

    Args:
        rng: A JAX splittable pseudo-random number generator, which is consumed.

    Returns:
        A function of a single argument that will return a copy of that argument
        reshuffled along the first axis as a JAX array. The result will always
        be the same for the same input. Arrays of the same length will be sorted
        correlatively.
    """
    def nruter(in_array):
        return jax.random.permutation(rng, jnp.asarray(in_array))

    return nruter


def update_energy_offset(params, offset):
    """Update the bias of the last layer of a model.
    
    This is normaly done so that a model trained on forces uses the right
    origin of energies.

    Args:
        params: The FrozenDict containing the parameters of the model.
        offset: The energy per atom to be removed from the bias.

    Returns:
        An updated version of the 'params' FrozenDict.
    """
    unfrozen = flax.serialization.to_state_dict(params)
    flat_params = {
        "/".join(k): v
        for k,
        v in flax.traverse_util.flatten_dict(unfrozen).items()
    }
    flat_params["params/denormalizer/bias"] -= offset
    unfrozen = flax.traverse_util.unflatten_dict(
        {tuple(k.split("/")): v
         for k, v in flat_params.items()}
    )
    return flax.serialization.from_state_dict(params, unfrozen)


def create_gp_activation(original_function):
    """Create a shifted and scaled Gaussian-Poincaré activation.

    Take an original differentiable activation function f(x) and compute
    a and b so that g(x) = a*f(x) + b fulfills the condition that the expected
    values of both g(x)**2 and g'(x)**2 are one when x is distributed
    according to a standard Gaussian distribution.

    Args:
        original_function: A JAX-differentiable activation function of one
            variable.

    Returns:
        A single-argument function that computes a*f(x) + b.
    """
    expected_f = sp.integrate.quad(
        lambda x: sp.stats.norm.pdf(x) * original_function(x),
        -onp.infty,
        onp.infty
    )[0]
    expected_f2 = sp.integrate.quad(
        lambda x: sp.stats.norm.pdf(x) * original_function(x)**2,
        -onp.infty,
        onp.infty
    )[0]
    scalar_derivative = jax.grad(original_function)
    expected_fp2 = sp.integrate.quad(
        lambda x: sp.stats.norm.pdf(x) * scalar_derivative(x)**2,
        -onp.infty,
        onp.infty
    )[0]
    scale_factor = 1. / onp.sqrt(expected_fp2)
    quadratic_equation = [
        1., 2. * scale_factor * expected_f, scale_factor**2 * expected_f2 - 1.
    ]
    offset = min(onp.roots(quadratic_equation))

    def nruter(x):
        return scale_factor * original_function(x) + offset

    return nruter
