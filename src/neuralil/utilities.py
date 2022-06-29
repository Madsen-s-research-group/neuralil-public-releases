#!/usr/bin/env python
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

import os
import struct

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import jax.random

# This module containst miscellaneous utilities used throughout the code.

__all__ = ["draw_urandom_int32", "create_array_shuffler"]


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
        for k, v in flax.traverse_util.flatten_dict(unfrozen).items()
    }
    flat_params["params/denormalizer/bias"] -= offset
    unfrozen = flax.traverse_util.unflatten_dict(
        {tuple(k.split("/")): v
         for k, v in flat_params.items()}
    )
    return flax.serialization.from_state_dict(params, unfrozen)
