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

# Code used to calculate the coefficients of the spherical Bessel functions of
# the first kind. Modelled after the the Apache-licensed reference
# implementation of the spherical Bessel descriptors
# (doi: 10.1063/1.5111045 , https://github.com/harharkh/sb_desc ).

import jax
import numpy as np

from . import functions, roots


class SphericalBesselCoefficients:
    """Calculator and cache of the coefficients of the descriptors.

    The coefficients are available directly through the attributes c_1 and c_2.

    Args:
        n_max: The maximum value of n to be used when generating the spherical
            Bessel descriptors. This affects which coefficients are calculated.
        dtype: The floating-point type to be used for the calculations.

    Raises:
        ValueError if n_max is negative.
    """
    def __init__(self, n_max: int, dtype: np.dtype = np.float32):
        if n_max < 0:
            raise ValueError("n_max cannot be negative")
        self.n_max = n_max
        # We need to go one step beyond n_max to get all the coefficients we
        # need, so we keep an internal version incremented by one.
        self.__n_max = self.n_max + 1
        # Initialize the table of Bessel roots.
        self.roots = roots.SphericalBesselRoots(self.__n_max, dtype)
        # Create the tables to store the coefficients. Just like for the
        # roots, we store them regular arrays for convenience but only use
        # half of each array. Non initialized elements are set to NaN.
        self.c_1 = (np.nan * np.ones((self.__n_max + 1, self.__n_max + 1)
                                    )).astype(dtype)
        self.c_2 = (np.nan * np.ones((self.__n_max + 1, self.__n_max + 1)
                                    )).astype(dtype)
        for order in range(self.__n_max + 1):
            function = jax.jit(functions.create_j_l(order + 1, dtype))
            u_0 = self.roots.table[order, :self.__n_max - order + 1]
            u_1 = self.roots.table[order, 1:self.__n_max - order + 2]
            coeff = np.sqrt(2. / (u_0 * u_0 + u_1 * u_1))
            self.c_1[order, :self.__n_max - order +
                     1] = u_1 / function(u_0) * coeff
            self.c_2[order, :self.__n_max - order +
                     1] = u_0 / function(u_1) * coeff


if __name__ == "__main__":
    import datetime
    import timeit

    test_n_max = 139

    start = timeit.default_timer()
    coefficients = SphericalBesselCoefficients(test_n_max)
    end = timeit.default_timer()
    print("Time elapsed:", str(datetime.timedelta(seconds=end - start)))
