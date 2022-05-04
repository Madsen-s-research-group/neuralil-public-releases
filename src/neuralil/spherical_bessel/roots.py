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

# Code used to calculate the roots of the spherical Bessel functions of the
# first kind. Inspired on the C implementation by Jeremy K. Mason, also under
# an Apache license and included as part of the reference implementation of
# the spherical Bessel descriptors (doi: 10.1063/1.5111045 ,
# https://github.com/harharkh/sb_desc ). However, here we use the scipy root
# finding routines.

import jax
import numpy as np
import scipy as sp
import scipy.optimize

from . import functions

# The calculation starts with the roots of the order-0 function and
# proceeds upwards. Therefore, caching is essential.


class SphericalBesselRoots:
    """Calculator and cache of the roots of spherical Bessel functions.

    Access to the roots is supposed to happen directly through the "table"
    attribute. The first index is the order of the function, and the second
    is the order of the root. A negative value means that the corresponding
    root has not been calculated.

    Args:
        n_max: The maximum value of n to be used when generating the spherical
            Bessel descriptors. This affects which roots are calculated.
        dtype: The floating-point type to be used for the calculations.

    Raises:
        ValueError if n_max is negative.
    """
    def __init__(self, n_max: int, dtype: np.dtype = np.float32):
        if n_max < 0:
            raise ValueError("n_max cannot be negative")
        self.n_max = n_max
        # Initialize the table of precomputed values. Note that, even though
        # a rectangular NumPy array is used for convenience and performance,
        # the higher the order of the function, the fewer roots are computed.
        self.table = -np.ones((self.n_max + 1, self.n_max + 2))
        # The zeros of j_0(r) are trivially generated.
        self.table[0, :] = np.pi * np.arange(1, n_max + 3)
        # Proceed upwards from there, by using the fact that the roots of the
        # order-l function bracket the roots of the order-(l+1) function, per
        # Abramowitz and Stegun.
        for order in range(1, self.n_max + 1):
            # This could be made faster by using the SciPy implementation of
            # the spherical Bessels functions directly, but it only needs
            # to be done once and we get values of the roots that are
            # better adapted to our implementation.
            function = jax.jit(functions.create_j_l(order, dtype))
            for n in range(self.n_max + 2 - order):
                left = self.table[order - 1, n]
                right = self.table[order - 1, n + 1]
                self.table[order, n] = sp.optimize.brentq(function, left, right)


if __name__ == "__main__":
    # Time the calculation and check the maximum error in the roots.
    import datetime
    import timeit
    test_n_max = 140

    print(f"Building the table of roots up to n_max={test_n_max}")
    start = timeit.default_timer()
    roots = SphericalBesselRoots(test_n_max)
    end = timeit.default_timer()
    print("Time elapsed:", str(datetime.timedelta(seconds=end - start)))

    print("Computing the maximum error of the roots")
    start = timeit.default_timer()
    max_error = 0.
    for order in range(0, test_n_max + 1):
        function = functions.create_j_l(order)
        for n in range(test_n_max + 2 - order):
            max_error = max(max_error, function(roots.table[order, n]))
    end = timeit.default_timer()
    print("Time elapsed:", str(datetime.timedelta(seconds=end - start)))
    print("Maximum error:", max_error)
