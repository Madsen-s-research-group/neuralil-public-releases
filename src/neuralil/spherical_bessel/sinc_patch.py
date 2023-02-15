# Fixes issue https://github.com/google/jax/issues/10750 until new jax version

from functools import partial

import jax
import numpy as np
from jax import lax
from jax._src import dtypes
from jax._src.api import custom_jvp, jit
from jax._src.lax import lax as lax_internal
from jax._src.numpy.util import (
    _check_arraylike,
    _promote_args,
    _promote_args_inexact,
    _promote_dtypes_inexact,
    _promote_shapes,
    _where,
    _wraps,
)

_lax_const = lax_internal._const


@jax.jit
def _fixed_sinc(x):
    _check_arraylike("sinc", x)
    (x,) = _promote_dtypes_inexact(x)
    eq_zero = lax.eq(x, _lax_const(x, 0))
    pi_x = lax.mul(_lax_const(x, np.pi), x)
    safe_pi_x = _where(eq_zero, _lax_const(x, 1), pi_x)
    return _where(
        eq_zero,
        _sinc_maclaurin(0, pi_x),
        lax.div(lax.sin(safe_pi_x), safe_pi_x),
    )


@partial(custom_jvp, nondiff_argnums=(0,))
def _sinc_maclaurin(k, x):
    # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
    # compute the monomial term in the jvp rule)
    # TODO(mattjj): see https://github.com/google/jax/issues/10750
    if k % 2:
        return x * 0
    else:
        return x * 0 + _lax_const(x, (-1) ** (k // 2) / (k + 1))


@_sinc_maclaurin.defjvp
def _sinc_maclaurin_jvp(k, primals, tangents):
    (x,), (t,) = primals, tangents
    return _sinc_maclaurin(k, x), _sinc_maclaurin(k + 1, x) * t
