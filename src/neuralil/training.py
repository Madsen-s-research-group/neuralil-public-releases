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

from collections import namedtuple
from contextlib import ExitStack

import jax
import jax.flatten_util
import jax.nn
import jax.numpy as jnp
import jax.random
import jax.tree_util
import optax
import tqdm
try:
    from learned_optimization.optimizers.opt_to_optax import (
        GradientTransformationWithExtraArgs,
    )
except ImportError:
    from optax import (
        GradientTransformationExtraArgs as GradientTransformationWithExtraArgs,
    )
from learned_optimization.research.general_lopt import prefab

# This module contains code specific to the training step.

__all__ = [
    "create_log_cosh",
    "create_velo_minimizer",
    "create_training_step",
    "create_training_epoch",
    "fmae_validation_statistic",
    "emae_validation_statistic",
    "frmse_validation_statistic",
    "ermse_validation_statistic",
    "create_elogcosh_validation_statistic",
    "create_flogcosh_validation_statistic",
    "create_validation_step",
]


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.0:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            jax.nn.softplus(2.0 * parameter * x) - jnp.log(2)
        ) / parameter - x

    return nruter


# Adapted from the implementation in TensorFlow Probability.
@jax.custom_jvp
def _log_cosh(x):
    """Numerically stable implementation of log(cosh(x))."""
    dtype = x.dtype
    bound = 45.0 * jnp.power(jnp.finfo(dtype).tiny, 1.0 / 6.0)
    abs_x = jnp.abs(x)
    mask = abs_x <= bound
    safe_argument = jnp.where(mask, 1.0, abs_x)
    logcosh = abs_x + jax.nn.softplus(-2.0 * safe_argument) - jnp.log(2)
    return jnp.where(
        mask,
        jnp.exp(jnp.log(abs_x) + jnp.log1p(-jnp.square(abs_x) / 6.0)),
        logcosh,
    )


@_log_cosh.defjvp
def _log_cosh_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    primal_out = _log_cosh(x)
    tangent_out = jnp.tanh(x) * xdot
    return (primal_out, tangent_out)


def create_heteroscedastic_log_cosh(scale):
    """Return a differentiable implementation of the heteroscedastic log cosh
    loss.

    Args:
        scale: The positive parameter determining the transition between
            the linear and quadratic regimes. Note that this is the inverse
            of `parameter` in the homoscedastic implementation, and is also
            dimensionless.

    Returns:
        A (float, float)->float function to compute a contribution to the loss.
        The first parameter is the signed deviation and the second is sigma.

    Raises:
        ValueError: If parameter is not positive.
    """
    if scale <= 0.0:
        raise ValueError("the scale must be positive")

    def nruter(x, sigma):
        return _log_cosh(jnp.pi * x / 2.0 / sigma / scale) + jnp.log(sigma)

    return nruter


def create_one_cycle_minimizer(
    n_batches, min_learning_rate, max_learning_rate, fin_learning_rate
):
    """Create an optax minimizer with an 1-cycle schedule.

    The learning rate starts at min_learning_rate, is ramped up linearly until
    reaching max_learning_rate 45% into the cycle, decreases back linearly to
    min_learning_rate, which is therefore hit at the 90% point, and finally
    drops linearly to fin_learning_rate.

    Args:
        n_batches: The number of steps in the cycle.
        min_learning rate: The starting point for the schedule.
        max_learning_rate: The maximum learning rate reached along the way.
        fin_learning_rate: Final learning rate, typically a significantly lower
            value than min_learning rate.

    Returns:
        An optax minimizer with the required parameters.

    Raises:
        ValueError if the parameters do not make sense.
    """
    if min(min_learning_rate, max_learning_rate, fin_learning_rate) <= 0.0:
        raise ValueError("all learning rates must be positive")
    if min_learning_rate >= max_learning_rate:
        raise ValueError(
            "min_learning_rate should be lower than max_learning_rate"
        )
    if fin_learning_rate >= min_learning_rate:
        raise ValueError(
            "fin_learning_rate should be lower than min_learning_rate"
        )
    training_schedule = optax.linear_onecycle_schedule(
        n_batches,
        max_learning_rate,
        0.45,
        0.9,
        max_learning_rate / min_learning_rate,
        max_learning_rate / fin_learning_rate,
    )
    # Integrate the training schedule in the optimizer.
    # Note the minus sign, required for gradient **descent**.
    # Yogi with training schedule
    optimizer = optax.chain(
        optax.scale_by_yogi(),
        optax.scale_by_schedule(training_schedule),
        optax.scale(-1.0),
    )

    return optimizer


def reset_one_cycle_minimizer(optimizer_state):
    """Reset the state of the 1-cycle training schedule to begin a new epoch.

    Args:
        optimizer_state: The current state of the optimizer.

    Returns:
        A copy of optimizer_state with the learning-rate schedule reset to the
        beginning of the cycle.
    """
    return type(optimizer_state)(
        [
            optax.ScaleByScheduleState(0)
            if isinstance(state, optax.ScaleByScheduleState)
            else state
            for state in optimizer_state
        ]
    )


def create_velo_minimizer(n_batches: int, n_epochs: int):
    """Create a VeLO nonlinear optimizer.

    Instead of any variation of stochastic gradient descent, VeLO uses a
    pretrained NN to update the weights of the model. See
    https://arxiv.org/pdf/2211.09760.pdf for details. There is no learning
    rate to be adjusted.

    Args:
        n_batches: The number of barches per epoch.
        n_epochs: The number of epochs used for training.

    Returns:
        A VeLO optimizer with a standard optax interface.
    """
    return prefab.optax_lopt(n_batches * n_epochs)


class BatchedIterator:
    """Iterator over batches created from a set of arrays.

    Batches are understood as non-overlapping random subsets of the arrays with
    the desired size. If the common length of the first axis of the arrays is
    not a multiple of n_batch, the union of all batches will be smaller than the
    input arrays.

    Args:
        n_batch: The size of the desired random batches.
        rng: A JAX splittable pseudo-random number generator, which is consumed.
        in_arrays: A sequence of arrays of the same length along the first axis.

    Raises:
        ValueError if the arguments do not have the same length of if they are
        shorter than n_batch.
    """

    def __init__(self, n_batch, rng, *in_arrays):
        self._in_arrays = [jnp.asarray(a) for a in in_arrays]
        lengths = [a.shape[0] for a in self._in_arrays]
        l_a = lengths[0]
        for l_b in lengths[1:]:
            if l_b != l_a:
                raise ValueError(
                    "all input arrays must have the same first dimension"
                )
        if l_a < n_batch:
            raise ValueError(
                "the arrays are too short for the batch size requested"
            )
        n_batches = l_a // n_batch
        self._n_batches = n_batches
        n_used = n_batch * n_batches
        perm = jax.random.permutation(rng, l_a)[:n_used]
        self._perms = perm.reshape((n_batches, n_batch))

    def __len__(self):
        "Return the total number of batches."
        return self._n_batches

    def __iter__(self):
        "Initialize the iteration. Normally invoked through iter()."
        self._pos = 0
        return self

    def __next__(self):
        "Perform a single step of the iteration."
        if self._pos < self._n_batches:
            indices = self._perms[self._pos]
            nruter = tuple(a[indices, ...] for a in self._in_arrays)
            self._pos += 1
            return nruter
        raise StopIteration


def create_training_step(model, optimizer, calc_loss_contribution):
    """Create a function that takes care of a single training step.

    Args:
        model: The Flax model object to be trained.
        optimizer: The Optax optimizer that will drive the training.
        calc_loss_contribution: A function used to compute the
            contribution to the loss of a single point. It must take four
            arguments, in the order
            (pred_energy, pred_forces, obs_energy, obs_forces)
            and return a single scalar.

    Returns:
        A function that will compute the loss and its gradient and update the
        state of the model and the optimizer. It takes six arguments in the
        order (optimizer_state, model_params, positions_batch, types_batch,
        cells_batch, energies_batch, forces_batch) and returns three values:
        (loss_val, optimizer_state, model_params).
    """

    @jax.jit
    def training_step(
        optimizer_state,
        model_params,
        positions_batch,
        types_batch,
        cells_batch,
        energies_batch,
        forces_batch,
    ):
        """Run a single step of the training process.

        Args:
            optimizer_state: The current state of the Optax optimizer.
            model_params: The current state of the Flax model.
            positions_batch: The minibatch of (natoms, 3) arrays of Cartesian
                atomic positions.
            types_batch: The minibatch of (natoms,) arrays of integer atomic
                types.
            cells_batchs: The minibatch of (3, 3) matrices of cell vectors.
            energies_batch: The minibatch of observed total energies.
            forces_batch: The minibatch of (natoms, 3) observed forces on atoms.

        Returns:
            A tuple of three objects: the value of the loss after the step,
            the new state of the optimizer, and the new parameters of the model.
        """

        def calc_loss(model_params):
            """Compute the total loss by averaging over samples.

            Args:
                model_params: The current parameters of the Flax model.
            """

            def invoke_loss_contribution(
                positions, types, cell, obs_energy, obs_forces
            ):
                "Compute the contribution to the loss of a single data point."
                pred_energy, pred_forces = model.apply(
                    model_params,
                    positions,
                    types,
                    cell,
                    method=model.calc_potential_energy_and_forces,
                )
                pred_forces = jnp.where(
                    jnp.expand_dims(types, axis=-1) >= 0,
                    pred_forces,
                    jnp.zeros(3),
                )
                return calc_loss_contribution(
                    pred_energy, pred_forces, obs_energy, obs_forces, types
                )

            return jax.vmap(jax.checkpoint(invoke_loss_contribution))(
                positions_batch,
                types_batch,
                cells_batch,
                energies_batch,
                forces_batch,
            ).mean(axis=0)

        calc_loss_and_grad = jax.value_and_grad(calc_loss)
        loss_val, loss_grad = calc_loss_and_grad(model_params)

        # Treat the VeLO optimizer differently
        if isinstance(
            optimizer,
            GradientTransformationWithExtraArgs,
        ):
            kwargs = dict(extra_args=dict(loss=loss_val))
        else:
            kwargs = dict()
        updates, optimizer_state = optimizer.update(
            loss_grad, optimizer_state, model_params, **kwargs
        )
        model_params = optax.apply_updates(model_params, updates)
        return (loss_val, optimizer_state, model_params)

    return training_step


def create_training_epoch(
    positions_train,
    types_train,
    cells_train,
    energies_train,
    forces_train,
    n_batch,
    training_step_driver,
    rng,
    progress_bar=True,
):
    """Create a driver for a training epoch.

    Args:
        positions_train: The full batch of (natoms, 3) arrays of Cartesian
            atomic positions to be used for training.
        types_train: The full batch of (natoms,) arrays of integer atomic types
            to be used for training.
        cells_train: The full batch of (3, 3) matrices of cell vectors to be
            used for training.
        energies_train: The full batch of observed total energies to be used for
            training.
        forces_train: The full batch of (natoms, 3) observed forces on atoms to
            be used for training.
        n_batch: The size of a minibatch.
        training_step_driver: The function taking care of each training step.
            See the documentation for create_training_step().
        rng: A JAX splittable pseudo-random number generator, which is consumed.
        progress_bar: A boolean toggle determining whether a progress bar will
            be shown on screen.

    Returns:
        A function that takes the state of the optimizer and the current state
        of the model as arguments, and returns the updated states of both
        objects.
    """

    def training_epoch(optimizer_state, model_params):
        """Run a full epoch of the training process.

        Args:
            optimizer_state: The current state of the Optax optimizer.
            model_params: The current state of the Flax model parameters.

        Returns:
            A tuple (optimizer_state, model_params) with the updated versions of
                the arguments.
        """
        training_epoch.rng, rng = jax.random.split(training_epoch.rng)
        batched_iterator = BatchedIterator(
            n_batch,
            rng,
            positions_train,
            types_train,
            cells_train,
            energies_train,
            forces_train,
        )

        with ExitStack() as with_stack:
            iterator = iter(batched_iterator)
            if progress_bar:
                iterator = with_stack.enter_context(
                    tqdm.auto.tqdm(
                        iter(batched_iterator),
                        total=len(batched_iterator),
                        dynamic_ncols=True,
                    )
                )
                iterator.set_description(f"EPOCH #{training_epoch.epoch + 1}")
            for (
                positions_batch,
                types_batch,
                cells_batch,
                energies_batch,
                forces_batch,
            ) in iterator:
                (loss, optimizer_state, model_params) = training_step_driver(
                    optimizer_state,
                    model_params,
                    positions_batch,
                    types_batch,
                    cells_batch,
                    energies_batch,
                    forces_batch,
                )

                if progress_bar:
                    iterator.set_postfix(loss=loss)

        training_epoch.epoch += 1
        return (optimizer_state, model_params)

    training_epoch.rng = rng
    training_epoch.epoch = 0
    return training_epoch


# A validation statistic, like the MAE or the RMSE, is defined by a "map"
# function to be applied to each data point, a "reduce" operation that can
# be applied to arbitrary sequences of results from the "map" function
# and a "postprocess" function that is applied to the final result of
# the reduction. The "reduce" operation must be commutative and associative.
# The final result is equivalent to
# postprocess_function(reduce_function([map_function(i) for i in data]), data)
# but reduce_function is actually vectorized over batches.
# The prototypes are:
# map_function(pred_energy, pred_forces, obs_energy, obs_forces, types)
# reduce_function(sequence_of_results_from_map)
# postprocess_function(final_reduction, n_data)
ValidationStatistic = namedtuple(
    "ValidationStatistic",
    ["map_function", "reduce_function", "postprocess_function"],
)


# Definitions for a few simple statistics.
def _fmae_map_function(
    pred_energy, pred_forces, obs_energy, obs_forces, types
):
    return (jnp.fabs(obs_forces - pred_forces)).sum() / (
        3 * (types >= 0).sum()
    )


def _emae_map_function(
    pred_energy, pred_forces, obs_energy, obs_forces, types
):
    return jnp.fabs(obs_energy - pred_energy) / (types >= 0).sum()


def _mae_postprocessing_function(in_value, n_samples):
    return in_value / float(n_samples)


def _frmse_map_function(
    pred_energy, pred_forces, obs_energy, obs_forces, types
):
    return ((obs_forces - pred_forces) ** 2).sum() / (3 * (types >= 0).sum())


def _ermse_map_function(
    pred_energy, pred_forces, obs_energy, obs_forces, types
):
    return ((obs_energy - pred_energy) / (types >= 0).sum()) ** 2


def _rmse_postprocessing_function(in_value, n_samples):
    return jnp.sqrt(in_value / float(n_samples))


# Mean absolute error in the forces.
fmae_validation_statistic = ValidationStatistic(
    _fmae_map_function, jnp.sum, _mae_postprocessing_function
)

# Mean absolute error in the energies.
emae_validation_statistic = ValidationStatistic(
    _emae_map_function, jnp.sum, _mae_postprocessing_function
)

# Root mean square error in the forces.
frmse_validation_statistic = ValidationStatistic(
    _frmse_map_function, jnp.sum, _rmse_postprocessing_function
)

# Root mean square error in the energies.
ermse_validation_statistic = ValidationStatistic(
    _ermse_map_function, jnp.sum, _rmse_postprocessing_function
)


# Average logcosh of the energies.
def create_elogcosh_validation_statistic(logcosh_parameter):
    log_cosh = create_log_cosh(logcosh_parameter)

    def map_function(pred_energy, pred_forces, obs_energy, obs_forces, types):
        return log_cosh((obs_energy - pred_energy) / (types >= 0).sum())

    return ValidationStatistic(
        map_function, jnp.sum, _mae_postprocessing_function
    )


# Average logcosh of the forces.
def create_flogcosh_validation_statistic(logcosh_parameter):
    log_cosh = create_log_cosh(logcosh_parameter)

    def map_function(pred_energy, pred_forces, obs_energy, obs_forces, types):
        delta_forces = jnp.sqrt(((obs_forces - pred_forces) ** 2).sum(axis=-1))
        return log_cosh(delta_forces).sum() / (types >= 0).sum()

    return ValidationStatistic(
        map_function, jnp.sum, _mae_postprocessing_function
    )


def _create_individual_validation_calculator(model, validation_statistics):
    """Create a function to evaluate validation statistics for a single point.

    Args:
        model: The Flax model object to be evaluated.
        validation_statistics: A dictionary of ValidationStatistics named
            tuples.

    Returns:
        A function that takes six parameters: (model_params, positions, types,
        cell, energy, forces) and returns a dictionary of contributions to the
        validation statistics from a single data point.
    """

    def nruter(model_params, positions, types, cell, energy, forces):
        """Evaluate the model on a single data point.

        This function will calculate the contribution to the validation
        statistics of a single data point. Those values will then be reduced
        and postprocessed to obtain the validation statistics for the whole
        model.

        Args:
            model_params: The current state of the Flax model parameters.
            positions: The (natoms, 3) Cartesian coordinates for this
                configuration.
            types: The (natoms,) integer array of atom types.
            cell: The (3, 3) array with the simulation box vectors.
            energy: The observed energy for this configuration.
            forces: The (natoms, 3) observed forces in this configuration.

        Returns:
            A dictionary of results of the "map" function of each validation
            statistic.
        """
        pred_energy, pred_forces = model.apply(
            model_params,
            positions,
            types,
            cell,
            method=model.calc_potential_energy_and_forces,
        )
        pred_forces = jnp.where(
            jnp.expand_dims(types, axis=-1) >= 0,
            pred_forces,
            jnp.zeros(3),
        )
        return {
            k: validation_statistics[k].map_function(
                pred_energy, pred_forces, energy, forces, types
            )
            for k in validation_statistics
        }

    return nruter


def _create_batch_validation_calculator(
    individual_calculator, validation_statistics
):
    """Vectorize an individual validation calculator over the batch axis.

    Args:
        individual_calculator: The calculator to be vectorized.
        validation_statistics: The same dictionary of ValidationStatistics
            named tuples used when building the individual calculator.

    Returns:
        A version of individual_calculator that works over batches and performs
        a reduction over that axis using the right "reduce" function.
    """
    vectorized_calculator = jax.vmap(
        individual_calculator, in_axes=[None, 0, 0, 0, 0, 0]
    )

    @jax.jit
    def nruter(
        model_params,
        positions_batch,
        types_batch,
        cell_batch,
        energies_batch,
        forces_batch,
    ):
        """Evaluate the model on a minibatch of data points.

        This function will calculate the contribution to the validation
        statistics of a batch of data points and reduce them to a single value
        per statistic. Those values will then be reduced and postprocessed to
        obtain the validation statistics for the whole model.

        Vectorizing the validation calculation over the whole data set can
        easily lead to out-of-memory problems. Combining vectorization over a
        minibatch with a loop over batches lets the user choose their own
        tradeoff.

        Args:
            model_params: The current state of the Flax model parameters.
            positions_batch: The minibatch of (natoms, 3) arrays of Cartesian
                atomic positions.
            types_batch: The minibatch of (natoms,) arrays of integer atomic
                types.
            cells_batchs: The minibatch of (3, 3) matrices of cell vectors.
            energies_batch: The minibatch of observed total energies.
            forces_batch: The minibatch of (natoms, 3) observed forces on atoms.

        Returns:
            A dictionary of results of the "map" function of each validation
            statistic, reduced over the batch with its "reduce" function.
        """
        batch_contributions = vectorized_calculator(
            model_params,
            positions_batch,
            types_batch,
            cell_batch,
            energies_batch,
            forces_batch,
        )

        return {
            k: validation_statistics[k].reduce_function(batch_contributions[k])
            for k in validation_statistics
        }

    return nruter


def create_validation_step(
    model,
    validation_statistics,
    positions,
    types,
    cells,
    energies,
    forces,
    n_batch,
    progress_bar=True,
):
    """Create a driver for a validation step.

    Args:
        model: The Flax model object to be evaluated.
        validation_statistics: A dictionary of ValidationStatistics named
            tuples.
        positions: The full batch of (natoms, 3) arrays of Cartesian atomic
            positions to be used for validation.
        types: The full batch of (natoms,) arrays of integer atomic types to be
            used for validation.
        cells: The full batch of (3, 3) matrices of cell vectors to be used for
            validation.
        energies: The full batch of observed total energies to be used for
            validation.
        forces: The full batch of (natoms, 3) observed forces on atoms to be
            used for validation.
        n_batch: The size of the minibatches over which the calculation of the
            validation statistics will be vectorized.
        progress_bar: A boolean toggle determining whether a progress bar will
            be shown on screen.

    Returns:
        A function taking the current model parameters a returning a dictionary
        with the values of the validation statistics.
    """
    n_samples = positions.shape[0]
    remainder = n_samples % n_batch
    if remainder == 0:
        remainder = n_batch

    individual_calculator = _create_individual_validation_calculator(
        model, validation_statistics
    )
    batch_calculator = _create_batch_validation_calculator(
        individual_calculator, validation_statistics
    )

    def validation_step(model_params):
        """Run a full step of the validation process.

        Args:
            model_params: The current state of the model parameters.

        Returns:
            A dictionary with the values of the validation statistics.
        """
        if progress_bar:
            bar = tqdm.auto.tqdm(total=n_samples, dynamic_ncols=True)
            bar.set_description(f"VALIDATION")
        reduced = batch_calculator(
            model_params,
            positions[:remainder, ...],
            types[:remainder],
            cells[:remainder, ...],
            energies[:remainder],
            forces[:remainder, ...],
        )
        if progress_bar:
            bar.update(remainder)
        index_from = remainder
        index_to = remainder
        while index_to < n_samples:
            index_to = min(index_to + n_batch, n_samples)
            positions_batch = positions[index_from:index_to, ...]
            types_batch = types[index_from:index_to, ...]
            cells_batch = cells[index_from:index_to, ...]
            energies_batch = energies[index_from:index_to, ...]
            forces_batch = forces[index_from:index_to, ...]
            contribution = batch_calculator(
                model_params,
                positions_batch,
                types_batch,
                cells_batch,
                energies_batch,
                forces_batch,
            )
            for k in validation_statistics:
                reduced[k] = validation_statistics[k].reduce_function(
                    jnp.array([reduced[k], contribution[k]])
                )
            if progress_bar:
                bar.update(index_to - index_from)
            index_from = index_to
        if progress_bar:
            bar.close()
        return {
            k: validation_statistics[k].postprocess_function(
                reduced[k], n_samples
            )
            for k in validation_statistics
        }

    return validation_step
