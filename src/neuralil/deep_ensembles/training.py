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

import operator
from collections import namedtuple
from contextlib import ExitStack

import colorama
import flax
import flax.core
import jax
import jax.nn
import jax.numpy as jnp
import jax.random
import jax.tree_util
import optax
import tqdm
from learned_optimization.optimizers.opt_to_optax import (
    GradientTransformationWithExtraArgs,
)

from neuralil.training import (
    BatchedIterator,
    emae_validation_statistic,
    ermse_validation_statistic,
    fmae_validation_statistic,
    frmse_validation_statistic,
)

# This module contains code specific to the training step of a deep ensemble
# of NeuralIL models. For details about the interface, see training.py in
# the main module.

__all__ = [
    "create_training_step",
    "create_training_epoch",
    "fmae_validation_statistic",
    "emae_validation_statistic",
    "frmse_validation_statistic",
    "ermse_validation_statistic",
    "create_validation_step",
    "unpack_params",
    "pack_params",
]


def get_n_models(model_params):
    return model_params["params"]["h_neuralil"]["energy_denormalizer"][
        "bias"
    ].shape[0]


def create_training_step(model, optimizer, calc_loss_contribution):
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
        def calc_loss(model_params):
            def invoke_loss_contribution(
                positions, types, cell, obs_energy, obs_forces
            ):
                "Compute the contribution to the loss of a single data point."
                (
                    pred_energy,
                    pred_forces,
                    sigma2_energy,
                    sigma2_forces,
                ) = model.apply(
                    model_params,
                    positions,
                    types,
                    cell,
                    method=model.calc_all_results,
                )
                pred_forces = jnp.where(
                    jnp.expand_dims(types, axis=-1) >= 0,
                    pred_forces,
                    jnp.zeros(3),
                )
                sigma2_forces = jnp.where(
                    jnp.expand_dims(types, axis=-1) >= 0,
                    sigma2_forces,
                    jnp.zeros(3),
                )
                loss = jax.vmap(
                    calc_loss_contribution,
                    in_axes=(0, 0, None, 0, 0, None, None),
                )(
                    pred_energy,
                    sigma2_energy,
                    obs_energy,
                    pred_forces,
                    sigma2_forces,
                    obs_forces,
                    types,
                )
                return loss

            return (
                jax.vmap(jax.checkpoint(invoke_loss_contribution))(
                    positions_batch,
                    types_batch,
                    cells_batch,
                    energies_batch,
                    forces_batch,
                )
                .mean(axis=0)
                .sum(axis=0)
            )

        loss_val = calc_loss(model_params)
        loss_grad = jax.jacrev(calc_loss)(model_params)

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
    def training_epoch(optimizer_state, model_params):
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
                        position=0,
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


def _create_individual_validation_calculator(model, validation_statistics):
    def nruter(model_params, positions, types, cell, energy, forces):
        (pred_energy, pred_forces, sigma2_energy, sigma2_forces) = model.apply(
            model_params, positions, types, cell, method=model.calc_all_results
        )
        pred_forces = jnp.where(
            jnp.expand_dims(types, axis=-1) >= 0,
            pred_forces,
            jnp.zeros(3),
        )
        sigma2_forces = jnp.where(
            jnp.expand_dims(types, axis=-1) >= 0,
            sigma2_forces,
            jnp.zeros(3),
        )
        return {
            k: jax.vmap(
                validation_statistics[k].map_function,
                in_axes=(0, 0, None, None, None),
            )(pred_energy, pred_forces, energy, forces, types)
            for k in validation_statistics
        }

    return nruter


def _create_batch_validation_calculator(
    individual_calculator, validation_statistics
):
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
        batch_contributions = vectorized_calculator(
            model_params,
            positions_batch,
            types_batch,
            cell_batch,
            energies_batch,
            forces_batch,
        )

        return {
            k: jax.vmap(
                validation_statistics[k].reduce_function, in_axes=1, out_axes=0
            )(batch_contributions[k])
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
                reduced[k] = jax.jit(
                    jax.vmap(
                        validation_statistics[k].reduce_function,
                        in_axes=1,
                        out_axes=0,
                    )
                )(jnp.asarray([reduced[k], contribution[k]]))
            if progress_bar:
                bar.update(index_to - index_from)
            index_from = index_to
        if progress_bar:
            bar.close()
        return {
            k: jax.vmap(
                validation_statistics[k].postprocess_function,
                in_axes=(0, None),
            )(reduced[k], n_samples)
            for k in validation_statistics
        }

    return validation_step


def unpack_params(model_params):
    "Extract a list of individual parameters sets from the ensemble parameters."
    n_models = get_n_models(model_params)
    nruter = []
    for i_model in range(n_models):
        subparams = jax.tree_map(operator.itemgetter(i_model), model_params)
        individual_params = dict(params=subparams["params"]["h_neuralil"])
        nruter.append(flax.core.freeze(individual_params))
    return nruter


def pack_params(list_of_params):
    "Compile a set of ensemble parameters from a list of individual parameters."
    merged_params = jax.tree_map(
        lambda *args: jnp.stack(list(args), axis=0), *list_of_params
    )
    nruter = dict(params=dict(h_neuralil=merged_params["params"]))
    return flax.core.freeze(nruter)
