"""Sparsity penalties on the field's weights, and the proximal step that makes them bite.

A distillation is asked to write the field as rate laws over the problem's own variables, so what
matters is not how small the weights are but WHICH VARIABLES the field still depends on. That
dependency is decided in the first layer: a variable enters the network through its own columns,
and zeroing those columns removes it outright.

Two pieces, because a penalty alone does not get there. Added to the objective and optimised with
Adam, a group lasso is a subgradient method -- it shrinks a group towards zero without reaching it,
leaving every variable weakly present and nothing downstream able to act on the ranking. The
proximal operator for the group norm does reach zero::

    w_g <- w_g * max(0, 1 - strength / ||w_g||)

applied to the first layer's columns after each step. A group whose norm falls under ``strength``
becomes exactly zero and the field stops depending on that input.

Weight matrices only, never biases: a bias is a constant production term, not a dependency on an
input, so penalising it makes no dependency graph sparser.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

#: The penalties :func:`penalty` accepts. ``none`` must be asked for explicitly.
NAMES = ("none", "l1", "l2", "l21")


def variable_groups(n_columns: int, products: Sequence[tuple[int, int]] = ()) -> tuple[tuple[int, ...], ...]:
    """First-layer columns grouped by the input VARIABLE they carry.

    Without product features each column is its own variable. With them the groups OVERLAP: a
    product column ``j*k`` belongs to both ``j`` and ``k``, so dropping variable ``j`` also drops
    every product it could otherwise have survived through.
    """
    if n_columns < 1:
        raise ValueError(f"a first layer with {n_columns} columns has no variables to group")
    base = n_columns - len(products)
    groups = []
    for variable in range(base):
        columns = [variable]
        columns += [base + i for i, (a, b) in enumerate(products) if variable in (a, b)]
        groups.append(tuple(columns))
    return tuple(groups)


def l1(matrices: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Sum of absolute weights."""
    return sum(jnp.sum(jnp.abs(weight)) for weight in matrices)


def l2(matrices: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Sum of squared weights -- shrinkage, which does not sparsify."""
    return sum(jnp.sum(jnp.square(weight)) for weight in matrices)


def l21(matrices: Sequence[jnp.ndarray], groups=None) -> jnp.ndarray:
    """Group lasso over the first layer's input-variable groups."""
    first = matrices[0]
    groups = groups or variable_groups(first.shape[1])
    return sum(jnp.sqrt(jnp.sum(jnp.square(first[:, columns])) + 1e-12) for columns in groups)


def penalty(name: str, matrices: Sequence[jnp.ndarray], *, groups=None) -> jnp.ndarray:
    """The named penalty. Never defaults to ``none``: that would silently train an unregularised
    field and report it as regularised."""
    if name == "none":
        return jnp.asarray(0.0)
    if name == "l1":
        return l1(matrices)
    if name == "l2":
        return l2(matrices)
    if name == "l21":
        return l21(matrices, groups)
    raise KeyError(f"unknown penalty {name!r}; expected one of {list(NAMES)}")


def proximal_group_threshold(first: jnp.ndarray, strength: float, groups) -> jnp.ndarray:
    """The first layer with each variable group soft-thresholded, producing EXACT zeros."""
    if strength <= 0.0:
        return first
    updated = first
    for columns in groups:
        block = first[:, list(columns)]
        norm = jnp.sqrt(jnp.sum(jnp.square(block)) + 1e-12)
        updated = updated.at[:, list(columns)].set(block * jnp.maximum(0.0, 1.0 - strength / norm))
    return updated


def dependency_strength(matrices: Sequence[jnp.ndarray], *, groups=None) -> jnp.ndarray:
    """Absolute path strength from each input variable to each output equation, scaled to a max of
    1: one row per equation, one column per variable group. A zero column is a variable the field
    no longer reads."""
    chain = jnp.abs(matrices[0])
    for weight in matrices[1:]:
        chain = jnp.abs(weight) @ chain
    groups = groups or variable_groups(chain.shape[1])
    strength = jnp.stack([jnp.sum(chain[:, list(columns)], axis=1) for columns in groups], axis=1)
    return strength / jnp.maximum(jnp.max(strength), 1e-12)
