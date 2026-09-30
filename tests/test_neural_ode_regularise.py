"""The sparsity penalties and the proximal step, on hand-computed cases.

The module's claim is that a penalty ALONE does not sparsify -- added to the objective and
optimised with Adam it is a subgradient method that shrinks a group without reaching zero -- and
that the proximal operator does. These check exactly that, since an unreached zero leaves every
variable weakly present and nothing downstream can act on the ranking.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from gemot.neural_ode import regularise as R


def test_each_column_is_its_own_variable_until_products_overlap_them():
    assert R.variable_groups(3) == ((0,), (1,), (2,))
    # 3 base variables then the products (0,1) and (1,1) in columns 3 and 4. A product column
    # belongs to BOTH its factors, so dropping a variable drops every product carrying it.
    groups = R.variable_groups(5, products=[(0, 1), (1, 1)])
    assert groups == ((0, 3), (1, 3, 4), (2,))
    assert set(groups[0]) & set(groups[1]) == {3}, "the shared product column must be in both"
    with pytest.raises(ValueError, match="no variables to group"):
        R.variable_groups(0)


def test_the_penalty_is_named_and_never_defaults():
    w = [jnp.array([[3.0, -4.0]]), jnp.array([[1.0], [2.0]])]
    assert float(R.penalty("l1", w)) == pytest.approx(3 + 4 + 1 + 2)
    assert float(R.penalty("l2", w)) == pytest.approx(9 + 16 + 1 + 4)
    assert float(R.penalty("none", w)) == 0.0
    # l21 reads the FIRST layer only, one group per column here: |3| and |-4|.
    assert float(R.penalty("l21", w)) == pytest.approx(3 + 4, abs=1e-5)
    # Silently treating an unknown name as `none` would train an unregularised field and report
    # it as regularised, which is the failure this guards.
    with pytest.raises(KeyError, match="unknown penalty"):
        R.penalty("lasso", w)


def test_the_proximal_step_reaches_exact_zero_where_shrinkage_would_not():
    # Column 0 has norm 0.3, column 1 norm 5. At strength 0.5 the first group is annihilated and
    # the second is shrunk but survives -- that is the point of the operator over the penalty.
    first = jnp.array([[0.3, 3.0], [0.0, 4.0]])
    groups = R.variable_groups(2)
    out = R.proximal_group_threshold(first, 0.5, groups)
    assert jnp.all(out[:, 0] == 0.0), "a group under the threshold must be EXACTLY zero"
    assert float(jnp.linalg.norm(out[:, 1])) == pytest.approx(5 - 0.5, abs=1e-4)
    assert float(jnp.linalg.norm(out[:, 1])) < float(jnp.linalg.norm(first[:, 1]))
    # Zero strength is a no-op, so an unpenalised run is not silently thresholded.
    assert jnp.array_equal(R.proximal_group_threshold(first, 0.0, groups), first)


def test_dependency_strength_zeroes_the_column_of_a_variable_the_field_dropped():
    # Two inputs, one hidden unit each, two outputs. Input 1's column is already zero.
    first = jnp.array([[2.0, 0.0], [1.0, 0.0]])
    second = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    strength = R.dependency_strength([first, second])
    assert strength.shape == (2, 2)
    assert jnp.all(strength[:, 1] == 0.0), "a dropped variable reads as no dependency"
    assert float(jnp.max(strength)) == pytest.approx(1.0), "scaled to a max of 1"
    assert float(strength[0, 0]) > 0.0 and float(strength[1, 0]) > 0.0
