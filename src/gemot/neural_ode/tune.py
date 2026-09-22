"""What the data affords the field: parameter budgets and weight counts.

This was a tuner. Its ladders chose a latent count by climbing training nll and a width by taking
the widest field the measurements afford, and BOTH criteria were then refuted by measurement:

- A step-budget sweep over 22 fitted trajectories found held-out loss turning upward while
  training nll kept falling in 21 of them, with a rank correlation of -1.000 between final
  training nll and final held-out across perelson's nine at fixed budget. Climbing nll selects
  the worst generaliser on offer.
- Width 16 against width 7 on fujita, everything else equal: training nll -425.05 against
  -382.62, and for that, BIC with the weights charged going -65.52 to +1370.39 against the
  reference, and held-out from +202,871 to +599,282. The widest affordable field is the wrong
  end of the axis.

So the ladders are gone rather than retuned; an architecture is pinned by the caller, and where
one must be chosen, the way that works is to enumerate a small grid and take the best DISTILLED
BIC -- which clears the constant null on crauste where the ladders' own criterion does not.

What remains is arithmetic with no opinion in it: :func:`augmented_ceiling` bounds the latent
count where ``k >= n`` would saturate the likelihood, and the weight counts say how big a field
a width implies. The BIC here is the FIELD's, over its training nll; it is never a model score.
"""

from __future__ import annotations

import math


class KnobError(ValueError):
    """A malformed setting or count -- a code defect, not a failed problem."""


#: The defaults of :class:`~.train.TrainConfig` and :class:`~.distil.DistilConfig`.
N_AUGMENTED_PRIOR = 0
WIDTH_PRIOR = 32
DEPTH_PRIOR = 2
MAXSIZE_PRIOR = 20


def knob(value: float, source: str, because: str) -> dict:
    """One setting with its provenance; ``source`` is ``"prior"`` or ``"measured"``, nothing else.
    A non-finite value serialises as ``null`` with ``unbounded`` set, because JSON has no inf."""
    if source not in ("prior", "measured") or not because.strip():
        raise KnobError(f"a knob is 'prior' or 'measured' with a reason, got {source!r}")
    finite = isinstance(value, bool) or math.isfinite(value)
    return {"value": value if finite else None, "unbounded": not finite,
            "source": source, "because": because}


def _ic_slots(n_states: int, n_experiments: int, *, shared_ics: bool) -> int:
    return n_states if shared_ics else n_experiments * n_states


def augmented_ceiling(*, n_measurements: int, n_experiments: int, n_modelled_states: int,
                      n_dosed: int = 0, n_fixed_ics: int = 0, n_sigma: int = 1,
                      shared_ics: bool = False) -> dict:
    """The most latent states the parameter budget supports: the largest ``a >= 0`` with
    ``slots(n_states + a) - n_fixed_ics + n_sigma <= n_measurements``. A ceiling, not a target."""
    if min(n_measurements, n_experiments, n_modelled_states) < 1:
        raise KnobError("the latent budget needs positive measurement, experiment and state counts")
    if min(n_dosed, n_fixed_ics, n_sigma) < 0:
        raise KnobError("negative count in the latent budget")
    per_latent = _ic_slots(1, n_experiments, shared_ics=shared_ics)
    k_zero = max(0, _ic_slots(n_modelled_states + n_dosed, n_experiments, shared_ics=shared_ics)
                 - n_fixed_ics) + n_sigma
    room = n_measurements - k_zero
    allowed = max(0, room // per_latent)
    if allowed == 0:
        return knob(0.0, "measured",
                    f"no latent fits: a zero-latent model already costs k={k_zero} against "
                    f"n={n_measurements}; the lever is shared_initial_conditions, not the count")
    return knob(float(allowed), "measured",
                f"k={k_zero} at zero latents against n={n_measurements} leaves {room} parameters "
                f"of room at {per_latent} ICs per latent")


def mlp_weight_count(*, n_features: int, n_states: int, width: int, depth: int) -> int:
    """Weights and biases of ``equinox.nn.MLP``; ``depth`` counts HIDDEN layers."""
    if min(n_features, n_states, width) < 1 or depth < 1:
        raise KnobError("an MLP with a zero dimension has no weights to count")
    return (n_features * width + width) + (depth - 1) * (width * width + width) + (width * n_states + n_states)


def n_field_features(*, n_states: int, n_inputs: int) -> int:
    """First-layer inputs: the states and the field's inputs."""
    if n_states < 1 or n_inputs < 0:
        raise KnobError("a field needs at least one state")
    return n_states + n_inputs


