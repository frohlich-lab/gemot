# Rationale Log

#179 `identifiability.effective_parameter_count` does not take the `hess` the objective reports,
because AMICI's `ReturnData::fFIM` is `sresᵀ·sres` alone: its own source comment (`rdata.cpp`)
derives the exact expected Hessian and then drops the `(∂σ∂σᵀ)(2r² − σ²)/σ⁴` term, since that
term can be indefinite and Fides converged better without it on the Boehm2014 benchmark. Sound
for an optimizer, wrong for a statistic that inverts the matrix: it leaves an ESTIMATED sigma's
curvature at half its Fisher information (on the toy, σσ 31.8 against 63.6 by finite differences)
and inflated `k_eff` from 3.3 to 4.0 at k=3. The ρ²-weighted sigma rows restore exactly the
dropped term's expected value, which is always positive, so the matrix built here is AMICI's FIM
plus the piece it documents leaving out — and identical to it wherever sigma is fixed or
inner-solved. Do not simplify it back to `ret[HESS]`.

#179 `identifiability._residual_rows` cannot simply call a hierarchical objective in residual
mode: pyPESTO 0.6 does not implement that path. `calculate_directly` leaves `res`/`sres` at their
empty initial values and raises `TypeError: 'NoneType' object is not subscriptable` on the
gradient, and the path that does fill them is reached only when a Hessian is requested — which
`ObjectiveBase` refuses in residual mode. Without the detour the count was `None` for every fit
with an inner-solved sigma, i.e. most real examples. The hand-driven scores were checked against
the profiled objective's own gradient and agree to 3e-7 relative, which is the envelope theorem:
at the inner optimum the profiled gradient IS the partial one.

#179 `k_eff` is deliberately absent from every result the agent reads — `goodness_of_fit`'s
output, `fit_result_block`, `get_fit_summary`, the version meta `list_versions` returns, and
every tool docstring. A number on every fit result is a number the agent starts optimising, and
nobody asked for that. Adding it back to the score table for symmetry with `bic`/`aic` is the
change to refuse.

#179 `None` rather than `k` for an unevaluable count, because the two mean opposite things: a `k`
standing in for a missing measurement reads as "the data determine every parameter", the one
conclusion the statistic exists to be able to contradict. Inner-solved parameters are charged
rather than dropped because a profiled parameter the data pin is exactly the case where the count
and the trace agree. Measured cost: 0.085 s on blasi's k=11, n=252.

#169 Added the RATIONALE.md file as a location to record non-obvious implementation
decisions and reduce the volume of inline comments in the source code written by coding
agents.
