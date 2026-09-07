# Rationale Log

#242 `runner.provenance._agent_meta` spreads the agent driver's WHOLE return into `result.json`
rather than copying the fields it wants. The 26-key hand copy in `runner/drive.py` that it
replaces never named `openrouter_endpoints`, so `metrics._flatten_served_endpoints` read `None`
on every run ever recorded and `served_provider` / `served_n_distinct` / `served_drifted` stayed
blank for the whole life of the field — including the one column that says whether a run was
split across two upstream endpoints, i.e. whether it is one measurement at all. Six guard
counters (`tools_executed`, `dropped_dup_tool_calls`, `idle_nudges`, `gave_up_no_tools`,
`task_arg_rejections`, `unreadable_graph_state`) had already been lost the same way, and the rule
was "enforced" by a regex over `drive.py`'s source text, which any occurrence of the name — a
comment included — satisfies. Deliberately NOT a filter with an allowlist, however documented:
the driver measures a run and the artifact records it, so anything between the two is a hole that
opens silently. The test is now the artifact itself (`tests/test_run_loop_no_model.py`): a real
`run_deepagents` return, written by the real writer, compared key by key.

#242 `guards._reasoning_runaway_guard` no longer re-seeds its three counters with `setdefault`.
It duplicated `new_guard_stats`'s only job and made a partly-seeded stats dict look healthy,
which is how the seeding contract went unexamined while six of its keys were being dropped
downstream. The driver seeds once, before any guard is built; a caller that hands the guards a
bare `{}` now fails on the first increment instead of writing an artifact with three counters in
it and thirteen missing.

#167 No scalar weight on the complexity penalty in `petab_mcp.criteria` — named penalties and
nothing between them. One was built and then removed: every effect-size threshold in the loop
(`NEW_BEST_MIN_DELTA_SCORE`, `PLATEAU_MIN_SCORE_GAIN`, `dedup.WORSE_SCORE_MARGIN`, the benchmark's
own bar) is a fixed number of deviance units calibrated on BIC. Those carry over when the penalty
is SWAPPED, because every criterion here is on the same 2·Δnll scale, but not when it is SCALED:
at half weight a 2-unit "positive evidence" bar swamps the penalty under study, at double weight
it is noise beside it, so re-weighting silently changes what every gate means. Deriving those
thresholds from the criterion instead of writing them as literals in `constants.py` is the
precondition for reintroducing the knob.

#167 Criteria considered for `petab_mcp.criteria` and deliberately left out. ICOMP (Bozdogan's
information complexity) is pointwise off the same Fisher matrix and is the one measure that
charges for parameter INTERDEPENDENCE — this project's actual failure mode — but it is not a
per-parameter deviance penalty: zero for any number of orthogonal parameters and infinite on a
rank-deficient FIM, so no fixed effect-size threshold means anything under it. It belongs beside
`k_eff` as a diagnostic, not in the table. Anything integrating over parameter space is excluded
by construction, since a fit here is one optimum and not a posterior: marginal likelihood and
Bayes factors, DIC, WAIC, PSIS-LOO, Rissanen's stochastic complexity. FPE, Mallows' Cp and
GCV/PRESS are Gaussian rewrites of AIC/TIC on the same leverages and would rank nothing
differently. EBIC needs a countable model space, which a freely authored reaction network is not.

#167 The criterion is session state (`PetabSession.criterion`, resolved once by
`criteria.resolve`), not a module-level global. Review feedback on the PR. The first version kept
it in `criteria._active` behind `configure()`, set at import from the environment and again at
server construction: two servers in one process then shared it, so one caller's
`criterion="aicc"` re-scored every server built after it, and because the default call passes
`None` there was nothing to observe the leak by. Threading the object to each scorer costs a
parameter and removes the failure mode outright.

#167 `Criterion.parameter_price` returns `None` under TIC rather than AIC's 1. What deleting a
parameter takes off `k_eff` is only knowable after the refit, and for the pair a merge hint
targets — two parameters the data cannot separate — it is about zero, because TIC never charged
for that parameter in the first place. Pricing the merge at 1 would promise a saving that is not
there.

#167 The judge grades under the criterion the SERVER reported on the scored fit, while recording
canonical BIC for every run whatever the criterion. Grading by anything else would assess a run
on a penalty it never optimised; reading the label off the fit rather than re-resolving the
environment keeps the judge from disagreeing with the process that did the ranking. Canonical BIC
stays on every result so a criterion sweep's arms remain comparable on one fixed scale — without
it the arms pool in `metrics_history.csv` as if they were one configuration, the failure
`example_version`/`git_sha` already exist for. Where the criterion has no value for the reference
(aicc at n <= k+1, tic against a reference written without a `k_eff`) the gate falls back to
canonical BIC and says so in `criterion_note`, rather than failing every run on that example as
"could not evaluate".


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
and the trace agree — where TIC and AIC coincide. Measured cost: 0.085 s on blasi's k=11, n=252.

#170 `petab_harness.analysis` has no price table, deliberately. `run_cost` returns the cost a
run recorded (`token_usage.cost_usd`) and `None` otherwise. The hand-kept OpenRouter table it
replaced carried no date and was suspected stale within a week of being consolidated; a table
that prices runs wrong silently is worse than a `-`. Fetching current prices at analysis time
was rejected too: it reprices old runs at today's rates, and the analysis path is offline by
design. The fix is to record the billed figure at run time (#178). Do not add a price table back.

#169 Added the RATIONALE.md file as a location to record non-obvious implementation
decisions and reduce the volume of inline comments in the source code written by coding
agents.
