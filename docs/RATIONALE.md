# Rationale Log

**Adding an entry is the exception, not the routine.** Most PRs end with nothing written here, and
that is the correct outcome. An entry earns its place only when a reader would otherwise UNDO the
choice — the obvious simplification is wrong, the tried alternative failed, the number was
measured. Design detail that the code, its docstrings or the commit message already carries is not
a rationale; it is noise that buries the entries that matter.

**Does it belong to one module that still exists? Then it is not a log entry.** Put it in that
module's docstring, next to the code a reader would undo. The log is for what spans modules, what
is deliberately ABSENT (a criterion not in the table, a field no result carries, a price table
there is no code for), and what outlives the code it explains — an entry about a function deleted
last week is doing its job.

**Budget: at most one new entry per PR, two or three sentences.** A PR that seems to need two
entries needs one, about the choice a reader is likeliest to undo. Name the failure, the measured
cost or the PR number, and stop; do not narrate the investigation, restate what the code does, or
enumerate every case you checked. Prefer editing or deleting a stale entry over appending beside
it. `#167`, `#179` and `#245` predate this budget and still carry two or three entries each; they
were left split because separate code sites cite a different one of them by name, so they are a
legacy exception and not a precedent.

#89 `validation.stoichiometry_conflict_issues` and its `set_model` caller are gone: antimony's 3.2
floor makes the id-uniqueness collision they translated — `N A -> N B` failing conversion outright
— unreachable. The STATIC `frozen_parametric_stoichiometry_issues` stays, because the footgun is
NOT fixed; on 3.2.0 the collision merely went quiet, emitting a frozen `stoichiometry` plus an
assignment rule reading an undeclared symbol, which libSBML passes with warnings and no error. A
floor rather than the rejected cap, since an older antimony would restore an exception nothing here
translates any more.

#245 `identifiability._compute_fim` and `effective_parameter_count` take the FIM and the scores
from the residual rows of one call, not from the objective's half-curvature `hess` — a deliberate
change to agent-facing statistics, not a refactor. With sigmas FIXED the two agree to 3.6e-17
relative, so no shipped example moves; an ESTIMATED sigma's own diagonal doubles, so `se_scaled`
falls by sqrt(2) and a sigma-sigma merge tie doubles (predator_prey 1.026 -> 2.093 nll). Building
the rows toggles AMICI's `add_sigma_residuals` by hand on the engine's ONE cached objective, so it
happens under `session.lock` and is restored in a `finally`; the remaining hazard is a
`get_fit_diagnosis` racing a simulate on another thread, which doubles that simulation's
residuals.

#245 The three eigenvalue floors this matrix feeds — `pinv` in `_identifiability_stats`,
truncation in `effective_parameter_count`, the CLAMP in `merge_candidates` — are deliberately NOT
unified: truncating in `merge_candidates` would price the one FREE merge as the report's most
expensive. Unifying the FIM is not a licence to unify its inverses.

#200 Anthropic caches nothing unless the request marks breakpoints, so
`driver._apply_cache_breakpoints` marks two on the CONVERTED payload — system prompt and last
markable message — for any model in `models._CACHING`. NOT the top-level `cache_control` field: on
the endpoints this account can reach that bills the 1h write premium and still returns `cached: 0`.
One endpoint tag and a 1h TTL, because each provider caches separately and the longest gap between
two calls is a fit; measured cold $2.51/M, then $0.22/M.

#243 A `make_changes` result carries no flat `warnings` list — the payload diet cut it at the tool
boundary (17% of payload bytes, duplicated on 62.8% of 915 payloads) while `bundle_result` went on
computing it. Do not reintroduce it; filter `messages` by level. The operations-layer `warnings` is
a DIFFERENT field and stays.

#242 `runner.provenance._agent_meta` spreads the driver's WHOLE return into `result.json` rather
than copying named fields, and `guards._reasoning_runaway_guard` no longer re-seeds counters the
driver already seeded. The 26-key hand copy it replaces silently dropped `openrouter_endpoints` for
the whole life of the field — so `served_drifted`, which says whether a run was one measurement at
all, was always blank — plus six guard counters, under a rule "enforced" by a regex over
`drive.py`'s source text. Deliberately NOT an allowlist filter, however documented: anything
between the driver that measures a run and the artifact that records it is a hole that opens
silently, so the test is now a real return through the real writer, compared key by key.

#167 No scalar weight on the complexity penalty in `petab_mcp.criteria` — named penalties and
nothing between them. One was built and removed: every effect-size threshold in the loop
(`NEW_BEST_MIN_DELTA_SCORE`, `PLATEAU_MIN_SCORE_GAIN`, `dedup.WORSE_SCORE_MARGIN`, the benchmark
bar) is a fixed count of deviance units calibrated on BIC, which survives SWAPPING the penalty but
not SCALING it. Deriving those thresholds from the criterion rather than writing them as literals
in `constants.py` is the precondition for reintroducing the knob.

#167 Criteria left out of the table on purpose. ICOMP is the one measure that charges for
parameter INTERDEPENDENCE, this project's actual failure mode, but it is zero for any number of
orthogonal parameters and infinite on a rank-deficient FIM, so no fixed effect-size threshold means
anything under it — it belongs beside `k_eff` as a diagnostic. Anything integrating over parameter
space is excluded by construction (a fit here is one optimum, not a posterior): marginal
likelihood, DIC, WAIC, PSIS-LOO, stochastic complexity; FPE / Mallows' Cp / GCV would rank nothing
differently, and EBIC needs a countable model space that a freely authored network is not.

#167 The judge grades under the criterion the SERVER reported on the scored fit, and records
canonical BIC on every run whatever the criterion. Grading by anything else assesses a run on a
penalty it never optimised; the fixed BIC scale is what keeps a criterion sweep's arms from pooling
in `metrics_history.csv` as one configuration. Where the criterion has no value for the reference
(aicc at n <= k+1, tic against a reference with no `k_eff`) the gate falls back to canonical BIC for
both sides and says so in `criterion_note`, rather than failing the example as "could not
evaluate".

#179 `identifiability.effective_parameter_count` does not take the objective's `hess`, because
AMICI's `fFIM` is `sresᵀ·sres` alone: `rdata.cpp` derives the exact expected Hessian and then drops
the `(∂σ∂σᵀ)(2r² − σ²)/σ⁴` term, which is sound for an optimizer and wrong for a statistic that
inverts the matrix — an ESTIMATED sigma keeps half its Fisher information (toy σσ 31.8 against 63.6
by finite differences), inflating `k_eff` from 3.3 to 4.0 at k=3. The ρ²-weighted sigma rows restore
exactly that term's expected value, so this matrix is AMICI's FIM plus the piece it documents
leaving out, and identical to it wherever sigma is fixed or inner-solved. Do not simplify it back to
`ret[HESS]`.

#179 `k_eff` is deliberately absent from every result the agent reads — `goodness_of_fit`,
`fit_result_block`, `get_fit_summary`, `list_versions` meta, every tool docstring. A number on
every fit result is a number the agent starts optimising; adding it to the score table for symmetry
with `bic`/`aic` is the change to refuse.

#170/#178 A run's cost is what OpenRouter billed, summed per call into `token_usage.cost_usd` at
run time; `analysis.run_cost` reports that or `None`, and there is no price table anywhere.
Do not add a table back: the hand-kept one drifted from the pinned endpoints within a week and
priced runs wrong silently, and fetching live prices at analysis time was rejected too, since it
reprices old runs at today's rates and this path is offline by design. A run with no recorded
figure is `None`, never 0.0, which would read as free in a mean.
