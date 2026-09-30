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
it.

#347 An estimated sigma starts at its data's spread (`stage._sigma_seeds`), not at the table's
nominal: Adam in log space moves at most its learning rate per step, and perelson's `sd_viral_load`
nominal of 1e5 for a log10 sigma kept the likelihood flat for the whole budget -- the weights saw
no gradient and the "fit" was the sigma descending at the rate cap, identical across two runs whose
nll differed by exactly `n ln(ln 10)`.

#300 A fixed-θ held-out set may bring ROWS of the training condition table but never a column, and the rows are appended at score time (`validation_data._with_held_out_conditions`) rather than shipped in the agent's `problem/conditions.tsv`. The shipped-table route is the obvious simplification and worked on main before #300; it tells the agent which experiments are held out and moves `Scenario.version`, which is what kept raimundez's EGFR knockdown (Fig 6E) filed as a transfer set in #290's first draft although the paper predicted it at the fitted point.

#256 Manuscript prose cites numbers only through `paper/generated/values.tex` macros: the
findings come from different sweeps, shas and arms, so `paper/lint_drafted.py` must reject bare
numerals rather than warn about values that cannot be regenerated from prose. The byte-identity
test excludes `provenance.tex` because its sha changes on each commit; the paired provenance test
must retain `repoSha` and `generatedOn`, with the latter derived from HEAD's UTC commit date so
regeneration does not drift with the wall clock.

#200 Anthropic caches nothing unless the request marks breakpoints, so
`driver._apply_cache_breakpoints` marks two on the CONVERTED payload — system prompt and last
markable message — for any model in `models._CACHING`. NOT the top-level `cache_control` field: on
the endpoints this account can reach that bills the 1h write premium and still returns `cached: 0`.
One endpoint tag and a 1h TTL, because each provider caches separately and the longest gap between
two calls is a fit; measured cold $2.51/M, then $0.22/M.

#242 `runner.provenance._agent_meta` spreads the driver's WHOLE return into `result.json` rather
than copying named fields, and `guards._reasoning_runaway_guard` no longer re-seeds counters the
driver already seeded. The 26-key hand copy it replaces silently dropped `openrouter_endpoints` for
the whole life of the field — so `served_drifted`, which says whether a run was one measurement at
all, was always blank — plus six guard counters, under a rule "enforced" by a regex over
`drive.py`'s source text. Deliberately NOT an allowlist filter, however documented: anything
between the driver that measures a run and the artifact that records it is a hole that opens
silently, so the test is now a real return through the real writer, compared key by key.

#167 The judge grades under the criterion the SERVER reported on the scored fit, and records
canonical BIC on every run whatever the criterion. Grading by anything else assesses a run on a
penalty it never optimised; the fixed BIC scale is what keeps a criterion sweep's arms from pooling
in `metrics_history.csv` as one configuration. Where the criterion has no value for the reference
(aicc at n <= k+1, tic against a reference with no `k_eff`) the gate falls back to canonical BIC for
both sides and says so in `criterion_note`, rather than failing the example as "could not
evaluate".

#170/#178 A run's cost is what OpenRouter billed, summed per call into `token_usage.cost_usd` at
run time; `analysis.run_cost` reports that or `None`, and there is no price table anywhere.
Do not add a table back: the hand-kept one drifted from the pinned endpoints within a week and
priced runs wrong silently, and fetching live prices at analysis time was rejected too, since it
reprices old runs at today's rates and this path is offline by design. A run with no recorded
figure is `None`, never 0.0, which would read as free in a mean.

#192 perelson does not tell the agent its pre-treatment state; it makes it preequilibrate to one.
The export's `T0`, `Tstar(0)` and `V(0)` = `Vin(0)` are to the digit one endemic steady state, and
`K0` is the paper's `c = N*k*T0` identity (footnote 10) solved for `k` — four numbers that are one
object, and right only at `c = 2.06`. Writing that state down for the agent is the modelling this
example exists to test, so `rit` gates the drug's effect on new virions and the target pool's
turnover and the solver finds it: preeq reaches `T` = 11000.0 from any start (`c - N*k*T` = -7.4e-12,
the identity DERIVED) and the treated phase reproduces Eq. 6 to 6.8e-12 with `T` held, which is the
paper's own assumption for the treated week. Eqs 1/2 with `T` pinned are linear and homogeneous, so
no gate alone can do this — the pool has to turn over untreated for the endemic state to be an
attractor rather than a ray through the origin. `NN`/`T0`/`K0` are gauge (`S = NN*Tstar` removes all
three), so none is shipped and `TRUE_ANT` carries none.

#192 The bar stays the PRISTINE published problem, which has no parameter for the pre-treatment
scale while a preequilibrated model has one, so a correct build is `k` = 4 and pays `ln(16)` = +1.5
BIC. `REFERENCE_FREE_PARAMS` cannot close this — it frees rows of the PUBLISHED table and `V0` exists
only in the patched model — and patching the training reference would redefine what a reference is
for every example. Revisit if live runs show +1.5 biting.

#192 `TRAINING_MEASUREMENTS` restates the published measurement table from one the example commits.
The collection's perelson series is a digitisation of Fig. 1B where the study's own release has the
measured values, and they disagree materially at one point. Restating happens in `_load_problem`, so
the agent's copy and every reference stage read one source; a table naming an observable or condition
the published problem lacks is refused rather than silently extending it.

#192 A transfer set may free a parameter the (patched) reference model defines even when the
published parameter table lacks it: what makes one unmanageable is having no counterpart at all, not
being absent from a table the patch is entitled to extend. `PREEQUILIBRATION` may also accompany a
transfer set — the synthesized preequilibration condition goes into the held-out condition table,
which REPLACES the training one at refit time.
