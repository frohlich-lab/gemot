"""Portable data, partial correlations and rendering for manuscript Figure 3.

Consumes the compact, archived Fujita export; never loads or refits agent models.
See paper/fujita_generalisation.md for feature definitions and source limitations.
"""

from __future__ import annotations

from collections import deque
import csv
import hashlib
import io
import json
import math

CORE_FEATURES = ("training_nll", "n_parameters", "n_reactions", "n_states")
#: `n_parameters` is the count BIC and AIC actually charge: the estimated parameters PLUS the
#: literals the model hardcoded, which the criterion prices rather than ignores. It is an
#: ADJUSTMENT here, not a bar: the panel asks about the criteria themselves rather than about a
#: parameter count standing in for them.
#:
#: The criteria are exact functions of the two things already in the core set —
#: ``BIC = 2 NLL + k ln n`` and ``AIC = 2 NLL + 2k`` — so a control set holding the training NLL
#: fixed would hand back `k` in disguise, and holding one criterion fixed while asking about the
#: other is rank-deficient outright (`BIC = AIC + k(ln n - 2)`). Each is therefore adjusted for
#: the STRUCTURAL features only, which is the question a reader actually has: does the criterion
#: rank generalisation beyond what the raw size of the model already says?
CRITERION_FEATURES = ("training_bic", "training_aic")
CRITERION_CONTROLS = ("n_reactions", "n_states")
#: `latent_depth` (pEGFR -> pS6) was dropped: it is not a reference connection, it is not a bar in
#: D, and as a skip path it only ever restated what the two adjacent links already say.
DEPTH_FEATURES = (
    "input_depth",
    "upstream_depth",
    "downstream_depth",
    "input_to_akt_depth",
    "full_depth",
)
FEATURES = (
    "training_nll",
    "training_bic",
    "training_aic",
    "n_reactions",
    "n_states",
    *DEPTH_FEATURES[:3],
)
LABELS = (
    "Training\nNLL",
    "Training\nBIC",
    "Training\nAIC",
    "Reactions",
    "States",
    "EGF\n→ pEGFR",
    "pEGFR\n→ pAkt",
    "pAkt\n→ pS6",
)
#: The adjacent readout links are still estimated, exported and put through the leave-one-out
#: diagnostic; they are simply not drawn. All three sit within 0.5 of zero except one that a single
#: model flips, so the panel spent a third of its width on an association it cannot support.
PLOTTED_FEATURES = FEATURES[:5]
PLOTTED_LABELS = LABELS[:5]
CONNECTIONS = (
    ("input_depth", "EGF", "pEGFR"),
    ("upstream_depth", "pEGFR", "pAkt"),
    ("downstream_depth", "pAkt", "pS6"),
    ("input_to_akt_depth", "EGF", "pAkt"),
    ("full_depth", "EGF", "pS6"),
)
#: How the models are grouped and drawn is the INPUT's business, not this module's: the figure
#: covers however many experiments the manifest carries, and hard-coding three arms here is what
#: stopped it covering more. ``slurm/_fujita_figure_input.py`` owns the spec.


def arms(metadata):
    """``{arm: drawing spec}`` in the manifest's own order."""
    return metadata["arms"]


def pools(metadata):
    """The correlation pools: ONE PER LANGUAGE MODEL, over that model's rungs of the ablation.

    The ablation is a ladder within a model, and a coefficient pooled across both models averages
    two different agents into one number — the arms differ in what the harness gave them, but the
    two families differ in what the agent could do with it. Within a pool the models containing
    the reference connections are pooled regardless of which rung produced them, and the pool is
    not additionally adjusted for arm membership (stated in the caption).
    """
    specs = arms(metadata)
    found = {}
    for index, model in enumerate(factor_levels(metadata, "model")):
        here = tuple(arm for arm, spec in specs.items() if spec["factors"]["model"] == model)
        code = specs[here[0]]["code"][0]
        found[model] = {
            "arms": here,
            "code": code,
            "label": f"{code}  {short_model(model)}",
            "color": SUPPORT_RAMPS[index % len(SUPPORT_RAMPS)][1],
        }
    return found


def controls_for(feature):
    """The covariates a feature's partial coefficient adjusts for."""
    if feature in CRITERION_FEATURES:
        return list(CRITERION_CONTROLS)
    if feature in CORE_FEATURES:
        return [key for key in CORE_FEATURES if key != feature]
    return list(CORE_FEATURES)


def forward_readout_paths(graph, anchors):
    """Collapse hidden intermediates, while forbidding paths through another readout.

    Unlike unrestricted reachability, this distinguishes a serial chain from an input
    that independently drives several readouts. Edge length may differ from reference.
    """
    paths = {}
    for key, source, target in CONNECTIONS:
        starts, targets = anchors[source], anchors[target]
        blocked = (
            set().union(*(nodes for name, nodes in anchors.items() if name not in (source, target)))
            - starts
            - targets
        )
        queue = deque((node, [node]) for node in sorted(starts))
        visited = set(starts) | blocked
        paths[key] = None
        while queue:
            node, path = queue.popleft()
            if node in targets:
                paths[key] = path
                break
            for nxt in sorted(graph.get(node, ())):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
    return paths


def read_input(root):
    """Read verified portable observations, including explicit missing scores/paths."""
    path = root / "paper/fujita_generalisation.csv"
    provenance = path.with_suffix(".json")
    if not path.exists() and not provenance.exists():
        return None
    metadata = json.loads(provenance.read_text())
    if metadata.get("schema_version") != 4:
        raise ValueError(
            "unsupported Fujita figure schema; rebuild the input with slurm/_fujita_figure_input.py"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["source"]["data_sha256"]:
        raise ValueError("Fujita observations do not match their provenance hash")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference_fit = {run["task"]: run["training_reference"] for run in metadata["runs"]}
    counts = (
        "head_version",
        "conditions_ok",
        "n_parameters",
        "n_estimated",
        "n_charged_literals",
        "n_reactions",
        "n_states",
        "n_species",
        *DEPTH_FEATURES,
    )
    scores = ("training_dbic", "training_nll", "heldout_nll", "heldout_delta_nll")
    for row in rows:
        for key in (*counts, *scores):
            row[key] = (int(row[key]) if key in counts else float(row[key])) if row[key] else None
        if row["complete_heldout"] not in ("True", "False"):
            raise ValueError(f"{row['task']}: invalid completion flag")
        row["complete_heldout"] = row["complete_heldout"] == "True"
        complete = row["conditions_ok"] == metadata["evaluation"]["n_conditions"]
        if complete != row["complete_heldout"]:
            raise ValueError(f"{row['task']}: incomplete condition coverage")
        heldout = row["heldout_nll"]
        delta = row["heldout_delta_nll"]
        if complete:
            if (
                heldout is None
                or not math.isfinite(heldout)
                or delta is None
                or not math.isclose(
                    delta, heldout - metadata["reference"]["heldout_nll"], rel_tol=0, abs_tol=1e-9
                )
            ):
                raise ValueError(f"{row['task']}: invalid held-out delta")
        elif heldout is not None or delta is not None:
            raise ValueError(f"{row['task']}: incomplete integration must not have a score")
        if row["n_parameters"] != row["n_estimated"] + row["n_charged_literals"]:
            raise ValueError(f"{row['task']}: parameter count disagrees with scoring k")
        # The criteria themselves, rebuilt from the same NLL and charged k the ΔBIC axis used, on
        # the TRAINING problem's measurement count. Recomputing rather than reading them back is
        # what lets the next line PROVE the two panels are pricing the same model: if this BIC
        # minus the reference's is not the recorded ΔBIC, one of the two is wrong.
        fit = reference_fit[row["task"]]
        row["training_bic"] = (
            None
            if row["training_nll"] is None
            else 2 * row["training_nll"] + row["n_parameters"] * math.log(fit["n"])
        )
        row["training_aic"] = (
            None
            if row["training_nll"] is None
            else 2 * row["training_nll"] + 2 * row["n_parameters"]
        )
        if row["training_bic"] is not None:
            reference_bic = 2 * fit["nll"] + fit["k"] * math.log(fit["n"])
            recorded = row["training_dbic"] - float(row["training_dbic_adjustment"])
            if not math.isclose(
                recorded, row["training_bic"] - reference_bic, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError(f"{row['task']}: rebuilt BIC disagrees with the recorded ΔBIC")
        row["undefined_depths"] = ",".join(key for key in DEPTH_FEATURES if row[key] is None)
    tasks = [row["task"] for row in rows]
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) != {run["task"] for run in metadata["runs"]}
    ):
        raise ValueError("Fujita task IDs must uniquely match the provenance manifest")
    if {row["arm"] for row in rows} != set(arms(metadata)):
        raise ValueError(
            "every Fujita arm in the manifest must have models, and every model an arm"
        )
    labels = [row["label"] for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("Fujita model labels must be unique across experiments")
    for row in rows:
        expected = arms(metadata)[row["arm"]]["prefix"]
        if not row["label"].startswith(expected):
            raise ValueError(
                f"{row['task']}: label {row['label']} does not carry its arm's prefix {expected}"
            )
    read_topology(root, metadata, rows)
    read_validation_deltas(root, metadata)
    read_costs(metadata, rows)
    return metadata, rows


def read_costs(metadata, rows):
    """Both bills for every run, including the ones that delivered no model.

    A run that failed still bought tokens and still held its allocation, so leaving it out would
    understate exactly the arms that fail most. The CPU half is an extrapolation onto rented
    hardware — the cluster charges nothing — and the price it uses is recorded in the manifest.
    """
    price = metadata["cost_model"]["cpu"]
    per_core_hour = price["usd_per_hour"] / price["vcpu"]
    delivered = {row["task"] for row in rows}
    arm = {row["task"]: row["arm"] for row in rows}
    arm |= {entry["task"]: entry["arm"] for entry in metadata["no_model_runs"]}
    costs = []
    for entry in metadata["runs"] + metadata["no_model_runs"]:
        cost = entry["cost"]
        task = cost["task"]
        if task not in arm:
            raise ValueError(f"{task}: a cost record with no arm")
        if cost.get("cpu_seconds") is None or not cost["elapsed_seconds"]:
            raise ValueError(f"{task}: no recorded CPU time to price")
        if cost["llm_usd"] is None:
            raise ValueError(f"{task}: no recorded LLM spend")
        costs.append(
            {
                "task": task,
                "arm": arm[task],
                "llm_usd": cost["llm_usd"],
                # The core-seconds actually burned, not the reservation: most of the
                # allocation sat idle waiting on the language model, and charging that
                # would price the wait rather than the work.
                "cpu_usd": cost["cpu_seconds"] / 3600 * per_core_hour,
                "reserved_usd": cost["alloc_cpus"] * cost["elapsed_seconds"] / 3600 * per_core_hour,
                "delivered_model": task in delivered,
            }
        )
    seen = {cost["task"] for cost in costs}
    if seen != set(arm):
        raise ValueError("every run needs exactly one cost record")
    metadata["costs"] = costs
    return costs


def cost_totals(metadata, arm):
    """``(LLM, CPU, models, total)`` for one arm: what it spent, per MODEL it delivered.

    Per model rather than per run, because a run that delivered nothing still bought tokens and
    still held its allocation: that spend is part of what the arm's surviving models cost.
    """
    sample = [cost for cost in metadata["costs"] if cost["arm"] == arm]
    delivered = sum(cost["delivered_model"] for cost in sample) or len(sample)
    llm = sum(cost["llm_usd"] for cost in sample) / delivered
    cpu = sum(cost["cpu_usd"] for cost in sample) / delivered
    return llm, cpu, delivered, (llm + cpu) * delivered


def read_validation_deltas(root, metadata):
    """Panel F's input: held-out ΔNLL per run on the OTHER validation examples.

    Attached to the metadata rather than returned, so every consumer of ``read_input`` sees the
    same object. Verified against its hash like the other two tables.
    """
    path = root / "paper/fujita_validation_deltas.csv"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != metadata["source"]["validation_deltas_sha256"]
    ):
        raise ValueError("Fujita validation deltas do not match their provenance hash")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    declared = {entry["example"]: entry for entry in metadata["validation_examples"]}
    for row in rows:
        if row["example"] not in declared:
            raise ValueError(f"{row['example']}: not declared in the manifest")
        row["delta_nll"] = float(row["delta_nll"])
        row["heldout_nll"] = float(row["heldout_nll"])
        row["baseline_nll"] = float(row["baseline_nll"])
        row["delta_is_lower_bound"] = row["delta_is_lower_bound"] == "True"
    for example, entry in declared.items():
        found = sum(1 for row in rows if row["example"] == example)
        if found != entry["n_runs"]:
            raise ValueError(f"{example}: {found} runs in the table, {entry['n_runs']} declared")
    metadata["validation_deltas"] = rows
    return rows


def read_topology(root, metadata, rows):
    """Join the archived path audit and classify the forward readout graph."""
    path = root / "paper/fujita_topology.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["source"]["topology_data_sha256"]:
        raise ValueError("Fujita topology paths do not match their provenance hash")
    with path.open(newline="") as handle:
        audited = list(csv.DictReader(handle))
    lookup = {row["task"]: row for row in audited}
    if len(lookup) != len(audited) or set(lookup) != {row["task"] for row in rows}:
        raise ValueError("Fujita topology audit must cover each selected model once")
    expected = {
        key for key, path in metadata["topology_definition"]["reference_paths"].items() if path
    }
    for row in rows:
        paths = {key: lookup[row["task"]][key] for key, _, _ in CONNECTIONS}
        present = {key for key, path in paths.items() if path}
        anchors = {
            "EGF": {row["stimulus_anchor"]},
            "pEGFR": set(row["receptor_states"].split(",")),
            "pAkt": set(row["akt_states"].split(",")),
            "pS6": set(row["s6_states"].split(",")),
        }
        overlapping = any(anchors[source] & anchors[target] for _, source, target in CONNECTIONS)
        for key, source, target in CONNECTIONS:
            if not paths[key]:
                continue
            chain = paths[key].split(" -> ")
            other_readouts = set().union(
                *(nodes for name, nodes in anchors.items() if name not in (source, target))
            )
            if (
                chain[0] not in anchors[source]
                or chain[-1] not in anchors[target]
                or set(chain[1:-1]) & other_readouts
            ):
                raise ValueError(f"{row['task']}: invalid collapsed readout path {key}")
        row["topology_paths"] = paths
        row["contains_reference_topology"] = expected <= present and not overlapping
        row["exact_reference_topology"] = present == expected and not overlapping
        row["additional_forward_connections"] = ",".join(sorted(present - expected))
        row["topology_reason"] = "; ".join(
            filter(
                None,
                (
                    "overlapping readout anchors" if overlapping else "",
                    "missing " + ",".join(sorted(expected - present)) if expected - present else "",
                ),
            )
        )


def topology_counts(rows):
    """Every selected model counts, independently of held-out integration success.

    Arms come from the rows in the order the input lists them, which is the manifest's own arm
    order — so a figure covering more experiments needs no change here.
    """
    output = []
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        matched = [row for row in selected if row["contains_reference_topology"]]
        output.append(
            {
                "arm": arm,
                "n_models": len(selected),
                "n_retained": len(matched),
                "retained_tasks": ",".join(row["task"] for row in matched),
                "excluded_tasks": ",".join(
                    row["task"] for row in selected if not row["contains_reference_topology"]
                ),
            }
        )
    return output


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def symlog_transform(values, config):
    """Matplotlib's symmetric-log coordinates, without importing its plotting stack.

    The renderer checks this against the actual axis transform for every plotted point.
    Keeping this here lets the text/data export run without Matplotlib, as before.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    threshold = config["linthresh"]
    linear = config["linscale"] / (1 - 1.0 / config["base"])
    log_base = np.log(config["base"])
    absolute = np.abs(values)
    outside = absolute > threshold
    transformed = values * linear
    transformed[outside] = (
        np.sign(values[outside])
        * threshold
        * (linear - np.log(threshold) / log_base + np.log(absolute[outside]) / log_base)
    )
    return transformed


ORDINALS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def macro_name(name):
    """``O1c`` -> ``OOneC``. LaTeX command names cannot contain digits, so every digit becomes a
    word; the rest of the name is carried through in capitals."""
    spelled = ""
    for char in name:
        if char.isdigit() and 1 <= int(char) <= len(ORDINALS):
            spelled += ORDINALS[int(char) - 1]
        elif char.isalpha():
            spelled += char.upper()
        else:
            raise ValueError(f"unsupported Fujita label {name}")
    return spelled


def macros(metadata, rows):
    """Caption values share the plotted input, rather than being transcribed into LaTeX."""
    evaluation = metadata["evaluation"]
    scored = [row for row in rows if row["complete_heldout"]]
    best, runner_up = sorted(scored, key=lambda row: row["heldout_nll"])[:2]
    # The supplementary maps are named in the export; they must be the two the scores rank first.
    for key, row in (("best_generalising", best), ("runner_up", runner_up)):
        drawn = metadata["process_diagrams"][key]["source"]["name"]
        if drawn != run_name(metadata, row["label"]):
            raise ValueError(
                f"{key} map shows {drawn}, but the scores rank "
                f"{run_name(metadata, row['label'])} there"
            )
    values = {
        "fujitaFigureRunCount": str(len(rows)),
        "fujitaFigureArmCount": str(len(arms(metadata))),
        "fujitaFigureExperimentCount": str(len(metadata["source"]["experiments"])),
        "fujitaFigureNoModelCount": str(len(metadata["no_model_runs"])),
        "fujitaFigureLaunchedRunCount": str(len(rows) + len(metadata["no_model_runs"])),
        "fujitaFigureTotalUSD": f"{sum(cost['llm_usd'] + cost['cpu_usd'] for cost in metadata['costs']):,.0f}",
        "fujitaFigureCostInstance": metadata["cost_model"]["cpu"]["instance"],
        "fujitaFigureCostRate": f"{metadata['cost_model']['cpu']['usd_per_hour']:.4g}",
        "fujitaFigureCostCores": str(metadata["cost_model"]["cpu"]["vcpu"]),
        "fujitaFigureCostRegion": metadata["cost_model"]["cpu"]["region"],
        "fujitaFigureCostQuoted": metadata["cost_model"]["cpu"]["quoted"],
        "fujitaFigureBestLabel": run_name(metadata, best["label"]),
        "fujitaFigureBestHeldoutNLL": f"{best['heldout_nll']:.2f}",
        "fujitaFigureBestDeltaNLL": f"{best['heldout_delta_nll']:.2f}",
        "fujitaFigureRunnerUpLabel": run_name(metadata, runner_up["label"]),
        "fujitaFigureRunnerUpHeldoutNLL": f"{runner_up['heldout_nll']:.2f}",
        "fujitaFigureRunnerUpDeltaNLL": f"{runner_up['heldout_delta_nll']:.2f}",
        "fujitaFigureConditionCount": str(evaluation["n_conditions"]),
        "fujitaFigureMeasurementCount": str(evaluation["n_measurements"]),
        "fujitaFigureReferenceNLL": f"{metadata['reference']['heldout_nll']:.2f}",
        "fujitaFigureAbsoluteTolerance": f"\\ensuremath{{{evaluation['atol']:g}}}",
        "fujitaFigureRelativeTolerance": f"\\ensuremath{{{evaluation['rtol']:g}}}",
        "fujitaFigureSymlogBase": str(metadata["symlog"]["base"]),
        "fujitaFigureSymlogThreshold": str(metadata["symlog"]["linthresh"]),
        "fujitaFigureSymlogLinearScale": str(metadata["symlog"]["linscale"]),
        "fujitaFigureReferenceTopologyCount": str(
            sum(row["contains_reference_topology"] for row in rows)
        ),
        "fujitaFigureTopologyScoredCount": str(
            sum(row["contains_reference_topology"] and row["complete_heldout"] for row in rows)
        ),
    }
    for entry in topology_counts(rows):
        spec = arms(metadata)[entry["arm"]]
        named = "fujitaFigureArm" + macro_name(spec["code"])
        values[named + "ReferenceTopology"] = str(entry["n_retained"])
        values[named + "Models"] = str(entry["n_models"])
        values[named + "NoModel"] = str(spec["n_no_model"])
        llm, cpu, _, total = cost_totals(metadata, entry["arm"])
        values[named + "MeanUSD"] = f"{llm + cpu:,.2f}"
        values[named + "LLMUSD"] = f"{llm:,.2f}"
        values[named + "TotalUSD"] = f"{total:,.0f}"
    from gemot.paper_fujita_models import annexed

    values["fujitaAnnexModelCount"] = str(len(annexed(metadata, rows)))
    deltas = metadata["validation_deltas"]
    values["fujitaFigureValidationExampleCount"] = str(len(metadata["validation_examples"]))
    values["fujitaFigureValidationRunCount"] = str(len(deltas))
    for entry in metadata["validation_examples"]:
        sample = sorted(row["delta_nll"] for row in deltas if row["example"] == entry["example"])
        name = entry["example"].capitalize()
        middle = len(sample) // 2
        median = sample[middle] if len(sample) % 2 else (sample[middle - 1] + sample[middle]) / 2
        values[f"fujitaValidation{name}MedianDeltaNLL"] = f"{median:.1f}"
        values[f"fujitaValidation{name}RunCount"] = str(len(sample))
    coefficients, _ = correlations(rows, metadata, metadata["symlog"])
    feature_names = dict(
        zip(
            FEATURES,
            (
                "TrainingNLL",
                "BIC",
                "AIC",
                "Reactions",
                "States",
                "InputDepth",
                "UpstreamDepth",
                "DownstreamDepth",
            ),
            strict=True,
        )
    )
    # Every coefficient is now per language model, so every macro names which one: a bare
    # \fujitaFigureReactionsSpearman would silently pick whichever pool came first.
    families = {
        pool: "OpenWeight" if config["code"] == "O" else "Frontier"
        for pool, config in pools(metadata).items()
    }
    for coefficient in coefficients:
        prefix = (
            "fujitaFigure" + families[coefficient["pool"]] + feature_names[coefficient["feature"]]
        )
        values[prefix + "Spearman"] = f"{coefficient['partial_spearman']:.2f}"
        values[prefix + "PearsonSymlog"] = f"{coefficient['partial_pearson_symlog']:.2f}"
    for feature, label in (
        ("training_nll", "Core"),
        ("training_bic", "Criterion"),
        ("downstream_depth", "Depth"),
    ):
        for pool, family in families.items():
            coefficient = next(
                row for row in coefficients if row["feature"] == feature and row["pool"] == pool
            )
            named = f"fujitaFigure{family}{label}Correlation"
            values[named + "DF"] = str(coefficient["residual_correlation_df"])
            values[named + "Controls"] = str(coefficient["n_covariates"])
            values[named + "N"] = str(coefficient["n"])
    for row in rows:
        prefix = "fujitaModel" + macro_name(run_name(metadata, row["label"]))
        values[prefix + "TrainingNLL"] = f"{row['training_nll']:.2f}"
        values[prefix + "HeldoutNLL"] = (
            f"{row['heldout_nll']:.2f}" if row["complete_heldout"] else "unscorable"
        )
        values[prefix + "DownstreamDepth"] = str(row["downstream_depth"])
    for diagnostic in correlation_sensitivity(rows, metadata, metadata["symlog"]):
        if diagnostic["feature"] == "downstream_depth" and diagnostic["omitted_label"] == "B3":
            prefix = "fujitaFigureDownstreamWithout" + macro_name(run_name(metadata, "B3"))
            for field, suffix in (
                ("partial_spearman", "Spearman"),
                ("partial_pearson_symlog", "PearsonSymlog"),
            ):
                values[prefix + suffix] = (
                    f"{float(diagnostic[field]):.2f}" if diagnostic[field] else "undefined"
                )
    return values


def export_data(root, metadata, rows, header):
    coefficients, sampled = correlations(rows, metadata, metadata["symlog"])
    leverage = correlation_sensitivity(rows, metadata, metadata["symlog"])
    output = io.StringIO(newline="")
    fields = [*coefficients[0], "tasks_used", "excluded_tasks"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in coefficients:
        sample = sampled[row["pool"]]["features"][row["feature"]]
        # Drop platform-dependent final floating-point bits from deterministic text exports.
        rounded = {
            key: f"{value:.12f}" if isinstance(value, float) else value
            for key, value in row.items()
        }
        writer.writerow(
            {
                **rounded,
                "tasks_used": ",".join(sample["tasks_used"]),
                "excluded_tasks": "; ".join(
                    f"{task}: {reason}" for task, reason in sample["excluded"].items()
                ),
            }
        )
    topology = [
        {
            "task": row["task"],
            "arm": row["arm"],
            "label": row["label"],
            "name": run_name(metadata, row["label"]),
            "contains_reference_topology": row["contains_reference_topology"],
            "exact_reference_topology": row["exact_reference_topology"],
            "additional_forward_connections": row["additional_forward_connections"],
            "exclusion_reason": row["topology_reason"],
            **row["topology_paths"],
        }
        for row in rows
    ]
    # No figure floats are emitted. A generated `\begin{figure*}` carries a generated CAPTION,
    # and a caption is prose the manuscript owns -- regenerating it silently reverts whatever was
    # written in Overleaf. The graphics and the value macros ship; the float does not.
    return {
        "fujita_generalisation.csv": (root / "paper/fujita_generalisation.csv").read_text(),
        "fujita_generalisation.json": (root / "paper/fujita_generalisation.json").read_text(),
        "fujita_correlations.csv": output.getvalue(),
        "fujita_correlation_sensitivity.csv": csv_text(leverage),
        "fujita_topology.csv": csv_text(topology),
        "fujita_topology_counts.csv": csv_text(topology_counts(rows)),
        "fujita_validation_deltas.csv": (root / "paper/fujita_validation_deltas.csv").read_text(),
    }


def partial_pearson(matrix):
    """Partial r for the first/last columns given the intervening controls.

    Affine standardization conditions the least-squares solve without changing
    the correlation or projection space. Check independently via precision.
    """
    import numpy as np

    n, columns = matrix.shape
    scale = matrix.std(axis=0)
    if n <= columns:
        raise ValueError("insufficient residual degrees of freedom")
    if not (scale > 0).all():
        raise ValueError("constant variable after omission")
    scaled = (matrix - matrix.mean(axis=0)) / scale
    design = np.column_stack([np.ones(n), scaled[:, 1:-1]])
    design_rank = int(np.linalg.matrix_rank(design))
    full_rank = int(np.linalg.matrix_rank(scaled))
    if design_rank != columns - 1:
        raise ValueError("collinear controls")
    x, y = scaled[:, 0], scaled[:, -1]
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.linalg.norm(rx), np.linalg.norm(ry)) <= 1e-9:
        raise ValueError("zero residual variance")
    value = float(np.corrcoef(rx, ry)[0, 1])
    if full_rank == columns:
        precision = np.linalg.inv(np.corrcoef(scaled.T))
        expected = -precision[0, -1] / np.sqrt(precision[0, 0] * precision[-1, -1])
        assert np.isclose(value, expected, atol=1e-10)
    return value, design_rank, full_rank


def correlations(rows, metadata, symlog):
    import numpy as np
    from scipy.stats import rankdata

    results, sampled = [], {}
    for pool, config in pools(metadata).items():
        selected = [row for row in rows if row["arm"] in config["arms"]]
        sampled[pool] = {
            "n_total": len(selected),
            "n_topology": sum(row["contains_reference_topology"] for row in selected),
            "n_finite": sum(
                row["complete_heldout"] and row["contains_reference_topology"] for row in selected
            ),
            "features": {},
        }
        for feature in FEATURES:
            controls = controls_for(feature)
            required = [feature, *controls]
            eligible = [
                row
                for row in selected
                if row["contains_reference_topology"]
                and row["complete_heldout"]
                and all(row[key] is not None for key in required)
            ]
            sampled[pool]["features"][feature] = {
                "n_used": len(eligible),
                "tasks_used": [row["task"] for row in eligible],
                "excluded": {
                    row["task"]: (
                        "missing reference connection: " + row["topology_reason"]
                        if not row["contains_reference_topology"]
                        else "incomplete held-out integration"
                        if not row["complete_heldout"]
                        else "undefined " + ",".join(key for key in required if row[key] is None)
                    )
                    for row in selected
                    if row not in eligible
                },
            }
            matrix = np.array(
                [[row[key] for key in (*required, "heldout_delta_nll")] for row in eligible],
                dtype=float,
            )
            assert np.isfinite(matrix).all()
            ranked = np.column_stack([rankdata(column, method="average") for column in matrix.T])
            rho, design_rank, full_rank = partial_pearson(ranked)
            transformed = matrix.copy()
            transformed[:, -1] = symlog_transform(matrix[:, -1], symlog)
            pearson, pearson_rank, pearson_full_rank = partial_pearson(transformed)
            n = len(eligible)
            results.append(
                {
                    "pool": pool,
                    "feature": feature,
                    "partial_spearman": rho,
                    "partial_pearson_symlog": pearson,
                    "n": n,
                    "n_covariates": len(controls),
                    "residual_correlation_df": n - len(controls) - 2,
                    "design_rank": design_rank,
                    "full_rank": full_rank,
                    "pearson_design_rank": pearson_rank,
                    "pearson_full_rank": pearson_full_rank,
                    "controlled_features": ",".join(controls),
                }
            )
    return results, sampled


def correlation_sensitivity(rows, metadata, symlog):
    """Delete each eligible model once, retaining the figure's controls and transforms.

    These diagnostics describe leverage; the smaller samples are not new estimates
    to select from. Undefined coefficients are explicit, including the input-depth
    predictor becoming constant when its only two-hop model is omitted.
    """
    import numpy as np
    from scipy.stats import rankdata

    coefficients, sampled = correlations(rows, metadata, symlog)
    lookup = {row["task"]: row for row in rows}
    output = []
    for coefficient in coefficients:
        feature, pool = coefficient["feature"], coefficient["pool"]
        tasks = sampled[pool]["features"][feature]["tasks_used"]
        controls = coefficient["controlled_features"].split(",")
        keys = [feature, *controls, "heldout_delta_nll"]
        for omitted in tasks:
            sample = [task for task in tasks if task != omitted]
            matrix = np.array([[lookup[task][key] for key in keys] for task in sample])
            ranked = np.column_stack([rankdata(column, method="average") for column in matrix.T])
            transformed = matrix.copy()
            transformed[:, -1] = symlog_transform(matrix[:, -1], symlog)
            record = {
                "pool": pool,
                "feature": feature,
                "omitted_label": lookup[omitted]["label"],
                "omitted_task": omitted,
                "n": len(sample),
                "n_covariates": len(controls),
                "residual_correlation_df": len(sample) - len(controls) - 2,
                "controlled_features": ",".join(controls),
                "tasks_used": ",".join(sample),
            }
            for name, values in (
                ("partial_spearman", ranked),
                ("partial_pearson_symlog", transformed),
            ):
                try:
                    value, _, _ = partial_pearson(values)
                except ValueError as error:
                    record[name], record[name + "_status"] = "", str(error)
                else:
                    record[name], record[name + "_status"] = f"{value:.12f}", "ok"
            output.append(record)
    return output


#: Panel A puts the design on TWO channels, because the design has two independent parts.
#:
#: The MARKER names the language model — a circle for the open-weight one, a square for the frontier
#: one — and the COLOUR is a four-step SEQUENTIAL ramp over how much of the harness the agent
#: kept. The arms are not six unrelated conditions: they are one graded ablation, from the whole
#: harness down to an agent given neither a curator nor the biology, so the channel that carries
#: them is graded too. Dark is the full harness, pale is the least of it.
#:
#: An arm's name then COMPOSES the two channels — O2 reads as "open-weight model, second rung" — which
#: is why the legend can teach the channels separately and leave every point, bar and tick label
#: to be read off them rather than looked up.
#:
#: Each model has its OWN HUE, stepped the same way, so the rung is legible without first deciding
#: which model a mark belongs to and the marker is a second, redundant reading of the same fact
#: rather than the only one. The interleaved eight were checked for colour-vision separation; the
#: pale steps carry a text label wherever they appear, which is what makes their contrast legal.
SUPPORT_RAMPS = (
    ("#0A3559", "#1F6BA8", "#4BA3E0", "#9BD0F2"),
    ("#4A2404", "#A05C0A", "#E8951F", "#F6CE9C"),
)
MODEL_MARKERS = ("o", "s")
#: The neutral mark of the symbol key, where a row means a property of a run and not an arm.
MARKER = "o"
#: Raster resolution of the exported figures. Only the PNG is affected in substance -- the PDF is
#: vector -- but both are written at the same number so the two exports of one figure cannot
#: disagree. Label placement is NOT affected: ``place_labels`` and ``claimed`` scale by
#: ``figure.dpi``, which is the figure's own, set at creation and untouched by ``savefig(dpi=)``.
FIGURE_DPI = 300
#: The measured quantities are NAMES, not an ordered quantity, so they get a categorical palette
#: rather than a ramp. Hues are chosen away from the model palette and checked for colour-vision
#: separation; each is always accompanied by its own text label, in panel B and the model
#: maps alike.
READOUT_PALETTE = ("#009E73", "#C0407F", "#6A4C9C", "#96780A")


def short_model(name):
    return str(name).split("/")[-1]


def factor_levels(metadata, factor):
    """The values a factor takes, in the order the manifest's arms introduce them."""
    return list(dict.fromkeys(spec["factors"][factor] for spec in arms(metadata).values()))


def model_ramp(metadata, model):
    """The shade ramp for one language model: its own hue, stepped by rung."""
    index = factor_levels(metadata, "model").index(model)
    return SUPPORT_RAMPS[index % len(SUPPORT_RAMPS)]


def factor_encoding(metadata):
    """``arm -> colour``: the model's hue, at the step its rung of the ablation claims."""
    return {
        arm: model_ramp(metadata, spec["factors"]["model"])[
            (spec["support"] - 1) % len(SUPPORT_RAMPS[0])
        ]
        for arm, spec in arms(metadata).items()
    }


def marker_encoding(metadata):
    """``arm -> marker``: one shape per language model, in the order the arms introduce them."""
    models = factor_levels(metadata, "model")
    return {
        arm: MODEL_MARKERS[models.index(spec["factors"]["model"]) % len(MODEL_MARKERS)]
        for arm, spec in arms(metadata).items()
    }


#: The harness as the paper names it. The manifests record the bare-agent control by its driver's
#: library, ``deepagents`` -- which the gemot panel is built with too, so in print that name says
#: nothing about which arm it is.
HARNESS_NAMES = {"deepagents": "bare agent"}


def support_label(kept):
    """A support rung as printed: ``biology + deepagents`` -> ``biology + bare agent``."""
    return " + ".join(HARNESS_NAMES.get(part, part) for part in kept.split(" + "))


def support_ladder(metadata):
    """``(level, what the arm kept, the arms on that rung)`` for the rungs that were run.

    The manifest names every rung of the ablation; only the ones with arms are drawn, so a legend
    never advertises a condition the sweeps never ran. What the arm kept is the printed label
    (:func:`support_label`).
    """
    ladder = []
    for level, kept in sorted(metadata["support_levels"].items(), key=lambda item: int(item[0])):
        here = [arm for arm, spec in arms(metadata).items() if spec["support"] == int(level)]
        if here:
            ladder.append((int(level), support_label(kept), here))
    return ladder


def run_name(metadata, label):
    """``B3`` -> ``O1c``: the arm's code, then which of its runs.

    The stored label carries the arm's one-letter key, which is an archive identifier and says
    nothing a reader can use. The displayed name is the arm's position in the ablation instead,
    so a point in panel A and a tick in panel C carry the same two-channel reading.
    """
    codes = {spec["prefix"]: spec["code"] for spec in arms(metadata).values()}
    prefix, index = label[:-1], label[-1]
    if prefix not in codes or not index.isdigit():
        raise ValueError(f"unsupported Fujita model label {label}")
    return codes[prefix] + chr(ord("a") + int(index) - 1)


def arm_style(metadata, arm):
    return factor_encoding(metadata)[arm], marker_encoding(metadata)[arm]


def shaded(color, amount=0.62):
    """The same hue, darker — the outline of a filled marker.

    A white outline separates overlapping marks but erases the pale end of a sequential ramp
    against the page. Drawing each mark's own colour darker keeps the separation and keeps the
    rung readable.
    """
    value = int(color.lstrip("#"), 16)
    parts = ((value >> 16) & 255, (value >> 8) & 255, value & 255)
    return "#%02X%02X%02X" % tuple(round(part * amount) for part in parts)


def factor_groups(metadata, rows):
    """Rows grouped by arm — with four factors encoded, the cell IS the arm."""
    groups = {arm: [] for arm in arms(metadata)}
    for row in rows:
        groups[row["arm"]].append(row)
    return {arm: sample for arm, sample in groups.items() if sample}


def candidate_network(metadata, row):
    """A candidate's readout network, in the same shape as the reference's.

    Built from the audited collapsed paths, so it says exactly what the topology panel counts. The
    influence SIGNS are the reference's alone — they were read off published kinetic laws — so a
    candidate's arcs are drawn unsigned rather than borrowing a sign nobody tested.
    """
    reference = metadata["reference_network"]
    anchors = {
        "EGF": [row["stimulus_anchor"]],
        "pEGFR": row["receptor_states"].split(","),
        "pAkt": row["akt_states"].split(","),
        "pS6": row["s6_states"].split(","),
    }
    by_label = {node["label"]: node for node in reference["nodes"]}
    nodes = [
        {**by_label[label], "species": [name for name in anchors[label] if name]}
        for label in ("EGF", "pEGFR", "pAkt", "pS6")
    ]
    edges = []
    for key, source, target in CONNECTIONS[:3]:
        chain = (row["topology_paths"].get(key) or "").split(" -> ")
        if len(chain) < 2:
            continue
        edges.append(
            {
                "source": by_label[source]["id"],
                "target": by_label[target]["id"],
                "hops": len(chain) - 1,
                "path": chain,
                "influence": "untested",
            }
        )
    return {"nodes": nodes, "edges": edges}


def network_summary(diagram):
    """Two terse lines under a readout map: how big the model is, and what it implements."""
    from gemot import sbgn

    counts = diagram["counts"]
    anchors = {readout["id"]: readout["species"] for readout in diagram["readouts"]}
    return (
        f"{counts['n_parameters']} parameters · {counts['n_reactions']} reactions · "
        f"{counts['n_states']} states",
        "implements " + ", ".join(sbgn.motifs(diagram, anchors)),
    )


def readout_colors(metadata):
    """Network node id -> its colour, in the order panel B stacks them."""
    nodes = metadata["reference_network"]["nodes"]
    return {
        node["id"]: READOUT_PALETTE[index % len(READOUT_PALETTE)]
        for index, node in enumerate(nodes)
    }


def readout_ranks(metadata):
    """Network node id -> its place in panel B, top first. The model maps align to this."""
    return {node["id"]: index for index, node in enumerate(metadata["reference_network"]["nodes"])}


def readout_anchors(metadata, diagram):
    """Entity pool -> the rank panel B gives the quantity it is measured as.

    This is what lets the model maps be read against each other and against B: one measurement sits
    at the same place along the flow in all three, whatever the model calls its species.
    """
    ranks = readout_ranks(metadata)
    anchors = {}
    for readout in diagram["readouts"]:
        if readout["id"] in ranks:
            anchors.update({name: ranks[readout["id"]] for name in readout["species"]})
    for node in diagram["nodes"]:
        if node["class"] == "perturbation" and node["id"] in ranks:
            anchors[node["id"]] = ranks[node["id"]]
    return anchors


def size_ranks(rows, counts):
    """Where a model's size sits among the candidates; rank 1 is the smallest.

    Ranked only against the models that RECOVERED the reference connections. A model that missed
    one is small or large for a reason that has nothing to do with parsimony, and letting it into
    the denominator makes a size look more ordinary than it is.
    """
    rows = [row for row in rows if row["contains_reference_topology"]]
    ranked = []
    for key, name in (
        ("n_parameters", "parameters"),
        ("n_states", "states"),
        ("n_reactions", "reactions"),
    ):
        value = counts[key]
        ranked.append((name, value, 1 + sum(row[key] < value for row in rows), len(rows)))
    return ranked


def readout_highlight(metadata, diagram):
    """Entity pool -> ``(colour, observable)`` for a process-description map.

    Two models of the same system share no species names, so the observable is the only thing the
    two maps have in common. Colouring and naming the pools each observable sums is what lets the
    reference and a candidate be read against one another at all.
    """
    colors = readout_colors(metadata)
    names = {node["id"]: node["label"] for node in metadata["reference_network"]["nodes"]}
    highlight = {}
    for readout in diagram["readouts"]:
        if readout["id"] not in colors:
            continue
        for species in readout["species"]:
            highlight[species] = (colors[readout["id"]], names[readout["id"]])
    for node in diagram["nodes"]:
        # The stimulus is the same quantity panel B puts at the top of the cascade.
        if node["class"] == "perturbation" and node["id"] in colors:
            highlight[node["id"]] = (colors[node["id"]], names[node["id"]])
    return highlight


def arm_tick_colors(axis, metadata, order):
    """Each arm's code on the x axis in that arm's colour, as panel A labels its runs."""
    for label, arm in zip(axis.get_xticklabels(), order, strict=True):
        label.set_color(arm_style(metadata, arm)[0])


def render_topology_counts(axis, rows, metadata, fontsize=6.0):
    """One bar per arm: models recovering the reference forward readout connections."""
    spec = arms(metadata)
    counts = topology_counts(rows)
    order = [row["arm"] for row in counts]
    bar_colors = [arm_style(metadata, arm)[0] for arm in order]
    axis.bar(
        range(len(counts)),
        [row["n_retained"] for row in counts],
        width=0.68,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        zorder=3,
    )
    maximum = max(row["n_models"] for row in counts)
    axis.set(ylim=(0, maximum + 0.9), yticks=range(0, maximum + 1, 2))
    for x, row in enumerate(counts):
        # The denominator is the models BUILT: a run with no fittable model has no topology to
        # contain, and is left to the caption rather than counted as a topology failure.
        axis.text(
            x,
            row["n_retained"] + 0.12,
            f"{row['n_retained']}/{row['n_models']}",
            ha="center",
            va="bottom",
            fontsize=fontsize - 1.8,
            fontweight="bold",
            color=bar_colors[x],
        )
    axis.set_ylabel("Models with\nreference connections", fontsize=fontsize, labelpad=1)
    axis.set_xticks(
        range(len(counts)), [spec[arm]["code"] for arm in order], fontsize=fontsize + 0.6
    )
    axis.set_xlim(-0.62, len(counts) - 0.38)
    axis.spines["bottom"].set_position(("data", 0))
    axis.tick_params(axis="x", length=0, pad=1)
    # Last: set_xlim and tick_params rebuild the tick labels, dropping a colour set earlier.
    arm_tick_colors(axis, metadata, order)
    axis.grid(axis="y", color="#E2E7EA", linewidth=0.6)


def heldout_win_counts(rows):
    """Per arm, how many models PREDICT better than the published reference.

    The denominator is every model the arm delivered. A model whose held-out integration failed
    made no usable prediction of the withheld protocols, and that counts as predicting worse than
    the reference: dropping it from the denominator would reward an arm for producing models that
    cannot be integrated. ``n_scored`` stays in the output so the two failure modes remain
    distinguishable in the data.
    """
    output = []
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        scored = [row for row in selected if row["complete_heldout"]]
        better = [row for row in scored if row["heldout_delta_nll"] < 0]
        output.append(
            {
                "arm": arm,
                "n_models": len(selected),
                "n_scored": len(scored),
                "n_better": len(better),
                "better_tasks": ",".join(row["task"] for row in better),
            }
        )
    return output


def render_heldout_wins(axis, rows, metadata, fontsize=6.0):
    """One bar per arm: models that beat the reference on the held-out protocols.

    Deliberately the same shape as panel C, because the two are the figure's paired question --
    C is whether the arm RECOVERED the reference wiring, this is whether it PREDICTS better than
    the reference does. Reading them side by side is the point, so they share bar width, colour,
    the count-over-total label and the factor rows.
    """
    spec = arms(metadata)
    counts = heldout_win_counts(rows)
    order = [row["arm"] for row in counts]
    bar_colors = [arm_style(metadata, arm)[0] for arm in order]
    axis.bar(
        range(len(counts)),
        [row["n_better"] for row in counts],
        width=0.68,
        color=bar_colors,
        edgecolor=bar_colors,
        linewidth=1.0,
        zorder=3,
    )
    maximum = max(row["n_models"] for row in counts)
    axis.set(ylim=(0, maximum + 0.9), yticks=range(0, maximum + 1, 2))
    for x, row in enumerate(counts):
        axis.text(
            x,
            row["n_better"] + 0.12,
            f"{row['n_better']}/{row['n_models']}",
            ha="center",
            va="bottom",
            fontsize=fontsize - 1.8,
            fontweight="bold",
            color=bar_colors[x],
        )
    axis.set_ylabel("Models beating\nthe reference", fontsize=fontsize, labelpad=1)
    axis.set_xticks(
        range(len(counts)), [spec[arm]["code"] for arm in order], fontsize=fontsize + 0.6
    )
    axis.set_xlim(-0.62, len(counts) - 0.38)
    axis.spines["bottom"].set_position(("data", 0))
    axis.tick_params(axis="x", length=0, pad=1)
    # Last: set_xlim and tick_params rebuild the tick labels, dropping a colour set earlier.
    arm_tick_colors(axis, metadata, order)
    axis.grid(axis="y", color="#E2E7EA", linewidth=0.6)


def time_totals(metadata, arm):
    """``(tool_hours, rest_hours, split)`` per model delivered, over every run the arm launched.

    Wall time is the Slurm elapsed time -- recorded for every task, including the runs that built
    no model, which the run's own clock is not. It is split by the runner's tool-call clock
    (``compute_seconds``: compiles, fits, simulations) into time spent computing and the rest,
    which is waiting on the LLM plus harness overhead. The barebones driver does not time its tool
    calls, so its arms record no split: ``split`` is False and the whole wall is returned as rest.
    """
    runs = [
        run["cost"]
        for run in metadata["runs"]
        if run["cost"]
        and next((c["arm"] for c in metadata["costs"] if c["task"] == run["task"]), None) == arm
    ]
    runs += [run["cost"] for run in metadata["no_model_runs"] if run["arm"] == arm]
    delivered = cost_totals(metadata, arm)[2]
    wall = sum(run["elapsed_seconds"] or 0.0 for run in runs)
    tool = sum(run.get("compute_seconds") or 0.0 for run in runs)
    # A few untimed seconds are not a measurement: under 1% of the wall, the split is absent.
    split = wall > 0 and tool / wall >= 0.01
    if not split:
        tool = 0.0
    return tool / 3600 / delivered, (wall - tool) / 3600 / delivered, split


def render_time(axis, rows, metadata, fontsize=6.0):
    """Hours per model delivered: tool calls (the CPU side) under LLM waiting and overhead."""
    spec = arms(metadata)
    order = [entry["arm"] for entry in topology_counts(rows)]
    totals = [time_totals(metadata, arm) for arm in order]
    positions = range(len(order))
    tool_ink, rest_ink = "#8A949B", "#26343E"
    for x, (tool, rest, split) in zip(positions, totals, strict=True):
        if split:
            axis.bar(
                x, tool, width=0.68, color="#B8C4CC", edgecolor=tool_ink, linewidth=0.5, zorder=3
            )
            axis.bar(x, rest, width=0.68, bottom=tool, color=rest_ink, edgecolor=rest_ink, zorder=3)
        else:
            # Not split: the barebones driver's tool calls are untimed, so the bar is the whole
            # wall, hatched, rather than a guess at the division.
            axis.bar(
                x,
                rest,
                width=0.68,
                color="white",
                edgecolor=rest_ink,
                linewidth=0.6,
                hatch="////",
                zorder=3,
            )
    top = max(tool + rest for tool, rest, _ in totals)
    for x, (tool, rest, _) in zip(positions, totals, strict=True):
        axis.text(
            x,
            tool + rest + top * 0.08,
            f"{tool + rest:.1f} h",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize - 1.6,
            color=rest_ink,
        )
    # Headroom for the legend above the tallest bar and its vertical label.
    axis.set_ylim(0, top * 2.4)
    axis.set_ylabel("Hours per model", fontsize=fontsize, labelpad=2)
    axis.set_xticks(positions, [spec[arm]["code"] for arm in order], fontsize=fontsize + 0.6)
    axis.set_xlim(-0.62, len(order) - 0.38)
    axis.spines["bottom"].set_position(("data", 0))
    axis.grid(axis="y", color="#E2E7EA", linewidth=0.6)
    from matplotlib.patches import Patch

    axis.legend(
        handles=[
            Patch(facecolor=rest_ink, label="LLM"),
            Patch(facecolor="#B8C4CC", edgecolor=tool_ink, linewidth=0.5, label="CPU"),
            Patch(
                facecolor="white",
                edgecolor=rest_ink,
                hatch="////",
                linewidth=0.6,
                label="not split",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.06, 1.0),
        frameon=False,
        fontsize=fontsize - 1.2,
        handlelength=0.9,
        handletextpad=0.35,
        borderpad=0.0,
        labelspacing=0.2,
    )
    axis.tick_params(axis="x", length=0, pad=1)
    arm_tick_colors(axis, metadata, order)


def render_cost(axis, rows, metadata, fontsize=6.0):
    """What one MODEL of each arm cost: tokens bought, plus the allocation as rented hardware.

    Every run the arm launched is charged, including the ones that delivered nothing, and the bill
    is then divided by the models that survived. A failure is the most expensive kind of run there
    is, and per-run averaging would hide it in exactly the arms that fail most.
    """
    spec = arms(metadata)
    order = [entry["arm"] for entry in topology_counts(rows)]
    llm = [cost_totals(metadata, arm)[0] for arm in order]
    cpu = [cost_totals(metadata, arm)[1] for arm in order]
    positions = range(len(order))
    llm_ink, cpu_ink = "#26343E", "#8A949B"
    axis.bar(positions, llm, width=0.68, color=llm_ink, edgecolor=llm_ink, zorder=3, label="LLM")
    axis.bar(
        positions,
        cpu,
        width=0.68,
        bottom=llm,
        color="#B8C4CC",
        edgecolor=cpu_ink,
        linewidth=0.5,
        zorder=3,
        label="CPU",
    )
    top = max(a + b for a, b in zip(llm, cpu, strict=True))
    # Each half priced on its own segment and in that segment's own ink: the split is the panel's
    # point, and a single total hides which half an arm is actually spending.
    # ONE price per bar, the total, vertical. At text width a bar is ~17 pt and two stacked 6 pt
    # prices per bar collided with each other and with the neighbours; the split stays legible
    # from the two segments themselves and is itemised in the protocol appendix.
    for x in positions:
        axis.text(
            x,
            llm[x] + cpu[x] + top * 0.03,
            f"${llm[x] + cpu[x]:,.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize - 1.6,
            color=llm_ink,
        )
    axis.set_ylim(0, top * 1.55)
    axis.set_yticks([tick for tick in axis.get_yticks() if 0 <= tick <= top * 1.55])
    axis.set_ylabel("USD per model", fontsize=fontsize, labelpad=2)
    axis.set_xticks(positions, [spec[arm]["code"] for arm in order], fontsize=fontsize + 0.6)
    axis.set_xlim(-0.62, len(order) - 0.38)
    axis.spines["bottom"].set_position(("data", 0))
    axis.tick_params(axis="x", length=0, pad=1)
    # Last: set_xlim and tick_params rebuild the tick labels, dropping a colour set earlier.
    arm_tick_colors(axis, metadata, order)
    axis.grid(axis="y", color="#E2E7EA", linewidth=0.6)
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=fontsize - 1.2,
        handlelength=0.9,
        handletextpad=0.35,
        borderpad=0.0,
        labelspacing=0.2,
    )


def render_marginals(top, right, rows, metadata, xlim, ylim, clip, fontsize=5.0):
    """Median and range per (model, harness) cell, on panel A's own axes.

    Drawn as a range bar with a median marker rather than a box, and the four cells jittered into
    ONE thin band instead of four rows. A box per row spent a fifth of the page restating what the
    cloud already shows; the question these answer — does the model or the harness move the median —
    survives the compression, and each cell is still named by its own colour and marker.
    """
    import numpy as np

    groups = factor_groups(metadata, rows)
    # In LADDER order, top to bottom: the strip is read against panel A's colour key and against
    # panels C and E, and all three then step through the ablation the same way. Ranking by median
    # instead put the rungs in a different order in every panel, which is a harder question to ask
    # of a reader than "does the median move as the harness is taken away".
    order = [arm for arm in arms(metadata) if arm in groups]
    offsets = np.linspace(0.40, -0.40, len(order))
    for offset, arm in zip(offsets, order, strict=True):
        color, mark = arm_style(metadata, arm)
        training = [clip(row["training_dbic"]) for row in groups[arm]]
        # A failed held-out integration ranks worst (Figure 3D's rule): at the top of the frame, so
        # it counts in the median and carries the range to the edge.
        heldout = [
            min(row["heldout_delta_nll"], ylim[1]) if row["complete_heldout"] else ylim[1]
            for row in groups[arm]
        ]
        for axis, sample, flip in ((top, training, False), (right, heldout, True)):
            if not sample:
                continue
            low, high, middle = min(sample), max(sample), float(np.median(sample))
            line = ([low, high], [offset, offset]) if not flip else ([offset, offset], [low, high])
            axis.plot(*line, color=color, linewidth=0.9, solid_capstyle="butt", zorder=3)
            for end in (low, high):
                cap = (
                    ([end, end], [offset - 0.1, offset + 0.1])
                    if not flip
                    else ([offset - 0.1, offset + 0.1], [end, end])
                )
                axis.plot(*cap, color=color, linewidth=0.9, zorder=3)
            at = ([middle], [offset]) if not flip else ([offset], [middle])
            axis.plot(
                *at,
                marker=mark,
                markersize=2.8,
                markerfacecolor=color,
                markeredgecolor=shaded(color),
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
            )
    top.set_xlim(*xlim)
    top.set_ylim(-0.56, 0.56)
    top.set_yticks([])
    top.tick_params(axis="x", labelbottom=False, length=0)
    top.set_ylabel(
        "median\n& range",
        fontsize=fontsize,
        color="#5F6B74",
        rotation=0,
        ha="right",
        va="center",
        labelpad=6,
    )
    right.set_ylim(*ylim)
    right.set_xlim(-0.56, 0.56)
    right.set_xticks([])
    right.tick_params(axis="y", labelleft=False, length=0)
    right.axhline(0, color="#526370", linestyle="--", linewidth=0.7)
    right.set_xlabel("median\n& range", fontsize=fontsize, color="#5F6B74")
    for axis in (top, right):
        for spine in axis.spines.values():
            spine.set_visible(False)


def place_labels(
    axis, items, fontsize, pad=4.0, taken=None, steps=(0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5)
):
    """Annotate points with names that do not overlap each other, adding a leader when pushed far.

    Two arms cluster in the same narrow band of training ΔBIC, and a fixed offset cycle cannot know
    that: each arm staggers its own labels and they land on top of one another anyway. This tries a
    fixed ladder of offsets per point and takes the first that is free, so the placement stays
    deterministic while the result depends on what is actually crowded.

    ``items`` are ``(x, y, text, colour)`` in data coordinates. Box sizes are estimated from the
    font size rather than measured: a renderer is not available before the figure is drawn, and an
    estimate is enough to keep names apart.

    ``taken`` carries claimed rectangles IN and OUT, so the marks themselves can be claimed before
    any name is placed and two calls can share one page. ``steps`` is the ladder of vertical
    offsets, in label heights, tried in order: a band that must not grow upwards tries below first.
    """
    scale = axis.figure.dpi / 72
    ladder = [(side, step) for step in steps for side in (1, -1)]
    edge = axis.transData.transform((axis.get_xlim()[1], 0))[0]
    taken = [] if taken is None else taken
    for x, y, text, color in items:
        px, py = axis.transData.transform((x, y))
        width = len(text) * fontsize * 0.62 * scale
        # Near the right edge a label placed rightwards runs off the panel, so the ladder is
        # mirrored there rather than letting the first free slot win.
        order = ladder if px + width + pad * scale < edge else [(-s, d) for s, d in ladder]
        height = fontsize * 1.25 * scale
        for side, step in order:
            dx = side * pad * scale
            dy = step * (fontsize + 1.0) * scale
            left = px + dx - (width if side < 0 else 0)
            box = (left, py + dy - height / 2, left + width, py + dy + height / 2)
            if all(
                box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]
                for other in taken
            ):
                break
        taken.append(box)
        axis.annotate(
            text,
            (x, y),
            xytext=(dx / scale, dy / scale),
            textcoords="offset points",
            ha="right" if side < 0 else "left",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=7,
            annotation_clip=False,
        )
        if abs(step) >= 2:
            # Far enough that which point it names is no longer obvious.
            axis.annotate(
                "",
                (x, y),
                xytext=(dx / scale, dy / scale),
                textcoords="offset points",
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.4,
                    "color": color,
                    "shrinkA": 1,
                    "shrinkB": 2,
                },
                zorder=3,
                annotation_clip=False,
            )
    return taken


def claimed(axis, points, radius=3.2):
    """The marks' own boxes, so a name is never laid over the point it is naming."""
    scale = axis.figure.dpi / 72
    boxes = []
    for x, y in points:
        px, py = axis.transData.transform((x, y))
        boxes.append(
            (px - radius * scale, py - radius * scale, px + radius * scale, py + radius * scale)
        )
    return boxes


def render_scatter(axis, rows, metadata, base):
    """Panel A: training fit against held-out prediction, one point per run.

    A LINEAR held-out axis, sized to the band where the models actually differ, with everything
    above it drawn OUTSIDE the frame. A symmetric-log axis kept every point inside but made a
    hundredfold difference look like a small step; putting the few large ones outside says plainly
    that they are off the scale. They are named, not valued: the axis is sized so the comparison
    reads inside it, and a printed +243,290 beside the frame drew the eye to exactly the runs it
    does not compare.

    Returns the x limits the marginal axes must share. Every run that built a model appears, and a
    run that cannot be placed on an axis is placed OFF it: past the right edge when its ΔBIC is
    beyond the range, above the top when its ΔNLL is. A run that built no model has neither
    coordinate and is left out; the arm table and panel E still count it. Nothing is drawn at a
    coordinate it never earned.
    """
    spec = arms(metadata)
    # The ΔBIC axis follows the bulk of the models; a run further than `spread` from the median is
    # drawn clipped at the edge, so one outlier cannot flatten the cluster.
    dbics = sorted(row["training_dbic"] for row in rows)
    median = dbics[len(dbics) // 2]
    spread = 3000
    core = [value for value in dbics if abs(value - median) <= spread]
    span = max(max(core) - min(core), 100)
    xmin, xmax = min(core) - 0.12 * span, max(max(core) + 0.10 * span, 65)
    outliers = {row["task"] for row in rows if abs(row["training_dbic"] - median) > spread}
    axis.set_xlim(xmin, xmax)
    beside = xmax + 0.035 * span  # the band past the right edge

    def xpos(row):
        return beside if row["task"] in outliers else row["training_dbic"]

    scored = [row for row in rows if row["complete_heldout"]]
    floor = min(row["heldout_delta_nll"] for row in scored)
    # SYMMETRIC about the reference, rounded out to a whole thousand: the axis then reads as "how
    # much better or worse than the published model", and a gain of 3,000 looks like a loss of
    # 3,000 rather than being squashed by a scale chosen from one side.
    ceiling = math.ceil(abs(floor) / 1000) * 1000
    # Padded equally at both ends -- still symmetric, ticks still at whole thousands -- so labels
    # of the best candidates, which crowd the floor, stay inside the frame at text width instead
    # of landing on the tick labels.
    ylim = (-1.2 * ceiling, 1.2 * ceiling)
    outside = ylim[1] + 2 * ceiling * 0.055  # the band just above the frame
    axis.set_ylim(*ylim)
    axis.grid(axis="y", color="#E2E7EA", linewidth=0.6)
    axis.axhline(0, color="#526370", linestyle="--", linewidth=0.8, zorder=2)
    if xmin < 0 < xmax:
        axis.axvline(0, color="#C5CDD2", linestyle=":", linewidth=0.7)
    axis.axhspan(ylim[0], 0, color="#EEF6F2", zorder=0)
    # Bottom right: the band's empty corner. Top left is where the best candidates -- the F arms
    # -- cluster, and at text width their labels landed on this one.
    axis.annotate(
        "Better than reference",
        (1, 0),
        xycoords="axes fraction",
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=base - 1.0,
        color="#3E7863",
    )
    axis.set_ylabel("Held-out ΔNLL  (candidate − reference)", fontsize=base)
    axis.tick_params(axis="x", labelbottom=True, length=2.5, direction="out", pad=2)
    axis.set_xlabel("Training ΔBIC  (candidate − reference)", fontsize=base, labelpad=2)
    labels, above, marks = [], [], []
    for arm in spec:
        color, mark = arm_style(metadata, arm)
        subset = sorted(
            [row for row in rows if row["arm"] == arm and row["complete_heldout"]],
            key=lambda row: row["training_dbic"],
        )
        for row in subset:
            value = row["heldout_delta_nll"]
            beyond = value > ceiling
            at = outside if beyond else value
            hollow = not row["contains_reference_topology"]
            off = beyond or row["task"] in outliers
            axis.scatter(
                xpos(row),
                at,
                s=20,
                marker=mark,
                clip_on=not off,
                zorder=5 if hollow else 4,
                facecolors="white" if hollow else color,
                edgecolors=color if hollow else shaded(color),
                linewidths=0.8 if hollow else 0.5,
            )
            name = run_name(metadata, row["label"]) + ("†" if hollow else "")
            marks.append((xpos(row), at))
            (above if off else labels).append((xpos(row), at, name, color))
        for row in rows:
            if row["arm"] == arm and not row["complete_heldout"]:
                # No score at this tolerance, so no value to place: an x in the same band, which
                # says off-the-scale-or-unscored, and the marker says which.
                axis.scatter(
                    xpos(row),
                    outside,
                    marker="x",
                    s=14,
                    color=color,
                    linewidths=1.1,
                    clip_on=False,
                    zorder=6,
                )
                marks.append((xpos(row), outside))
                above.append((xpos(row), outside, run_name(metadata, row["label"]), color))
    # Every mark claims its own box FIRST, so a crowded corner pushes names off the points rather
    # than onto them, and the two bands share one page instead of colliding across it.
    page = claimed(axis, marks)
    # The band above the frame names its runs beside or BELOW them, never above: a name above the
    # band is what the gap to the marginal would otherwise have to be tall enough to hold.
    place_labels(
        axis,
        sorted(above, key=lambda item: item[0]),
        base - 1.8,
        taken=page,
        steps=(0, -1, -2, -3, 1, 2),
    )
    place_labels(axis, sorted(labels, key=lambda item: item[0]), base - 1.8, taken=page)

    return (xmin, xmax), ylim, lambda value: min(max(value, xmin), xmax)


def render(rows, coefficients, metadata, output):
    """Figure 3, laid out for a portrait A4 page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    from gemot import paper_style, sbgn

    spec = arms(metadata)
    # Sized for ICLR's 5.5 in text width at 100%, with a 6 pt floor. Every size below is derived
    # from ``base``, and the smallest derivation in use is base - 1.8, so base is 7.8.
    base = 6.0 + 1.8
    title_size = 8.0
    matplotlib.rcParams.update(
        {
            **paper_style.rc(),
            "font.size": base,
            "axes.labelsize": base - 0.6,
            "axes.titlesize": title_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#69737B",
            "axes.labelcolor": "#26343E",
            "axes.linewidth": 0.6,
            "xtick.color": "#43515C",
            "ytick.color": "#43515C",
            "xtick.labelsize": base - 1.4,
            "ytick.labelsize": base - 1.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.fontsize": base - 1.6,
            "text.color": "#26343E",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "fujita-generalisation-figure3",
        }
    )
    page_width, margins = paper_style.TEXT_WIDTH, {"left": 0.10, "right": 0.99}
    # No title and no caption block: the figure environment owns both, and repeating them in the
    # artwork only fixes wording the manuscript is free to change.
    # Top band: A on the left; on the right, B's three maps over the key. Bottom band: C/D/E
    # across the full width. At 8.27 in the key was a column under A beside right-aligned C/D/E;
    # at 5.5 in that left the bar panels under 1 in each, too narrow for six 6-pt arm codes. The
    # maps are names and arrows only now, so they need less height than A and the key takes the
    # rest of the column.
    # B's rows in inches: spacer for the titles, the maps, the key. The maps are height-limited
    # at 0.19 in per box unit, so the map row is sized to its content -- 4 boxes of 0.8 units and
    # 3 gaps of MAP_GAP -- and a wider gap adds height instead of shrinking the boxes.
    map_gap, map_unit = 1.1, 0.19
    # B's column sets the band's height, so every row is cut to its content: the spacer holds the
    # two-line map titles (B's own title sits above the band), the key row the matrix key and its
    # exception row. Panel A needs no more than that since its off-scale band names runs beside or
    # below them: the gap to its marginal only holds the band, and the figure is shorter for it.
    b_rows = (0.31, map_unit * (4 * 0.8 + 3 * map_gap + 0.8 * 0.18 + 0.55 * map_gap), 0.77)
    bands, gap, headroom, footroom = (sum(b_rows), 0.86), 0.50, 0.20, 0.17
    page = (page_width, headroom + sum(bands) + gap + footroom)
    margins |= {"top": 1 - headroom / page[1], "bottom": footroom / page[1]}
    fig = plt.figure(figsize=page, facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=(bands[0], gap, bands[1]), hspace=0.0, **margins)
    top = outer[0].subgridspec(1, 2, width_ratios=(1.02, 1.0), wspace=0.16)
    # hspace is the off-scale band and a little air: nothing is named above it (render_scatter).
    block = top[0].subgridspec(
        2, 2, width_ratios=(1, 0.075), height_ratios=(0.16, 1), wspace=0.10, hspace=0.16
    )
    # Row 0 is room for B's title and the two-line map titles, which sit above the map axes.
    right = top[1].subgridspec(3, 3, height_ratios=b_rows, wspace=0.10, hspace=0.0)
    middle = outer[2].subgridspec(1, 4, wspace=0.62)
    keys_axis = fig.add_subplot(right[2, :])
    keys_axis.axis("off")
    top_margin = fig.add_subplot(block[0, 0])
    scatter = fig.add_subplot(block[1, 0], sharex=top_margin)
    right_margin = fig.add_subplot(block[1, 1])
    maps = [fig.add_subplot(right[1, index]) for index in range(3)]
    network = maps[0]
    topology = fig.add_subplot(middle[0, 0])
    heldout_wins = fig.add_subplot(middle[0, 1])
    cost = fig.add_subplot(middle[0, 2])
    walltime = fig.add_subplot(middle[0, 3])

    # A and B on ONE baseline, just above the top band: set_title hangs each off its own axes,
    # and A's (the marginal) and B's (the maps, under their two-line titles) do not share a top.
    band_top = outer[0].get_position(fig).y1
    for axis, letter, title in (
        (top_margin, "A", "Training ΔBIC vs held-out ΔNLL"),
        (network, "B", "Readout networks"),
    ):
        fig.text(
            axis.get_position().x0,
            band_top + 4 / 72 / page[1],
            f"{letter}   {title}",
            ha="left",
            va="baseline",
            fontweight="bold",
            fontsize=title_size,
        )
    for axis, letter, title in (
        (topology, "C", "Topology recovery"),
        (heldout_wins, "D", "Held-out wins"),
        (cost, "E", "Cost per model"),
        (walltime, "F", "Wall time"),
    ):
        axis.set_title(
            f"{letter}   {title}", loc="left", pad=4, fontweight="bold", fontsize=title_size
        )

    # Panel A and its marginals.
    xlim, ylim, clip = render_scatter(scatter, rows, metadata, base)
    # The axis is LINEAR; the symmetric-log transform survives only inside the Pearson
    # correlations, where the response is transformed before residualisation.
    assert scatter.get_yscale() == "linear"
    scatter.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    render_marginals(
        top_margin, right_margin, rows, metadata, xlim, ylim, clip, fontsize=base - 1.6
    )
    # ONE key for both channels, as a matrix: a row per support rung, a column per model, and in
    # each cell that arm's own marker in its own colour beside its code. The reader finds O3 where
    # "biology + bare agent" crosses glm, instead of joining a colour key to a marker key. A cell
    # stays empty where no sweep ever ran the arm (F1, F4): the key must not advertise it. Drawn
    # rather than a legend, because a legend spends a handle's width on every row label and the
    # matrix has to fit under the maps.
    size, renderer = base - 1.6, fig.canvas.get_renderer()
    area = keys_axis.get_position()
    across, down = area.width * page[0], area.height * page[1]  # the key area in inches

    def width(text):
        """``text``'s rendered width in inches, at the key's size."""
        probe = keys_axis.text(0, 0, text, fontsize=size)
        inches = probe.get_window_extent(renderer).width / fig.dpi
        probe.remove()
        return inches

    def put(x, y, text, **style):
        """Text at ``(x, y)`` inches from the key area's top left."""
        keys_axis.text(
            x / across,
            1 - y / down,
            text,
            fontsize=size,
            va="center",
            ha="left",
            transform=keys_axis.transAxes,
            **style,
        )

    models, ladder = factor_levels(metadata, "model"), support_ladder(metadata)
    cells = {
        (level, spec[arm]["factors"]["model"]): arm for level, _kept, here in ladder for arm in here
    }
    rows_text = [f"{level}  {kept}" for level, kept, _here in ladder]
    pitch, gap, glyph = 1.42 * size / 72, 0.12, 0.10  # row pitch, column gap, marker slot (in)
    left = -0.05 * across
    code = max(width(spec[arm]["code"]) for arm in cells.values())
    columns = [left + max(width(text) for text in rows_text + ["harness support"]) + gap]
    for model in models[:-1]:
        columns.append(columns[-1] + max(width(short_model(model)), glyph + code) + gap)
    header = "#5F6B74"
    put(left, 0.5 * pitch, "harness support", color=header)
    for model, x in zip(models, columns, strict=True):
        put(x, 0.5 * pitch, short_model(model), color=header)
    for row, ((level, _kept, _here), text) in enumerate(zip(ladder, rows_text, strict=True)):
        y = (row + 1.6) * pitch
        put(left, y, text)
        for model, x in zip(models, columns, strict=True):
            arm = cells.get((level, model))
            if arm is None:
                continue
            color, mark = arm_style(metadata, arm)
            keys_axis.scatter(
                (x + glyph / 2) / across,
                1 - y / down,
                s=14,
                marker=mark,
                facecolors=color,
                edgecolors=shaded(color),
                linewidths=0.5,
                transform=keys_axis.transAxes,
                clip_on=False,
            )
            put(x + glyph, y, spec[arm]["code"])
    # The marker exceptions, one row beneath the matrix.
    marks = [
        Line2D(
            [],
            [],
            color="#43515C",
            marker=MARKER,
            linestyle="none",
            markersize=3.6,
            markerfacecolor="white",
            label="† missing reference connection",
        ),
        Line2D(
            [],
            [],
            color="#43515C",
            marker="x",
            linestyle="none",
            markersize=3.6,
            label="no held-out score",
        ),
    ]
    beneath = scatter.legend(
        handles=marks,
        loc="upper left",
        ncol=2,
        columnspacing=0.9,
        bbox_to_anchor=(left / across, 1 - (len(ladder) + 1.35) * pitch / down),
        bbox_transform=keys_axis.transAxes,
        frameon=False,
        fontsize=size,
        handletextpad=0.35,
        borderpad=0.0,
        handlelength=1.2,
    )
    scatter.add_artist(beneath)

    # Panel B and the model maps share one colour scale for the measured quantities.
    accents = readout_colors(metadata)
    ranked = sorted(
        (row for row in rows if row["complete_heldout"]), key=lambda row: row["heldout_nll"]
    )[:2]
    # Each title in the ink panel A draws that run in: the candidates in their arm's colour, the
    # reference in the neutral heading ink it has everywhere else.
    drawn = [
        (
            "Published\nreference",
            metadata["reference_network"],
            metadata["process_diagrams"]["reference"],
            "#43515C",
        )
    ]
    for row, key in zip(ranked, ("best_generalising", "runner_up"), strict=True):
        drawn.append(
            (
                f"{run_name(metadata, row['label'])}\nΔNLL {row['heldout_delta_nll']:,.0f}",
                candidate_network(metadata, row),
                metadata["process_diagrams"][key],
                arm_style(metadata, row["arm"])[0],
            )
        )
    # What the text attributes to the two best candidates, per connection (results-heldout.tex,
    # "Parsimony and prediction"): receptor internalisation on EGF -> pEGFR, the pAkt feedback on
    # pEGFR -> pAkt, and the latent intermediates each threads between two readouts. Stated, not
    # derived: the route finder follows conversion chains, so it cannot see F3d's mTOR and S6K,
    # which sit on production and catalysis links. Checked against the annexed Antimony: F2c
    # `receptor_internalize` and `F`; F3d `pEGFR -> pEGFRi`, `Feedback`, `PI3Ka`, `mTORa`, `S6Ka`.
    receptor, akt, s6 = (
        ("EGF", "readout_pEGFR_tot"),
        ("readout_pEGFR_tot", "readout_pAkt_tot"),
        ("readout_pAkt_tot", "readout_pS6_tot"),
    )
    # The reference threads a complex through both downstream connections (pEGFR_Akt, pAkt_S6);
    # the candidates dropped both, which is part of what the text calls coarse-graining.
    latent = {
        "Published": {akt: "+ complex", s6: "+ complex"},
        "F2c": {receptor: "+ internal-\n   isation", akt: "+ feedback", s6: "+ mTOR,\n   S6K"},
        "F3d": {
            receptor: "+ internal-\n   isation",
            akt: "+ PI3K,\n   feedback",
            s6: "+ mTOR,\n   S6K",
        },
    }
    for axis, (name, shown, diagram, ink) in zip(maps, drawn, strict=True):
        readouts = {readout["id"]: readout["species"] for readout in diagram["readouts"]}
        sbgn.activity_flow(
            axis,
            shown,
            box=(2.30, 0.80),
            gap=map_gap,
            fontsize=base - 0.8,
            colors=accents,
            floor=paper_style.MIN_FONT,
            annotate=False,
            influence="positive influence",
            arrow_x=-0.32,
            notes=latent.get(name.split("\n")[0], {}),
            note_size=paper_style.MIN_FONT,
            arrow_scale=9,
            rounded=True,
            box_linewidth=1.0,
            classes={node["id"]: node["class"] for node in diagram["nodes"]},
            mechanics=sbgn.readout_mechanics(diagram, readouts),
        )
        axis.text(
            0.5,
            1.0,
            name,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=base - 0.6,
            fontweight="bold",
            color=ink,
        )
    render_topology_counts(topology, rows, metadata, fontsize=base)
    render_heldout_wins(heldout_wins, rows, metadata, fontsize=base)
    render_cost(cost, rows, metadata, fontsize=base)
    render_time(walltime, rows, metadata, fontsize=base)
    # The partial-Spearman panel that used to sit here is gone: correlating training
    # criteria against held-out ΔNLL answered a weaker question than simply counting how
    # often an arm PREDICTS better than the reference, which is what panel D now shows.
    # `correlations()` still computes the coefficients and `values.tex` still exports
    # them, so the numbers remain available to the text without spending a panel on them.

    extension = output.suffix.lower()
    stamps = {"CreationDate": None, "ModDate": None} if extension == ".pdf" else {}
    if extension == ".svg":
        stamps = {"Date": None}
    paper_style.check_fonts(fig)
    fig.savefig(output, dpi=paper_style.DPI, facecolor="white", metadata=stamps)
    plt.close(fig)


#: The process-description maps the supplementary figure draws, in panel order.
MODEL_MAPS = ("reference", "best_generalising", "runner_up")


def render_models(rows, metadata, output):
    """The process-description maps, side by side, as a figure of their own: the reference and
    the two candidates with the lowest held-out NLL.

    They came out of Figure 3 because a map that is worth reading needs room, more than a panel
    of a six-panel figure can be given. How to read them is the caption's to say, not a block of
    text under the drawing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gemot import paper_style, sbgn

    base = 6.4
    matplotlib.rcParams.update(
        {
            **paper_style.rc(),
            "font.size": base,
            "axes.titlesize": base + 1.4,
            "text.color": "#26343E",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "fujita-generalisation-models",
        }
    )
    page_width, margins = 8.27, {"left": 0.04, "right": 0.99}
    column = page_width * (margins["right"] - margins["left"])
    diagrams = [metadata["process_diagrams"][key] for key in MODEL_MAPS]
    anchors = [readout_anchors(metadata, diagram) for diagram in diagrams]
    grid, slots = sbgn.entity_rows(diagrams, anchors)
    geometry = {
        "entity": (1.42, 0.44),
        "dx": 2.05,
        "dy": 1.72,
        "flow": "vertical",
        "grid": grid,
        "slots": slots,
    }
    limits = [sbgn.process_description_limits(diagram, **geometry) for diagram in diagrams]
    along = (min(low for _x, (low, _high) in limits), max(high for _x, (_low, high) in limits))
    gap = 0.3
    width, tall = (column - gap * (len(diagrams) - 1)) / len(diagrams), along[1] - along[0]
    # Capped so the figure still fits a page; below the cap the maps are as large as the width
    # allows, above it the width goes unused rather than the figure running off the paper.
    body = min(9.15, min(width * tall / (high - low) for (low, high), _y in limits))
    headroom, footroom = 0.46, 0.14
    height = headroom + body + footroom
    margins |= {"top": 1 - headroom / height, "bottom": footroom / height}
    fig = plt.figure(figsize=(page_width, height), facecolor="white")
    cells = fig.add_gridspec(1, len(diagrams), wspace=gap / width, **margins)
    for index, (letter, diagram) in enumerate(zip("ABC", diagrams, strict=True)):
        axis = fig.add_subplot(cells[0, index])
        # Equal aspect shrinks a narrow map's box; anchoring it top-left keeps the heading over
        # the column rather than over whatever width the drawing happened to need.
        axis.set_anchor("NW")
        axis.set_title(
            f"{letter}   {diagram['label']}",
            loc="left",
            pad=13,
            fontweight="bold",
            fontsize=base + 1.4,
        )
        axis.text(
            0,
            1.0,
            " · ".join(
                f"{value} {name} {rank}/{total}"
                for name, value, rank, total in size_ranks(rows, diagram["counts"])
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=base - 1.2,
            color="#5F6B74",
        )
        sbgn.process_description(
            axis,
            diagram,
            fontsize=base - 1.8,
            along=along,
            curve=-0.16,
            pad=2.6,
            highlight=readout_highlight(metadata, diagram),
            **geometry,
        )
    extension = output.suffix.lower()
    stamps = {"CreationDate": None, "ModDate": None} if extension == ".pdf" else {}
    if extension == ".svg":
        stamps = {"Date": None}
    fig.savefig(output, dpi=FIGURE_DPI, facecolor="white", metadata=stamps)
    plt.close(fig)
