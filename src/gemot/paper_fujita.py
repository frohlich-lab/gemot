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

CORE_FEATURES = ('training_nll', 'n_parameters', 'n_reactions', 'n_states')
DEPTH_FEATURES = ('input_depth', 'upstream_depth', 'downstream_depth',
                  'input_to_akt_depth', 'latent_depth', 'full_depth')
FEATURES = CORE_FEATURES + DEPTH_FEATURES[:3]
LABELS = ('Training\nNLL', 'Parameters\n(scored k)', 'Reactions', 'States',
          'EGF\n→ pEGFR', 'pEGFR\n→ pAkt', 'pAkt\n→ pS6')
CONNECTIONS = (
    ('input_depth', 'EGF', 'pEGFR'), ('upstream_depth', 'pEGFR', 'pAkt'),
    ('downstream_depth', 'pAkt', 'pS6'), ('input_to_akt_depth', 'EGF', 'pAkt'),
    ('latent_depth', 'pEGFR', 'pS6'), ('full_depth', 'EGF', 'pS6'),
)
ARMS = {
    'baseline': {'label': 'Baseline', 'color': '#0072B2', 'marker': 'o', 'prefix': 'B'},
    'biocurator-off': {'label': 'Biocurator off', 'color': '#D55E00', 'marker': 's', 'prefix': 'C'},
    'biology-blind': {'label': 'Biology blinded', 'color': '#8C5BA6', 'marker': '^', 'prefix': 'X'},
}
POOLS = {
    'contains_reference_topology': {'arms': tuple(ARMS),
                         'label': 'Reference connections present', 'color': '#278278'},
}


def forward_readout_paths(graph, anchors):
    """Collapse hidden intermediates, while forbidding paths through another readout.

    Unlike unrestricted reachability, this distinguishes a serial chain from an input
    that independently drives several readouts. Edge length may differ from reference.
    """
    paths = {}
    for key, source, target in CONNECTIONS:
        starts, targets = anchors[source], anchors[target]
        blocked = set().union(*(nodes for name, nodes in anchors.items()
                                if name not in (source, target))) - starts - targets
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
    path = root / 'paper/fujita_generalisation.csv'
    provenance = path.with_suffix('.json')
    if not path.exists() and not provenance.exists():
        return None
    metadata = json.loads(provenance.read_text())
    if metadata.get('schema_version') != 1:
        raise ValueError('unsupported Fujita figure schema')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['source']['data_sha256']:
        raise ValueError('Fujita observations do not match their provenance hash')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    counts = ('head_version', 'conditions_ok', 'n_parameters', 'n_estimated',
              'n_charged_literals', 'n_reactions', 'n_states', 'n_species', *DEPTH_FEATURES)
    scores = ('training_dbic', 'training_nll', 'heldout_nll', 'heldout_delta_nll')
    for row in rows:
        for key in (*counts, *scores):
            row[key] = (int(row[key]) if key in counts else float(row[key])) if row[key] else None
        if row['complete_heldout'] not in ('True', 'False'):
            raise ValueError(f"{row['task']}: invalid completion flag")
        row['complete_heldout'] = row['complete_heldout'] == 'True'
        complete = row['conditions_ok'] == metadata['evaluation']['n_conditions']
        if complete != row['complete_heldout']:
            raise ValueError(f"{row['task']}: incomplete condition coverage")
        heldout = row['heldout_nll']
        delta = row['heldout_delta_nll']
        if complete:
            if (heldout is None or not math.isfinite(heldout) or delta is None
                    or not math.isclose(delta, heldout - metadata['reference']['heldout_nll'],
                                        rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"{row['task']}: invalid held-out delta")
        elif heldout is not None or delta is not None:
            raise ValueError(f"{row['task']}: incomplete integration must not have a score")
        if row['n_parameters'] != row['n_estimated'] + row['n_charged_literals']:
            raise ValueError(f"{row['task']}: parameter count disagrees with scoring k")
        row['undefined_depths'] = ','.join(key for key in DEPTH_FEATURES if row[key] is None)
    tasks = [row['task'] for row in rows]
    if (not tasks or len(set(tasks)) != len(tasks)
            or set(tasks) != {run['task'] for run in metadata['runs']}):
        raise ValueError('Fujita task IDs must uniquely match the provenance manifest')
    if {row['arm'] for row in rows} != set(ARMS):
        raise ValueError('Fujita observations must cover all three arms')
    read_topology(root, metadata, rows)
    return metadata, rows


def read_topology(root, metadata, rows):
    """Join the archived path audit and classify the forward readout graph."""
    path = root / 'paper/fujita_topology.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['source']['topology_data_sha256']:
        raise ValueError('Fujita topology paths do not match their provenance hash')
    with path.open(newline='') as handle:
        audited = list(csv.DictReader(handle))
    lookup = {row['task']: row for row in audited}
    if len(lookup) != len(audited) or set(lookup) != {row['task'] for row in rows}:
        raise ValueError('Fujita topology audit must cover each selected model once')
    expected = {key for key, path in metadata['topology_definition']['reference_paths'].items()
                if path}
    for row in rows:
        paths = {key: lookup[row['task']][key] for key, _, _ in CONNECTIONS}
        present = {key for key, path in paths.items() if path}
        anchors = {'EGF': {row['stimulus_anchor']},
                   'pEGFR': set(row['receptor_states'].split(',')),
                   'pAkt': set(row['akt_states'].split(',')),
                   'pS6': set(row['s6_states'].split(','))}
        overlapping = any(anchors[source] & anchors[target] for _, source, target in CONNECTIONS)
        for key, source, target in CONNECTIONS:
            if not paths[key]:
                continue
            chain = paths[key].split(' -> ')
            other_readouts = set().union(*(nodes for name, nodes in anchors.items()
                                           if name not in (source, target)))
            if (chain[0] not in anchors[source] or chain[-1] not in anchors[target]
                    or set(chain[1:-1]) & other_readouts):
                raise ValueError(f"{row['task']}: invalid collapsed readout path {key}")
        row['topology_paths'] = paths
        row['contains_reference_topology'] = expected <= present and not overlapping
        row['exact_reference_topology'] = present == expected and not overlapping
        row['additional_forward_connections'] = ','.join(sorted(present - expected))
        row['topology_reason'] = '; '.join(filter(None, (
            'overlapping readout anchors' if overlapping else '',
            'missing ' + ','.join(sorted(expected - present)) if expected - present else '',
        )))


def topology_counts(rows):
    """Every selected model counts, independently of held-out integration success."""
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row['arm'] == arm]
        matched = [row for row in selected if row['contains_reference_topology']]
        output.append({'arm': arm, 'n_models': len(selected), 'n_retained': len(matched),
                       'retained_tasks': ','.join(row['task'] for row in matched),
                       'excluded_tasks': ','.join(row['task'] for row in selected
                                                  if not row['contains_reference_topology'])})
    return output


def csv_text(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
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
    threshold = config['linthresh']
    linear = config['linscale'] / (1 - 1.0 / config['base'])
    log_base = np.log(config['base'])
    absolute = np.abs(values)
    outside = absolute > threshold
    transformed = values * linear
    transformed[outside] = np.sign(values[outside]) * threshold * (
        linear - np.log(threshold) / log_base + np.log(absolute[outside]) / log_base)
    return transformed


def macros(metadata, rows):
    """Caption values share the plotted input, rather than being transcribed into LaTeX."""
    counts = [sum(row['arm'] == arm for row in rows) for arm in ARMS]
    if len(set(counts)) != 1:
        raise ValueError('the Figure 3 caption requires equal arm sizes')
    evaluation = metadata['evaluation']
    values = {
        'fujitaFigureRunCount': str(len(rows)),
        'fujitaFigureRunsPerArm': str(counts[0]),
        'fujitaFigureConditionCount': str(evaluation['n_conditions']),
        'fujitaFigureMeasurementCount': str(evaluation['n_measurements']),
        'fujitaFigureReferenceNLL': f"{metadata['reference']['heldout_nll']:.2f}",
        'fujitaFigureAbsoluteTolerance': f"\\ensuremath{{{evaluation['atol']:g}}}",
        'fujitaFigureRelativeTolerance': f"\\ensuremath{{{evaluation['rtol']:g}}}",
        'fujitaFigureSymlogBase': str(metadata['symlog']['base']),
        'fujitaFigureSymlogThreshold': str(metadata['symlog']['linthresh']),
        'fujitaFigureSymlogLinearScale': str(metadata['symlog']['linscale']),
        'fujitaFigureReferenceTopologyCount': str(sum(row['contains_reference_topology'] for row in rows)),
        'fujitaFigureTopologyScoredCount': str(sum(row['contains_reference_topology']
                                                  and row['complete_heldout'] for row in rows)),
    }
    for entry in topology_counts(rows):
        arm = ''.join(word.capitalize() for word in entry['arm'].split('-'))
        values[f'fujitaFigure{arm}ReferenceTopology'] = str(entry['n_retained'])
    coefficients, _ = correlations(rows, metadata['symlog'])
    feature_names = dict(zip(FEATURES, ('TrainingNLL', 'Parameters', 'Reactions', 'States',
                                       'InputDepth', 'UpstreamDepth', 'DownstreamDepth'), strict=True))
    for coefficient in coefficients:
        prefix = 'fujitaFigure' + feature_names[coefficient['feature']]
        values[prefix + 'Spearman'] = f"{coefficient['partial_spearman']:.2f}"
        values[prefix + 'PearsonSymlog'] = f"{coefficient['partial_pearson_symlog']:.2f}"
    for feature, label in (('training_nll', 'Core'), ('downstream_depth', 'Depth')):
        coefficient = next(row for row in coefficients if row['feature'] == feature)
        values[f'fujitaFigure{label}CorrelationDF'] = str(coefficient['residual_correlation_df'])
        values[f'fujitaFigure{label}CorrelationControls'] = str(coefficient['n_covariates'])
    for row in rows:
        index = ('One', 'Two', 'Three', 'Four', 'Five')[int(row['label'][1]) - 1]
        prefix = 'fujitaModel' + row['label'][0] + index
        values[prefix + 'TrainingNLL'] = f"{row['training_nll']:.2f}"
        values[prefix + 'HeldoutNLL'] = (f"{row['heldout_nll']:.2f}" if row['complete_heldout']
                                       else 'unscored')
        values[prefix + 'DownstreamDepth'] = str(row['downstream_depth'])
    for diagnostic in correlation_sensitivity(rows, metadata['symlog']):
        if diagnostic['feature'] == 'downstream_depth' and diagnostic['omitted_label'] == 'B3':
            prefix = 'fujitaFigureDownstreamWithoutBThree'
            for field, suffix in (('partial_spearman', 'Spearman'),
                                  ('partial_pearson_symlog', 'PearsonSymlog')):
                values[prefix + suffix] = (f"{float(diagnostic[field]):.2f}"
                                          if diagnostic[field] else 'undefined')
    return values


def figure_tex(header):
    return '\n'.join([
        header,
        '% Figure 3: input as the third manuscript figure; LaTeX owns the counter.',
        '% Requires graphicx and paper/generated/values.tex in the preamble.',
        r'\begin{figure*}[p]',
        r'\centering',
        r'\includegraphics[width=\textwidth]{paper/generated/figure3.pdf}',
        r'\caption{\textbf{Training fit, model structure and held-out prediction in Fujita.}',
        r'\fujitaFigureRunCount{} final models (\fujitaFigureRunsPerArm{} per arm) compare',
        r'baseline, biocurator off with biology visible, and full biology blinding.',
        r'\textbf{A,} Recorded training $\Delta$BIC versus held-out $\Delta$NLL;',
        r'both are candidate minus reference, so lower is better. Fixed archived parameters',
        r'are evaluated on \fujitaFigureMeasurementCount{} measurements in',
        r'\fujitaFigureConditionCount{} conditions at absolute tolerance',
        r'\fujitaFigureAbsoluteTolerance{} and relative tolerance \fujitaFigureRelativeTolerance{}.',
        r'The stored reference NLL is \fujitaFigureReferenceNLL{}; its fitted vector was',
        r'unavailable for rescoring. Incomplete integrations occupy a separate unscored strip.',
        r'\textbf{B,} Adjacent forward paths between the input and readout nodes in the',
        r'published reference model. Nodes list the species contributing to each readout;',
        r'arrows give shortest influence-graph hop counts, with intervening species named.',
        r'This shows reference topology independently of its fitted parameter values.',
        r'\textbf{C,} Number of models in each arm recovering the reference forward readout',
        r'connections, allowing different hidden-path lengths and additional forward paths.',
        r'Each required path may not pass through another readout. This unsigned criterion',
        r'does not require the reference edge signs or a matching full mechanism.',
        r'All final models enter these counts, including those with incomplete held-out scores.',
        r'\textbf{D,} Partial Spearman correlations after ranking all variables.',
        r'\textbf{E,} Partial Pearson correlations with original-unit predictors and controls,',
        r'and held-out $\Delta$NLL transformed by the exact symmetric-log scale in A',
        r'(base \fujitaFigureSymlogBase{}, linear threshold \fujitaFigureSymlogThreshold{},',
        r'linear scale \fujitaFigureSymlogLinearScale{}), before residualization.',
        r'The correlation panels pool only the \fujitaFigureTopologyScoredCount{} models',
        r'containing all reference connections and having complete held-out scores. They show only',
        r'adjacent input/readout depths and use identical complete cases and controls.',
        r'Each core feature (training NLL, scored parameter count, reactions, states)',
        r'adjusts for the other core features; each adjacent depth adjusts',
        r'for all core features. Depths count shortest directed influence-graph hops',
        r'between implemented input/readout anchors, not exclusively unobserved intermediates.',
        r'Hollow points ($\dagger$) lack a required forward connection and are excluded',
        r'from correlations. The filtered pool is not additionally adjusted for arm membership.',
        r'Counts below bars are feature-specific. Positive correlations indicate worse',
        r'prediction; these small-sample estimates are exploratory.}',
        r'\label{fig:fujita-generalisation}',
        r'\end{figure*}', '',
    ])


def export_data(root, metadata, rows, header):
    coefficients, pools = correlations(rows, metadata['symlog'])
    output = io.StringIO(newline='')
    fields = [*coefficients[0], 'tasks_used', 'excluded_tasks']
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in coefficients:
        sample = pools[row['pool']]['features'][row['feature']]
        # Drop platform-dependent final floating-point bits from deterministic text exports.
        rounded = {key: f'{value:.12f}' if isinstance(value, float) else value
                   for key, value in row.items()}
        writer.writerow({**rounded, 'tasks_used': ','.join(sample['tasks_used']),
                         'excluded_tasks': '; '.join(f'{task}: {reason}' for task, reason
                                                     in sample['excluded'].items())})
    topology = [{'task': row['task'], 'arm': row['arm'], 'label': row['label'],
                 'contains_reference_topology': row['contains_reference_topology'],
                 'exact_reference_topology': row['exact_reference_topology'],
                 'additional_forward_connections': row['additional_forward_connections'],
                 'exclusion_reason': row['topology_reason'], **row['topology_paths']} for row in rows]
    return {
        'figure3.tex': figure_tex(header),
        'fujita_generalisation.csv': (root / 'paper/fujita_generalisation.csv').read_text(),
        'fujita_generalisation.json': (root / 'paper/fujita_generalisation.json').read_text(),
        'fujita_correlations.csv': output.getvalue(),
        'fujita_correlation_sensitivity.csv': csv_text(correlation_sensitivity(rows, metadata['symlog'])),
        'fujita_topology.csv': csv_text(topology),
        'fujita_topology_counts.csv': csv_text(topology_counts(rows)),
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
        raise ValueError('insufficient residual degrees of freedom')
    if not (scale > 0).all():
        raise ValueError('constant variable after omission')
    scaled = (matrix - matrix.mean(axis=0)) / scale
    design = np.column_stack([np.ones(n), scaled[:, 1:-1]])
    design_rank = int(np.linalg.matrix_rank(design))
    full_rank = int(np.linalg.matrix_rank(scaled))
    if design_rank != columns - 1:
        raise ValueError('collinear controls')
    x, y = scaled[:, 0], scaled[:, -1]
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.linalg.norm(rx), np.linalg.norm(ry)) <= 1e-9:
        raise ValueError('zero residual variance')
    value = float(np.corrcoef(rx, ry)[0, 1])
    if full_rank == columns:
        precision = np.linalg.inv(np.corrcoef(scaled.T))
        expected = -precision[0, -1] / np.sqrt(precision[0, 0] * precision[-1, -1])
        assert np.isclose(value, expected, atol=1e-10)
    return value, design_rank, full_rank


def correlations(rows, symlog):
    import numpy as np
    from scipy.stats import rankdata

    results, pools = [], {}
    for pool, config in POOLS.items():
        selected = [row for row in rows if row['arm'] in config['arms']]
        pools[pool] = {'n_total': len(selected),
                       'n_topology': sum(row['contains_reference_topology'] for row in selected),
                       'n_finite': sum(row['complete_heldout'] and row['contains_reference_topology']
                                       for row in selected),
                       'features': {}}
        for feature in FEATURES:
            controls = ([key for key in CORE_FEATURES if key != feature]
                        if feature in CORE_FEATURES else list(CORE_FEATURES))
            required = [feature, *controls]
            eligible = [row for row in selected if row['contains_reference_topology'] and row['complete_heldout']
                        and all(row[key] is not None for key in required)]
            pools[pool]['features'][feature] = {
                'n_used': len(eligible), 'tasks_used': [row['task'] for row in eligible],
                'excluded': {row['task']: ('missing reference connection: ' + row['topology_reason']
                             if not row['contains_reference_topology'] else 'incomplete held-out integration'
                             if not row['complete_heldout'] else 'undefined ' +
                             ','.join(key for key in required if row[key] is None))
                             for row in selected if row not in eligible}}
            matrix = np.array([[row[key] for key in (*required, 'heldout_delta_nll')]
                               for row in eligible], dtype=float)
            assert np.isfinite(matrix).all()
            ranked = np.column_stack([rankdata(column, method='average') for column in matrix.T])
            rho, design_rank, full_rank = partial_pearson(ranked)
            transformed = matrix.copy()
            transformed[:, -1] = symlog_transform(matrix[:, -1], symlog)
            pearson, pearson_rank, pearson_full_rank = partial_pearson(transformed)
            n = len(eligible)
            results.append({'pool': pool, 'feature': feature, 'partial_spearman': rho,
                            'partial_pearson_symlog': pearson,
                            'n': n, 'n_covariates': len(controls),
                            'residual_correlation_df': n - len(controls) - 2,
                            'design_rank': design_rank, 'full_rank': full_rank,
                            'pearson_design_rank': pearson_rank,
                            'pearson_full_rank': pearson_full_rank,
                            'controlled_features': ','.join(controls)})
    return results, pools


def correlation_sensitivity(rows, symlog):
    """Delete each eligible model once, retaining the figure's controls and transforms.

    These diagnostics describe leverage; the smaller samples are not new estimates
    to select from. Undefined coefficients are explicit, including the input-depth
    predictor becoming constant when its only two-hop model is omitted.
    """
    import numpy as np
    from scipy.stats import rankdata

    coefficients, pools = correlations(rows, symlog)
    lookup = {row['task']: row for row in rows}
    output = []
    for coefficient in coefficients:
        feature, pool = coefficient['feature'], coefficient['pool']
        tasks = pools[pool]['features'][feature]['tasks_used']
        controls = coefficient['controlled_features'].split(',')
        keys = [feature, *controls, 'heldout_delta_nll']
        for omitted in tasks:
            sample = [task for task in tasks if task != omitted]
            matrix = np.array([[lookup[task][key] for key in keys] for task in sample])
            ranked = np.column_stack([rankdata(column, method='average') for column in matrix.T])
            transformed = matrix.copy()
            transformed[:, -1] = symlog_transform(matrix[:, -1], symlog)
            record = {'pool': pool, 'feature': feature,
                      'omitted_label': lookup[omitted]['label'], 'omitted_task': omitted,
                      'n': len(sample), 'n_covariates': len(controls),
                      'residual_correlation_df': len(sample) - len(controls) - 2,
                      'controlled_features': ','.join(controls),
                      'tasks_used': ','.join(sample)}
            for name, values in (('partial_spearman', ranked),
                                 ('partial_pearson_symlog', transformed)):
                try:
                    value, _, _ = partial_pearson(values)
                except ValueError as error:
                    record[name], record[name + '_status'] = '', str(error)
                else:
                    record[name], record[name + '_status'] = f'{value:.12f}', 'ok'
            output.append(record)
    return output


def render_reference_network(axis, network):
    """Readout-level forward paths, with observable species sums kept explicit."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis('off')
    positions = {node['id']: 0.91 - index * 0.26
                 for index, node in enumerate(network['nodes'])}
    for node in network['nodes']:
        y = positions[node['id']]
        is_input = node['kind'] == 'input'
        color = '#3E7863' if is_input else '#486B88'
        axis.add_patch(FancyBboxPatch((0.035, y - 0.068), 0.93, 0.136,
                                     boxstyle='round,pad=0.012,rounding_size=0.025',
                                     edgecolor=color, facecolor='#EEF6F2' if is_input
                                     else '#EDF3F8', linewidth=1.3))
        axis.text(0.5, y + 0.023, node['label'], ha='center', va='center',
                  fontsize=12, fontweight='bold', color=color)
        formula = 'Input' if is_input else ' + '.join(node['species'])
        axis.text(0.5, y - 0.028, formula, ha='center', va='center', fontsize=8.4)
    for edge in network['edges']:
        start, end = positions[edge['source']], positions[edge['target']]
        axis.add_patch(FancyArrowPatch((0.5, start - 0.085), (0.5, end + 0.085),
                                      arrowstyle='-|>', mutation_scale=15,
                                      linewidth=1.5, color='#697A86'))
        label = f"{edge['hops']} hop" + ('s' if edge['hops'] != 1 else '')
        if edge['path'][1:-1]:
            label += '\nvia ' + ', '.join(edge['path'][1:-1])
        axis.text(0.57, (start + end) / 2, label, ha='left', va='center', fontsize=8.7)
    axis.text(0.5, -0.10, 'Reference topology: adjacent forward paths\n'
              'Readout nodes list their contributing species.', ha='center', va='top',
              fontsize=8.4, transform=axis.transAxes)


def render_topology_counts(axis, rows):
    counts = topology_counts(rows)
    colors = [ARMS[row['arm']]['color'] for row in counts]
    axis.bar(range(len(counts)), [row['n_retained'] for row in counts], width=0.62,
             color=colors, edgecolor=colors, linewidth=1.2, zorder=3)
    for x, row in enumerate(counts):
        axis.text(x, row['n_retained'] + 0.15, f"{row['n_retained']}/{row['n_models']}",
                  ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[x])
    maximum = max(row['n_models'] for row in counts)
    axis.set(ylim=(0, maximum + 0.7), yticks=range(maximum + 1),
             ylabel='Models containing reference connections')
    axis.set_xticks(range(len(counts)), ['Baseline', 'Biocurator\noff', 'Biology\nblinded'],
                    fontsize=9)
    axis.grid(axis='y', color='#E2E7EA', linewidth=0.7)
    axis.text(0.5, -0.16, 'All final models per arm\nIndependent of held-out integration success',
              ha='center', va='top', fontsize=8.4, transform=axis.transAxes)


def render(rows, coefficients, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    symlog = metadata['symlog']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    from scipy.stats import rankdata

    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                               'axes.labelsize': 11.5, 'axes.titlesize': 12.5,
                               'axes.spines.top': False, 'axes.spines.right': False,
                               'axes.edgecolor': '#69737B', 'axes.labelcolor': '#26343E',
                               'xtick.color': '#43515C', 'ytick.color': '#43515C',
                               'text.color': '#26343E', 'pdf.fonttype': 42,
                               'ps.fonttype': 42, 'svg.fonttype': 'none',
                               'svg.hashsalt': 'fujita-generalisation-figure3'})
    fig = plt.figure(figsize=(28, 8), facecolor='white')
    grid = fig.add_gridspec(2, 5, width_ratios=(1.1, 0.72, 0.7, 1.18, 1.18), height_ratios=(0.55, 6),
                           left=0.05, right=0.99, top=0.86, bottom=0.31,
                           wspace=0.24, hspace=0.045)
    failed = fig.add_subplot(grid[0, 0])
    scatter = fig.add_subplot(grid[1, 0], sharex=failed)
    network = fig.add_subplot(grid[:, 1])
    topology = fig.add_subplot(grid[:, 2])
    bars = fig.add_subplot(grid[:, 3])
    pearson_bars = fig.add_subplot(grid[:, 4], sharey=bars)
    fig.suptitle('Fujita: training fit, model structure and held-out prediction',
                 x=0.05, y=0.975, ha='left', fontsize=17, fontweight='bold')
    failed.set_title('A   Training ΔBIC vs held-out ΔNLL', loc='left', pad=14,
                     fontweight='bold')
    network.set_title('B   Reference readout network', loc='left', pad=14, fontweight='bold')
    render_reference_network(network, metadata['reference_network'])
    topology.set_title('C   Topology recovery', loc='left', pad=14, fontweight='bold')
    render_topology_counts(topology, rows)
    bars.set_title('D   Partial Spearman: rank scale', loc='left', pad=14,
                   fontweight='bold')
    pearson_bars.set_title('E   Partial Pearson: symlog scale from A', loc='left', pad=14,
                           fontweight='bold')
    scatter.set_xlim(-625, 65)
    scatter.set_yscale('symlog', **symlog)
    values = np.array([row['heldout_delta_nll'] for row in rows if row['complete_heldout']])
    # Matplotlib versions use algebraically identical log(a/b) or log(a)-log(b).
    # Allow their floating-point rounding difference, not a different scale.
    np.testing.assert_allclose(scatter.yaxis.get_transform().transform(values),
                               symlog_transform(values, symlog), rtol=1e-14, atol=1e-12)
    assert np.array_equal(rankdata(values),
                          rankdata(scatter.yaxis.get_transform().transform(values)))
    scatter.set_ylim(-4900, 410000)
    scatter.set_yticks([-3000, -1000, 0, 1000, 10000, 100000])
    scatter.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    scatter.grid(axis='y', color='#E2E7EA', linewidth=0.7)
    scatter.axhline(0, color='#526370', linestyle='--', linewidth=1.0, zorder=2)
    scatter.axvline(0, color='#C5CDD2', linestyle=':', linewidth=1)
    scatter.axhspan(-4900, 0, color='#EEF6F2', zorder=0)
    scatter.annotate('Better than reference', (-620, 0), xytext=(4, -14),
                     textcoords='offset points', ha='left', fontsize=9, color='#3E7863')
    scatter.set_xlabel('Training ΔBIC  (candidate − reference)')
    scatter.set_ylabel('Held-out ΔNLL  (candidate − reference)\nSymmetric log scale')
    failed.set_facecolor('#F2F4F6')
    failed.set_ylim(0, 1)
    failed.set_yticks([])
    failed.tick_params(axis='x', bottom=False, labelbottom=False)
    for spine in failed.spines.values():
        spine.set_visible(False)
    failed.text(0.012, 0.5, 'No complete score', transform=failed.transAxes,
                va='center', fontsize=9, color='#5F6B74')
    offsets = {'B1': (5, 6), 'B2': (6, 6), 'B3': (6, 5), 'B4': (-20, 7),
               'B5': (6, 7), 'C1': (6, 6), 'C2': (6, -12), 'C4': (6, 5),
               'C5': (6, 5), 'X2': (6, 5), 'X3': (6, 5), 'X4': (6, 5), 'X5': (-24, 6)}
    for arm, style in ARMS.items():
        subset = [row for row in rows if row['arm'] == arm and row['complete_heldout']]
        scatter.scatter([row['training_dbic'] for row in subset],
                        [row['heldout_delta_nll'] for row in subset],
                        s=70, c=style['color'], marker=style['marker'],
                        edgecolors='white', linewidths=0.75, label=style['label'], zorder=4)
        for row in subset:
            if not row['contains_reference_topology']:
                scatter.scatter(row['training_dbic'], row['heldout_delta_nll'],
                                s=70, marker=style['marker'], facecolors='white',
                                edgecolors=style['color'], linewidths=1.3, zorder=5)
            label = row['label'] + ('†' if not row['contains_reference_topology'] else '')
            scatter.annotate(label, (row['training_dbic'], row['heldout_delta_nll']),
                             xytext=offsets.get(row['label'], (6, 5)), textcoords='offset points',
                             fontsize=8.2, color=style['color'])
        for row in rows:
            if row['arm'] == arm and not row['complete_heldout']:
                failed.scatter(row['training_dbic'], 0.52, marker='x', s=48,
                               color=style['color'], linewidths=1.7)
                failed.annotate(row['label'], (row['training_dbic'], 0.52),
                                xytext=(7, -3), textcoords='offset points',
                                color=style['color'], fontsize=8.5)
    scatter.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3,
                   frameon=False, fontsize=9.4, handletextpad=0.35, columnspacing=1.2)
    positions = np.arange(len(FEATURES))
    width = 0.55
    for axis, metric, ylabel in ((bars, 'partial_spearman', 'Partial Spearman ρ'),
                                (pearson_bars, 'partial_pearson_symlog', 'Partial Pearson r')):
        for j, (pool, config) in enumerate(POOLS.items()):
            subset = {row['feature']: row for row in coefficients if row['pool'] == pool}
            values = [subset[feature][metric] for feature in FEATURES]
            x = positions + (j - (len(POOLS) - 1) / 2) * (width + 0.035)
            axis.bar(x, values, width=width, color=config['color'], zorder=3,
                     label=config['label'])
            for at, value, feature in zip(x, values, FEATURES, strict=True):
                axis.text(at, value + (0.04 if value >= 0 else -0.04), f'{value:+.2f}',
                          ha='center', va='bottom' if value >= 0 else 'top', fontsize=9.2,
                          color=config['color'])
                axis.text(at, -1.105, f"n={subset[feature]['n']}", ha='center', va='center',
                          fontsize=7.8, color=config['color'])
        axis.axhline(0, color='#526370', linewidth=1)
        axis.axvline(3.5, color='#BAC5CB', linestyle=':', linewidth=1)
        axis.axhspan(-1.19, -1.015, color='#F2F4F6', zorder=0)
        axis.set_ylim(-1.19, 1.10)
        axis.set_xlim(-0.65, len(FEATURES) - 0.35)
        axis.set_yticks([-1, -0.5, 0, 0.5, 1])
        axis.set_xticks(positions, LABELS, fontsize=9)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', color='#E2E7EA', linewidth=0.7)
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
                    frameon=False, fontsize=10, columnspacing=1.5)
    fig.text(0.05, 0.175,
             'C counts all final models. D/E pool models containing the reference connections with complete '
             'held-out scores; positive correlations associate with worse prediction.', fontsize=9.5)
    fig.text(0.05, 0.145,
             'D ranks all variables. E uses original-unit predictors/controls and symlog(held-out ΔNLL): '
             f"the exact transform from A (base {symlog['base']}, linear threshold "
             f"{symlog['linthresh']}, linear scale {symlog['linscale']}).", fontsize=9.5)
    fig.text(0.05, 0.115,
             'Core features: adjusted for the other three core features. '
             'Each depth: adjusted for training NLL, k, reactions and states.', fontsize=9.5)
    fig.text(0.05, 0.085,
             f"Fixed parameters; atol = {metadata['evaluation']['atol']:g}, "
             f"rtol = {metadata['evaluation']['rtol']:g}. ΔNLL uses the stored reference "
             f"({metadata['reference']['heldout_nll']:.2f}). "
             'Depths are shortest paths in the SBML influence graph.', fontsize=9.5)
    fig.text(0.05, 0.055,
             '† A reference connection is missing: hollow points are excluded from D/E. Additional paths and '
             'different hidden-path lengths are allowed; edge signs are not tested. Estimates are exploratory.', fontsize=9.5)
    extension = output.suffix.lower()
    stamps = {'CreationDate': None, 'ModDate': None} if extension == '.pdf' else {}
    if extension == '.svg':
        stamps = {'Date': None}
    fig.savefig(output, dpi=300, facecolor='white', metadata=stamps)
    plt.close(fig)
