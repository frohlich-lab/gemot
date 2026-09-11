"""Export the exact selected Fujita model sources for manuscript listings.

The archived Antimony and parameter table are separate artifacts. We preserve both;
we do not replace source initializations with fitted values or claim a standalone fit.
"""
from __future__ import annotations

import hashlib
import re

from petab_harness.paper_fujita import ARMS, csv_text


def export_data(root, metadata, rows, header):
    runs = {run['task']: run for run in metadata['runs']}
    files, manifest = {}, []
    labels = [row['label'] for row in rows]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r'[BCX][1-5]', label)
                                            for label in labels):
        raise ValueError('invalid or duplicate Fujita model listing labels')
    for row in rows:
        run, label = runs[row['task']], row['label']
        entry = {key: row[key] for key in ('label', 'arm', 'task', 'head_version')}
        entry['archive_id'] = run['archive_id']
        entry['contains_reference_topology'] = row['contains_reference_topology']
        for source_name, suffix, kind in (('model.ant', '.ant', 'antimony'),
                                          ('parameters.tsv', '.parameters.tsv', 'parameters')):
            path = root / 'paper/fujita_models' / (label + suffix)
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != run['sha256'][source_name]:
                raise ValueError(f'{label}: {source_name} does not match its archived source hash')
            name = f'fujita_{label}{suffix}'
            # ASCII sources have no newline/encoding conversion in the text exporter.
            files[name] = content.decode('ascii')
            entry[kind + '_file'], entry[kind + '_sha256'] = name, digest
        manifest.append(entry)
    files['fujita_models.csv'] = csv_text(manifest)
    files['fujita_models.tex'] = listing_macros(rows, header)
    files['fujita_model_annex.tex'] = '\n'.join([
        header,
        '% Optional annex fragment; load fujita_models.tex and values.tex in the preamble.',
        '% Place after the manuscript appendix command; no counters are reset here.',
        r'\section{Agent-generated Fujita model sources}',
        r'\label{sec:fujita-model-sources}',
        r'The following listings contain the archived Antimony for all',
        r'\fujitaFigureRunCount{} final models in Figure~\ref{fig:fujita-generalisation}.',
        r'The label, arm, array task and selected version identify each source.',
        r'Files are preserved exactly, including anonymous identifiers and unset initializations.',
        r'Numeric assignments in these source listings are not necessarily the fitted values.',
        r'The matching parameter tables (including observation scales and fixed entries) and',
        r'SHA-256 hashes are exported alongside them; see \texttt{fujita\_models.csv}.',
        r'These sources and tables document the selected structures and parameters;',
        r'reproducing simulations also requires the original PEtab conditions, observables',
        r'and initialization semantics. No parameters were refitted for this annex.',
        r'\fujitaAntimonyAll', '',
    ])
    return files


def listing_macros(rows, header):
    """Listings are verbatim, breakable and require no shell escape or external tool."""
    lines = [header,
             '% Load once in the preamble. Model labels match Figure 3; paths are relative',
             '% to the manuscript root. Override fujitaModelDirectory after loading if needed.',
             r'\RequirePackage{listings}',
             r'\lstdefinelanguage{FujitaAntimony}{',
             r'  morekeywords={model,end,compartment,species,var,const,function},',
             r'  sensitive=true,morecomment=[l]{//},morestring=[b]"}',
             r'\lstdefinestyle{fujitaSource}{',
             r'  basicstyle=\ttfamily\footnotesize,columns=fullflexible,keepspaces=true,',
             r'  numbers=left,numberstyle=\tiny,numbersep=6pt,stepnumber=1,',
             r'  breaklines=true,breakatwhitespace=false,showstringspaces=false,',
             r'  tabsize=4,frame=single,captionpos=t,linewidth=\linewidth}',
             r'\newcommand{\fujitaModelDirectory}{paper/generated}']
    for row in rows:
        task = row['task'].replace('_', r'\_')
        caption = (f"{row['label']}: {ARMS[row['arm']]['label']}; "
                   f"task \\texttt{{{task}}}; selected version {row['head_version']}")
        lines.append(r'\expandafter\def\csname fujitaCaption' + row['label']
                     + r'\endcsname{' + caption + '}')
    for macro, suffix, description, language, prefix in (
            ('fujitaAntimony', '.ant', 'Archived Antimony', 'FujitaAntimony', 'antimony'),
            ('fujitaParameters', '.parameters.tsv', 'Archived parameter table', '', 'parameters')):
        lines.extend([
            '\\newcommand{\\' + macro + r'}[2][]{%',
            r'  \ifcsname fujitaCaption#2\endcsname',
            r'    \lstinputlisting[style=fujitaSource,language={' + language + '},',
            r'      caption={' + description + r' for \csname fujitaCaption#2\endcsname},',
            r'      label={lst:fujita-' + prefix + r'-#2},#1]',
            r'      {\fujitaModelDirectory/fujita_#2' + suffix + '}%',
            r'  \else',
            r"    \PackageError{fujita-models}{Unknown model label '#2'}",
            r'      {Use a Figure 3 label: B1--B5, C1--C5 or X1--X5.}%',
            r'  \fi}',
        ])
    lines.append(r'\newcommand{\fujitaAntimonyAll}{%')
    lines.extend(r'  \fujitaAntimony{' + row['label'] + '}%' for row in rows)
    lines.extend(['}', ''])
    return '\n'.join(lines)
