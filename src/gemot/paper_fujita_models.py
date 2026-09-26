"""Export the exact selected Fujita model sources for manuscript listings.

The archived Antimony and parameter table are separate artifacts. We preserve both;
we do not replace source initializations with fitted values or claim a standalone fit.
"""
from __future__ import annotations

import hashlib
import re

from petab_harness.paper_fujita import arms, csv_text, run_name


def annexed(metadata, rows):
    """The rows whose sources this annex ships.

    A run qualifies on its recorded hashes, so every listing stays verifiable: a parameter table,
    plus EITHER an archived ``model.ant`` (a version store) or an archived ``model.xml`` (a
    barebones run, which delivered a PEtab problem and never wrote Antimony). The second kind is
    listed as Antimony converted from that SBML -- see :func:`source_kind`.
    """
    runs = {run['task']: run for run in metadata['runs']}
    return [row for row in rows if source_kind(runs[row['task']]) is not None]


def source_kind(run):
    """``'archived'``, ``'converted'``, or ``None`` when nothing verifiable was recorded.

    ``converted`` means the listing is DERIVED: libantimony's rendering of the archived SBML,
    which is exported beside it so the hash in the manifest still verifies a real artifact. No
    Antimony for those models exists anywhere -- the agent delivered SBML.
    """
    digests = set(run.get('sha256') or {})
    if 'parameters.tsv' not in digests:
        return None
    if 'model.ant' in digests:
        return 'archived'
    return 'converted' if 'model.xml' in digests else None


def export_data(root, metadata, rows, header):
    runs = {run['task']: run for run in metadata['runs']}
    files, manifest = {}, []
    rows = annexed(metadata, rows)
    # Sources are filed under the name the FIGURE shows, so a listing and the point it came from
    # are the same string; the stored label stays in the manifest as the archive key.
    labels = [run_name(metadata, row['label']) for row in rows]
    if not rows or len(set(labels)) != len(labels) or any(
            not re.fullmatch(r'[OF][1-9][a-z]', label) for label in labels):
        raise ValueError('invalid or duplicate Fujita model listing labels')
    for row in rows:
        run, label = runs[row['task']], run_name(metadata, row['label'])
        entry = {key: row[key] for key in ('label', 'arm', 'task', 'head_version')}
        entry['name'] = label
        entry['archive_id'] = run['archive_id']
        entry['contains_reference_topology'] = row['contains_reference_topology']
        kind = source_kind(run)
        entry['antimony_source'] = kind
        # Uniform columns: a converted row records the SBML it came from, an archived row leaves
        # those cells empty rather than making the table ragged.
        entry['sbml_file'], entry['sbml_sha256'] = '', ''
        # EVERY listing ships as Antimony, including the eight whose agent delivered only SBML --
        # libantimony's rendering of the archived model.xml is readable and is what the annex
        # prints. The SBML is VERIFIED but not shipped: it is the artefact carrying the archived
        # hash for a derived listing (checking that listing against itself would prove nothing),
        # and shipping it too would put the same model in the annex twice in two languages.
        exported = [('parameters.tsv', '.parameters.tsv', 'parameters', True, True),
                    ('model.ant', '.ant', 'antimony', kind == 'archived', True)]
        if kind == 'converted':
            exported.append(('model.xml', '.model.xml', 'sbml', True, False))
        for source_name, suffix, field, verify, ship in exported:
            path = root / 'paper/fujita_models' / (label + suffix)
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if verify and digest != run['sha256'][source_name]:
                raise ValueError(f'{label}: {source_name} does not match its archived source hash')
            name = f'fujita_{label}{suffix}'
            if ship:
                # ASCII sources have no newline/encoding conversion in the text exporter.
                files[name] = content.decode('ascii')
            entry[field + '_file'] = name if ship else ''
            entry[field + '_sha256'] = digest
        manifest.append(entry)
    files['fujita_models.csv'] = csv_text(manifest)
    files['fujita_models.tex'] = listing_macros(rows, header, metadata)
    files['fujita_model_annex.tex'] = '\n'.join([
        header,
        '% Optional annex fragment; load fujita_models.tex and values.tex in the preamble.',
        '% Place after the manuscript appendix command; no counters are reset here.',
        r'\section{Agent-generated Fujita model sources}',
        r'\label{sec:fujita-model-sources}',
        r'The following listings contain the archived Antimony for the',
        r'\fujitaAnnexModelCount{} models of Figure~\ref{fig:fujita-generalisation} whose',
        r'sources were exported with their hashes.',
        r'The label, arm, array task and selected version identify each source.',
        r'Listings marked \emph{converted from the archived SBML} are libantimony renderings of'
        r' the \texttt{model.xml} those runs delivered: the bare-agent arms wrote no Antimony, so'
        r' none exists to reproduce. The hash recorded for such a row is their'
        r' \texttt{model.xml}, because verifying a derived listing against itself would prove'
        r' nothing; that SBML is not shipped, so the annex is Antimony throughout.'
        r' Every other listing is the archived file itself, preserved exactly,',
        r'including anonymous identifiers and unset initializations.',
        r'Numeric assignments in these source listings are not necessarily the fitted values.',
        r'The matching parameter tables (including observation scales and fixed entries) and',
        r'SHA-256 hashes are exported alongside them; see \texttt{fujita\_models.csv}.',
        r'These sources and tables document the selected structures and parameters;',
        r'reproducing simulations also requires the original PEtab conditions, observables',
        r'and initialization semantics. No parameters were refitted for this annex.',
        r'\fujitaAntimonyAll', '',
    ])
    return files


def listing_macros(rows, header, metadata):
    """Listings are verbatim, breakable and require no shell escape or external tool."""
    runs = {run['task']: run for run in metadata['runs']}
    lines = [header,
             '% Load once in the preamble. Model labels match Figure 3; paths are relative',
             '% to the manuscript root. Override fujitaModelDirectory after loading if needed.',
             r'\RequirePackage{listings}',
             r'\RequirePackage{xcolor}',
             # VS Code's Light+ token colours, which is the theme these sources were read in.
             # Light+ rather than Dark+ because a listing is printed on white: the dark theme's
             # pale blues and greens drop below print contrast on paper.
             r'\definecolor{antKeyword}{HTML}{0000FF}',
             r'\definecolor{antControl}{HTML}{AF00DB}',
             r'\definecolor{antComment}{HTML}{008000}',
             r'\definecolor{antString}{HTML}{A31515}',
             r'\definecolor{antFunction}{HTML}{795E26}',
             r'\definecolor{antLineNo}{HTML}{6E7681}',
             r'\definecolor{antFrame}{HTML}{D7DCE0}',
             # Operators and numbers keep the body colour, which is what Light+ gives them
             # anyway. Numbers could only be reached by a per-digit `literate`, and that would
             # repaint the 6 in S6K and the 3 in PIP3 as well.
             r'\lstdefinelanguage{FujitaAntimony}{',
             # One line per keyword class: a list wrapped across source lines puts a space in
             # front of the keyword after the break, and listings then never matches it.
             r'  morekeywords={model,end,function,species,compartment,var,const,formula,substanceOnly,unit,import,external,delete,is,has,in,at,after,encapsulating,extends},',
             r'  morekeywords=[2]{if,else,elseif,while,event},',
             r'  morekeywords=[3]{piecewise,abs,ceil,floor,exp,ln,log,log10,pow,sqrt,root,sin,cos,tan,arcsin,arccos,arctan,sinh,cosh,tanh,delay,max,min,sum,product,factorial,rateOf},',
             r'  sensitive=true,',
             r'  morecomment=[l]{//},morecomment=[s]{/*}{*/},morestring=[b]",',
             r'}',
             r'\lstdefinestyle{fujitaSource}{',
             r'  basicstyle=\ttfamily\footnotesize,columns=fullflexible,keepspaces=true,',
             r'  keywordstyle=\color{antKeyword},keywordstyle=[2]\color{antControl},',
             r'  keywordstyle=[3]\color{antFunction},',
             r'  commentstyle=\color{antComment}\itshape,stringstyle=\color{antString},',
             r'  numbers=left,numberstyle=\tiny\color{antLineNo},numbersep=6pt,stepnumber=1,',
             r'  breaklines=true,breakatwhitespace=false,showstringspaces=false,',
             r'  tabsize=4,frame=single,rulecolor=\color{antFrame},',
             r'  captionpos=t,linewidth=\linewidth}',
             r'\newcommand{\fujitaModelDirectory}{paper/generated}']
    for row in rows:
        task = row['task'].replace('_', r'\_')
        label = run_name(metadata, row['label'])
        origin = ('' if source_kind(runs[row['task']]) == 'archived'
                  else '; Antimony converted from the archived SBML')
        # A barebones run has no version store, and its head_version is the -1 sentinel for
        # that -- "selected version -1" reads as a bug to anyone but the person who wrote it.
        version = (f"selected version {row['head_version']}"
                   if isinstance(row['head_version'], int) and row['head_version'] >= 0
                   else 'delivered problem, no version store')
        caption = (f"{label}: {arms(metadata)[row['arm']]['label']}; "
                   f"task \\texttt{{{task}}}; {version}{origin}")
        lines.append(r'\expandafter\def\csname fujitaCaption' + label
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
            r'      {Use a Figure 3 label whose source this annex ships.}%',
            r'  \fi}',
        ])
    lines.append(r'\newcommand{\fujitaAntimonyAll}{%')
    lines.extend(r'  \fujitaAntimony{' + run_name(metadata, row['label']) + '}%'
                 for row in rows)
    lines.extend(['}', ''])
    return '\n'.join(lines)
