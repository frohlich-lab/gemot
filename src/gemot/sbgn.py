"""SBGN renderers for the manuscript figures: Activity Flow and Process Description.

Both draw from PORTABLE descriptions — node and arc lists produced by
``slurm/_fujita_figure_input.py`` — so a figure never loads or simulates a model.

Glyphs follow the SBGN specifications (sbgn.github.io):

* **Activity Flow.** A biological activity is a plain RECTANGLE carrying a unit of information on
  its top border whose SHAPE gives the entity class: rounded rectangle for a macromolecule, a
  hexagon with concave faces for a perturbation. The unit's text is abbreviated (``prot``,
  ``pert``) because at manuscript size the full SBGN spelling does not fit the glyph. Modulation arcs are distinguished by their
  terminator — positive influence an unfilled arrowhead, negative influence a bar, unknown
  influence an unfilled diamond, necessary stimulation a bar plus the unfilled arrowhead.
* **Process Description.** Entity pools are rounded rectangles (macromolecule), cut-corner
  rectangles (complex) and concave hexagons (perturbation); a process is a small square. A
  consumption arc is a plain line into the process, a production arc a line out of it with a
  FILLED arrowhead, and a modulation arc carries the same terminators as above. A reversible
  process takes production arrowheads on both sides.

Only the glyphs these figures use are implemented; anything else raises rather than guessing.
"""

from __future__ import annotations

import math

INK = "#26343E"
MUTED = "#5F6B74"
#: Entity-class fills. Deliberately pale: SBGN's meaning is in the SHAPE, and colour must not
#: become a second, conflicting code.
FILLS = {
    "macromolecule": "#EDF3F8",
    "complex": "#EAF1EC",
    "perturbation": "#FDF1E4",
    "simple chemical": "#F4F0F8",
    "source": "#FFFFFF",
    "sink": "#FFFFFF",
}
EDGES = {
    "macromolecule": "#486B88",
    "complex": "#3E7863",
    "perturbation": "#B4661E",
    "simple chemical": "#6A4C8C",
    "source": MUTED,
    "sink": MUTED,
}


# ---------------------------------------------------------------------------------------------
# glyph primitives
# ---------------------------------------------------------------------------------------------
def _rounded(axis, x, y, w, h, **kw):
    from matplotlib.patches import FancyBboxPatch

    pad = min(w, h) * 0.18
    axis.add_patch(
        FancyBboxPatch(
            (x - w / 2 + pad, y - h / 2 + pad),
            w - 2 * pad,
            h - 2 * pad,
            boxstyle=f"round,pad={pad:g},rounding_size={pad * 1.4:g}",
            **kw,
        )
    )


def _rectangle(axis, x, y, w, h, **kw):
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, **kw))


def _cut_corner(axis, x, y, w, h, **kw):
    """SBGN complex: a rectangle with its four corners cut off."""
    from matplotlib.patches import Polygon

    c = min(w, h) * 0.22
    axis.add_patch(
        Polygon(
            [
                (x - w / 2 + c, y - h / 2),
                (x + w / 2 - c, y - h / 2),
                (x + w / 2, y - h / 2 + c),
                (x + w / 2, y + h / 2 - c),
                (x + w / 2 - c, y + h / 2),
                (x - w / 2 + c, y + h / 2),
                (x - w / 2, y + h / 2 - c),
                (x - w / 2, y - h / 2 + c),
            ],
            closed=True,
            **kw,
        )
    )


def _perturbation(axis, x, y, w, h, **kw):
    """SBGN perturbation / perturbing agent: a hexagon whose vertical faces are concave."""
    from matplotlib.patches import Polygon

    c = w * 0.16
    axis.add_patch(
        Polygon(
            [
                (x - w / 2, y - h / 2),
                (x + w / 2, y - h / 2),
                (x + w / 2 - c, y),
                (x + w / 2, y + h / 2),
                (x - w / 2, y + h / 2),
                (x - w / 2 + c, y),
            ],
            closed=True,
            **kw,
        )
    )


def _source_sink(axis, x, y, w, h, **kw):
    """SBGN source-and-sink: a circle struck through by a diagonal."""
    from matplotlib.patches import Circle

    radius = min(w, h) / 2
    axis.add_patch(Circle((x, y), radius, **kw))
    offset = radius * 0.7071
    axis.plot(
        [x - offset, x + offset],
        [y - offset, y + offset],
        color=kw.get("edgecolor", INK),
        linewidth=kw.get("linewidth", 1.3),
        zorder=kw.get("zorder", 5) + 1,
        solid_capstyle="round",
    )


ENTITY_GLYPH = {
    "macromolecule": _rounded,
    "complex": _cut_corner,
    "perturbation": _perturbation,
    "simple chemical": _rounded,
    "source": _source_sink,
    "sink": _source_sink,
}


def _terminator(axis, start, end, kind, color=INK, scale=13.0, shrink=0.0, curve=0.0, pad=0.0):
    """Draw one arc with the SBGN terminator for ``kind`` at ``end``.

    ``curve`` bows the arc rather than running it straight. Two glyphs joined both ways get one
    curve each, bowing opposite sides, so a reaction and its reverse stop lying on top of one
    another. ``pad`` holds the arc off the glyphs at both ends, so an arrowhead does not sit
    against a box's outline.
    """
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    style = {
        "consumption": "-",
        "production": "-|>",
        "positive influence": "-|>",
        "stimulation": "-|>",
        "necessary stimulation": "-|>",
        "negative influence": "-",
        "inhibition": "-",
        "unknown influence": "-",
        "modulation": "-",
        "catalysis": "-",
    }[kind]
    filled = kind == "production"
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=scale,
            linewidth=1.2,
            shrinkA=pad,
            shrinkB=shrink + pad,
            facecolor=color if filled else "white",
            edgecolor=color,
            zorder=3,
            joinstyle="miter",
            connectionstyle=f"arc3,rad={curve:g}",
        )
    )
    if kind in ("negative influence", "inhibition", "necessary stimulation"):
        # A bar across the arc: the whole terminator for inhibition, and the extra stroke that
        # turns a positive arrowhead into necessary stimulation.
        direction = np.array(end, dtype=float) - np.array(start, dtype=float)
        length = float(np.hypot(*direction)) or 1.0
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        back = 0.055 if kind == "necessary stimulation" else 0.0
        at = np.array(end, dtype=float) - unit * (shrink / 72.0 + back)
        bar = 0.055
        axis.plot(
            [at[0] - normal[0] * bar, at[0] + normal[0] * bar],
            [at[1] - normal[1] * bar, at[1] + normal[1] * bar],
            color=color,
            linewidth=1.6,
            zorder=4,
            solid_capstyle="butt",
        )
    if kind in ("unknown influence", "modulation"):
        import numpy as np

        direction = np.array(end, dtype=float) - np.array(start, dtype=float)
        length = float(np.hypot(*direction)) or 1.0
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        tip = np.array(end, dtype=float)
        d = 0.05
        from matplotlib.patches import Polygon

        axis.add_patch(
            Polygon(
                [
                    tip,
                    tip - unit * d + normal * d * 0.7,
                    tip - unit * 2 * d,
                    tip - unit * d - normal * d * 0.7,
                ],
                closed=True,
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )
        )
    if kind == "catalysis":
        axis.plot(
            *end,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            zorder=4,
        )


# ---------------------------------------------------------------------------------------------
# Activity Flow
# ---------------------------------------------------------------------------------------------
def activity_flow(
    axis,
    network,
    *,
    box=(1.30, 0.40),
    gap=0.62,
    fontsize=11.5,
    colors=None,
    classes=None,
    mechanics=None,
    floor=0.0,
    compact=False,
    annotate=True,
    influence=None,
    arrow_x=0.0,
    notes=None,
    note_size=None,
    arrow_scale=15,
    rounded=False,
    box_linewidth=1.8,
):
    """Draw a readout network as an SBGN Activity Flow map, top to bottom.

    ``network`` is the manifest's ``reference_network``: nodes with ``label``/``kind``/``species``
    and edges carrying a tested ``influence``. The unit of information on each activity names the
    entity class; the species a readout sums are written under its label, because that sum is what
    the observable actually is.

    ``colors`` maps a node id to the colour that identifies that MEASURED QUANTITY elsewhere in the
    figure. It is applied to the stroke and the name, never to the fill: in SBGN the meaning lives
    in the glyph's shape, and a filled box would put two codes on one mark.

    ``classes`` maps a species to its entity class, so a readout can say WHAT it sums — monomers, a
    complex — instead of printing a formula for the reader to parse. ``mechanics`` adds, per
    readout, the reactions its own moiety runs and the motifs sitting on it: a total says how big a
    model is, this says where it spent it.
    """
    colors, classes, mechanics = colors or {}, classes or {}, mechanics or {}
    # ``influence`` draws every edge with one terminator, so maps whose edges were and were not
    # sign-tested read alike; ``arrow_x`` shifts the arrows off-centre (a fraction of the box
    # width) to leave room for ``notes``, text keyed by ``(source, target)`` and set beside the arrow.
    notes = notes or {}
    # ``floor`` holds the secondary text at a minimum print size (ICLR: 6 pt) where the map is
    # drawn small; the defaults keep the supplement's large maps exactly as they were.
    tag_size, detail_size, edge_size = (
        max(floor, fontsize - 2.8),
        max(floor, fontsize - 2.9),
        max(floor, fontsize - 2.0),
    )
    words = {
        "macromolecule": "monomer",
        "complex": "complex",
        "simple chemical": "chemical",
        "perturbation": "input",
    }
    width, height = box
    nodes = network["nodes"]
    axis.set_aspect("equal")
    axis.axis("off")
    positions = {}
    for index, node in enumerate(nodes):
        y = -index * (height + gap)
        positions[node["id"]] = y
        kind = "perturbation" if node["kind"] == "input" else "macromolecule"
        accent = colors.get(node["id"], INK)
        # The activity is a rectangle (rounded on request); the unit of information carries the class.
        # ``rounded`` keeps the outer size (the pad is taken inside), so it changes no layout.
        (_rounded if rounded else _rectangle)(
            axis,
            0,
            y,
            width,
            height,
            facecolor="white",
            edgecolor=accent,
            linewidth=box_linewidth,
            zorder=2,
        )
        # A perturbation's unit says "perturbation", the longest tag there is, inside a notched
        # glyph that loses width to its points: it gets the wider unit.
        unit_w = min(width * 0.70, 1.65) if kind == "perturbation" else min(width * 0.46, 1.05)
        unit_h = height * 0.32
        ux, uy = -width / 2 + unit_w / 2 + height * 0.10, y + height / 2
        # annotate=False draws the map as boxes, arrows and names only -- how Figure 3B shows it
        # at text width. The unit of information, the detail and the edge labels are secondary
        # text that cannot be set at 6 pt in a 0.9 in column; the supplement prints them all.
        axis.text(
            0,
            y + (height * 0.04 if annotate else 0),
            node["label"],
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=accent,
            zorder=4,
        )
        if not annotate:
            continue
        glyph = _perturbation if kind == "perturbation" else _rounded
        glyph(
            axis,
            ux,
            uy,
            unit_w,
            unit_h,
            facecolor="white",
            edgecolor=MUTED,
            linewidth=0.8,
            zorder=3,
        )
        # The unit of information names what the quantity is MEASURED BY where that is known: an
        # entity class a reader can already see from the glyph is the less useful of the two.
        tag = node.get("modality") or ("perturbation" if kind == "perturbation" else "protein")
        axis.text(ux, uy, tag, ha="center", va="center", fontsize=tag_size, color=MUTED, zorder=4)
        grouped: dict[str, list[str]] = {}
        for name in node["species"]:
            grouped.setdefault(words.get(classes.get(name, ""), "state"), []).append(name)
        said = [
            f"{word}{'s' if len(found) > 1 else ''} {', '.join(found)}"
            for word, found in grouped.items()
        ]
        detail = (
            "applied stimulus"
            if node["kind"] == "input"
            else "measured as " + ("\n" if len(node["species"]) > 1 else "") + " and ".join(said)
        )
        # compact keeps only the mechanics line: at text width with a 6 pt floor the species a
        # readout sums do not fit in the box, and the supplement's process-description maps
        # carry them in full.
        if compact and node["kind"] != "input":
            detail = ""
        reactions, motifs_here = mechanics.get(node["id"], (None, ()))
        if reactions is not None:
            line = (
                f"{reactions} reaction"
                + ("s" if reactions != 1 else "")
                + (" · " + ", ".join(motifs_here) if motifs_here else "")
            )
            detail = f"{detail}\n{line}" if detail else line
        axis.text(
            0,
            y - height * (0.26 if compact else 0.22),
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=MUTED,
            zorder=4,
            linespacing=1.15,
        )
    for edge in network["edges"]:
        start = positions[edge["source"]] - height / 2
        end = positions[edge["target"]] + height / 2
        kind = influence or {
            "positive": "positive influence",
            "negative": "negative influence",
        }.get(edge.get("influence"), "unknown influence")
        x = arrow_x * width
        _terminator(axis, (x, start), (x, end), kind, color=INK, scale=arrow_scale)
        note = notes.get((edge["source"], edge["target"]))
        if note:
            axis.text(
                x + width * 0.13,
                (start + end) / 2,
                note,
                ha="left",
                va="center",
                fontsize=note_size or edge_size,
                color=MUTED,
                zorder=5,
                linespacing=1.0,
                gid="note",
            )
        if not annotate:
            continue
        label = f"{edge['hops']} reaction" + ("s" if edge["hops"] != 1 else "")
        if edge["path"][1:-1]:
            # One line when compact: the gap between boxes holds a single 6 pt line, and a
            # second one ran into the next box's unit of information.
            label += (" via " if compact else "\nvia ") + ", ".join(edge["path"][1:-1])
        axis.text(
            0,
            (start + end) / 2,
            label,
            ha="center",
            va="center",
            fontsize=edge_size,
            color=MUTED,
            linespacing=1.05,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    axis.set_xlim(-width / 2 - width * 0.06, width / 2 + width * 0.06)
    axis.set_ylim(
        -(len(nodes) - 1) * (height + gap) - height / 2 - gap * 0.55, height / 2 + height * 0.18
    )


def activity_flow_key(axis, x, y, *, fontsize=7.4):
    """A one-line key naming the glyphs, so the panel is readable without the SBGN spec."""
    axis.text(
        x,
        y,
        "SBGN activity flow: rectangle = biological activity, tag = entity class "
        "(mt:prot macromolecule, pert perturbation),\nopen arrowhead = positive influence "
        "(sign tested on the published kinetics).",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=MUTED,
        transform=axis.transAxes,
    )


# ---------------------------------------------------------------------------------------------
# Process Description
# ---------------------------------------------------------------------------------------------
def _layers(diagram):
    """Breadth-first layers of the bipartite entity/process graph, from its inputs and sources.

    Breadth-first rather than longest-path because these networks contain cycles (receptor
    turnover, the phosphatase loops), where a longest path is undefined.
    """
    from collections import deque

    successors: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for arc in diagram["arcs"]:
        successors.setdefault(arc["source"], set()).add(arc["target"])
        indegree[arc["target"]] = indegree.get(arc["target"], 0) + 1
        indegree.setdefault(arc["source"], indegree.get(arc["source"], 0))
    ids = [node["id"] for node in diagram["nodes"]] + [p["id"] for p in diagram["processes"]]
    roots = [node["id"] for node in diagram["nodes"] if node["class"] == "perturbation"]
    roots += [p["id"] for p in diagram["processes"] if p["n_reactants"] == 0]
    roots += [i for i in ids if not indegree.get(i)]
    layer = {}
    queue = deque()
    for root in dict.fromkeys(roots):
        layer[root] = 0
        queue.append(root)
    while queue:
        current = queue.popleft()
        for nxt in sorted(successors.get(current, ())):
            if nxt not in layer:
                layer[nxt] = layer[current] + 1
                queue.append(nxt)
    depth = max(layer.values(), default=0)
    for i in ids:  # anything the traversal never reached sits below the reachable part
        layer.setdefault(i, depth + 1)
    return layer


def _order(diagram, layer, sweeps=12):
    """Barycentre ordering within each layer, to keep the arcs from crossing more than they must.

    More sweeps than a drawing this size would normally need, because pinning nodes to a shared
    grid puts arcs between columns that are far apart, and a couple of passes leaves those crossing
    everything in between.
    """
    neighbours: dict[str, set[str]] = {}
    for arc in diagram["arcs"]:
        neighbours.setdefault(arc["source"], set()).add(arc["target"])
        neighbours.setdefault(arc["target"], set()).add(arc["source"])
    by_layer: dict[float, list[str]] = {}
    for node, index in sorted(layer.items()):
        by_layer.setdefault(index, []).append(node)
    position = {node: float(i) for nodes in by_layer.values() for i, node in enumerate(nodes)}
    for sweep in range(sweeps):
        order = sorted(by_layer) if sweep % 2 == 0 else sorted(by_layer, reverse=True)
        for index in order:
            nodes = by_layer[index]
            centres = []
            for node in nodes:
                near = [position[n] for n in neighbours.get(node, ()) if layer[n] != index]
                centres.append((sum(near) / len(near) if near else position[node], node))
            centres.sort()
            by_layer[index] = [node for _, node in centres]
            for i, node in enumerate(by_layer[index]):
                position[node] = float(i)
    return by_layer


def _pools(diagram):
    """The entity pools that claim a row. A source or sink is punctuation, not an entity."""
    return {node["id"] for node in diagram["nodes"] if node["class"] not in ("source", "sink")}


def _conversions(diagram):
    """Pool pairs a single-reactant, single-product process converts between.

    One reactant and one product is the signature of a STATE CHANGE — phosphorylation, activation,
    internalisation — so the two pools are one entity wearing two hats. Read off the arcs rather
    than off the names, because an agent's identifiers carry no ontology.
    """
    pools = _pools(diagram)
    single = {
        process["id"]
        for process in diagram["processes"]
        if process["n_reactants"] == 1 and process["n_products"] == 1
    }
    ends: dict[str, tuple[set, set]] = {}
    for arc in diagram["arcs"]:
        if arc["target"] in single and arc["kind"] == "consumption" and arc["source"] in pools:
            ends.setdefault(arc["target"], (set(), set()))[0].add(arc["source"])
        if arc["source"] in single and arc["kind"] == "production" and arc["target"] in pools:
            ends.setdefault(arc["source"], (set(), set()))[1].add(arc["target"])
    return [
        (before, after)
        for consumed, produced in ends.values()
        for before in consumed
        for after in produced
    ]


def _combos(diagrams):
    """``complex -> the pools it is made of``, for the things that belong between two moieties.

    A complex is not a state of one entity, it is two entities stuck together, so it has no place
    in either one's row. Its parts come from the id the model gave it, which is a naming heuristic
    and recorded as one; the export has already used the same rule to call it a complex.
    """
    pools = {node for diagram in diagrams for node in _pools(diagram)}
    parts = {}
    for diagram in diagrams:
        for node in diagram["nodes"]:
            if node["class"] != "complex":
                continue
            pieces = node["id"].split("_")
            found = {"_".join(pieces[:cut]) for cut in range(1, len(pieces))}
            found |= {"_".join(pieces[cut:]) for cut in range(1, len(pieces))}
            made = sorted(found & pools)
            if made:
                parts[node["id"]] = made
    return parts


def motifs(diagram, anchors):
    """The mechanisms a model implements, named from its ARCS rather than from its identifiers.

    Only motifs that can be READ OFF the structure appear here — a cycle of conversions, a species
    made from a source that only ever modulates, a complex, a branch off the measured cascade. What
    the modeller would call each one (receptor internalisation, integral feedback) is a reading of
    the biology this export cannot make, so the structural name is what gets printed.
    """
    pools = _pools(diagram)
    combos = _combos([diagram])
    conversions = _conversions(diagram)
    forward: dict[str, set[str]] = {}
    for before, after in conversions:
        forward.setdefault(before, set()).add(after)

    def cycles(start):
        seen, frontier = {start}, [start]
        while frontier:
            node = frontier.pop()
            for nxt in sorted(forward.get(node, ())):
                if nxt == start:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    found = []
    if any(len(forward.get(node, ())) and cycles(node) for node in sorted(pools)):
        found.append("a state cycle")
    made_from_source = {
        arc["target"]
        for arc in diagram["arcs"]
        if arc["kind"] == "production"
        and arc["target"] in pools
        and not any(
            other["kind"] == "consumption"
            and other["source"] in pools
            and other["target"] == arc["source"]
            for other in diagram["arcs"]
        )
    }
    modulating = {
        arc["source"]
        for arc in diagram["arcs"]
        if arc["source"] in pools and arc["kind"] not in ("consumption", "production")
    }
    if made_from_source & modulating:
        found.append("a synthesised regulator")
    if combos:
        found.append(f"{len(combos)} complex{'es' if len(combos) > 1 else ''}")
    measured = {name for names in anchors.values() for name in names}
    off = {node for node in pools if node not in measured and node not in combos}
    if off:
        found.append(f"{len(off)} states off the readouts")
    return found


def readout_mechanics(diagram, anchors):
    """Per readout: how many reactions its own moiety runs, and which motifs sit on it.

    A total tells a reader how big a model is; this tells them WHERE the model spent it. Read off
    the arcs like every other mechanism name here, so a cycle is a cycle of conversions and a
    regulator is a species made from a source that only modulates.
    """
    pools = _pools(diagram)
    moiety = _moieties([diagram], [anchors])
    conversions = _conversions(diagram)
    forward: dict[str, set[str]] = {}
    for before, after in conversions:
        forward.setdefault(before, set()).add(after)
    made_from_source = {
        arc["target"]
        for arc in diagram["arcs"]
        if arc["kind"] == "production"
        and arc["target"] in pools
        and not any(
            other["kind"] == "consumption"
            and other["source"] in pools
            and other["target"] == arc["source"]
            for other in diagram["arcs"]
        )
    }
    regulators = {
        arc["source"]
        for arc in diagram["arcs"]
        if arc["source"] in made_from_source and arc["kind"] not in ("consumption", "production")
    }
    out = {}
    for readout, species in anchors.items():
        group = {
            node
            for node in pools
            if any(node in moiety and moiety[node] == moiety.get(name) for name in species)
        }
        touching = {
            arc["target"] if arc["kind"] == "consumption" else arc["source"]
            for arc in diagram["arcs"]
            if arc["kind"] in ("consumption", "production")
            and (arc["source"] in group or arc["target"] in group)
        }
        found = []
        if any(_reaches(forward, node, node, group) for node in sorted(group)):
            found.append("cycle")
        ruled = {
            arc["target"]
            for arc in diagram["arcs"]
            if arc["source"] in regulators and arc["kind"] not in ("consumption", "production")
        }
        if any(
            arc["source"] in ruled and arc["target"] in group and arc["kind"] == "production"
            for arc in diagram["arcs"]
        ):
            found.append("regulated")
        out[readout] = (len(touching), found)
    return out


def _reaches(forward, start, goal, within):
    """Whether ``start`` returns to ``goal`` through conversions inside ``within``."""
    seen, frontier = set(), list(forward.get(start, ()))
    while frontier:
        node = frontier.pop()
        if node == goal:
            return True
        if node in seen or node not in within:
            continue
        seen.add(node)
        frontier.extend(forward.get(node, ()))
    return False


def _moieties(diagrams, anchors):
    """``pool -> moiety`` over every map at once: one entity in all the states any map gives it.

    Three things put two pools in one moiety: a conversion between them, being summed into the same
    observable, and carrying the same name in two maps. The last is what makes a moiety mean the
    same thing in both, and the first two are what make it a moiety rather than a coincidence.
    """
    parent: dict[str, str] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        parent[find(left)] = find(right)

    combos = _combos(diagrams)
    for diagram, anchor in zip(diagrams, anchors, strict=True):
        for node in _pools(diagram):
            if node not in combos:
                find(node)
        for before, after in _conversions(diagram):
            if before not in combos and after not in combos:
                union(before, after)
        summed: dict[object, list[str]] = {}
        for node, key in anchor.items():
            if node in parent:
                summed.setdefault(key, []).append(node)
        for group in summed.values():
            for node in group[1:]:
                union(group[0], node)
    return {node: find(node) for node in parent}


def _base_first(members, conversions):
    """Order one moiety's pools: the unmodified form first, then by conversion steps from it.

    Breadth-first LAYERS cannot do this. These networks recycle — the candidate reaches its free
    receptor only through the internalised form — so a layering reaches the base state last and
    would draw every cascade upside down. Conversions give the chain instead, and the base is
    picked by a NAMING heuristic, recorded here as one: a state's id usually contains the id of the
    thing it is a state of (``pAkt`` contains ``Akt``, ``aS6K`` contains ``S6K``).
    """
    contains = {
        node: sum(1 for other in members if other != node and other in node) for node in members
    }
    base = min(members, key=lambda node: (contains[node], len(node), node))
    steps, frontier = {base: 0}, [base]
    while frontier:
        node = frontier.pop(0)
        for before, after in sorted(conversions):
            if before == node and after in members and after not in steps:
                steps[after] = steps[node] + 1
                frontier.append(after)
    # A pool with no conversion of its own — a complex summed into the same observable — follows
    # the pool it is named after rather than falling to the end.
    for node in sorted(members, key=lambda n: (contains[n], len(n), n)):
        if node not in steps:
            near = [steps[other] for other in members if other in steps and other in node]
            steps[node] = (max(near) if near else max(steps.values(), default=0)) + 0.5
    return sorted(members, key=lambda node: (steps[node], contains[node], len(node), node))


def _pack(sequences, shared):
    """Offsets for several orderings of one block, keeping the shared members level.

    Each sequence keeps its own order; a member two sequences have in common lands on the same
    offset in both, and the members between two such syncs take as many rows as the busiest
    sequence needs there. That is what lets one map's extra states sit beside the other's without
    either being pushed out of order.
    """
    syncs = [node for node in sequences[0] if node in shared]
    for sequence in sequences[1:]:
        syncs = [node for node in syncs if node in sequence]
    offsets, at = [{} for _ in sequences], 0
    cuts = [[sequence.index(node) for node in syncs] for sequence in sequences]
    starts = [0] * len(sequences)
    for step in range(len(syncs) + 1):
        runs = [
            sequence[start : (cut[step] if step < len(syncs) else len(sequence))]
            for sequence, start, cut in zip(sequences, starts, cuts, strict=True)
        ]
        for offset, run in zip(offsets, runs, strict=True):
            for index, node in enumerate(run):
                offset[node] = at + index
        at += max((len(run) for run in runs), default=0)
        if step < len(syncs):
            for offset in offsets:
                offset[syncs[step]] = at
            starts = [cut[step] + 1 for cut in cuts]
            at += 1
    return offsets, at


def entity_rows(diagrams, anchors):
    """``pool -> row``, one grid for every map, with each moiety on consecutive rows.

    A moiety is one entity in all the states any map gives it, so its pools belong together: that
    is what stops an unrelated pool landing between a protein and its phosphorylated form. Moieties
    are ordered by how far they sit from the stimulus along the graph BETWEEN moieties, which is
    acyclic where the graph itself is not.
    """
    from collections import deque

    moiety = _moieties(diagrams, anchors)
    conversions = {edge for diagram in diagrams for edge in _conversions(diagram)}
    blocks: dict[str, set[str]] = {}
    for node, group in moiety.items():
        blocks.setdefault(group, set()).add(node)
    order = _base_first
    ranked = {group: order(members, conversions) for group, members in blocks.items()}

    reach: dict[str, list[int]] = {}
    after: dict[str, set[str]] = {}
    regulators: set[tuple[str, str]] = set()
    synthesised: set[str] = set()
    converted: set[str] = set()
    for diagram in diagrams:
        pools = {node for node in _pools(diagram) if node in moiety}
        near: dict[str, set[str]] = {moiety[node]: set() for node in pools}
        for arc in diagram["arcs"]:
            ends = [end for end in (arc["source"], arc["target"]) if end in pools]
            if len(ends) == 2 and moiety[ends[0]] != moiety[ends[1]]:
                near[moiety[ends[0]]].add(moiety[ends[1]])
        # Through the processes too: a pool reaches another by consuming and producing around one.
        traded: dict[str, set[str]] = {}
        regulates: dict[str, set[str]] = {}
        for process in {p["id"] for p in diagram["processes"]}:
            touching = {
                moiety[arc["source"]]
                for arc in diagram["arcs"]
                if arc["target"] == process
                and arc["kind"] == "consumption"
                and arc["source"] in pools
            }
            made = {
                moiety[arc["target"]]
                for arc in diagram["arcs"]
                if arc["source"] == process
                and arc["kind"] == "production"
                and arc["target"] in pools
            }
            modulating = {
                moiety[arc["source"]]
                for arc in diagram["arcs"]
                if arc["target"] == process
                and arc["source"] in pools
                and arc["kind"] not in ("consumption", "production")
            }
            for group in touching:
                near.setdefault(group, set()).update(made - {group})
                traded.setdefault(group, set()).update(made - {group})
                for other in made - {group}:
                    traded.setdefault(other, set()).add(group)
            for group in modulating:
                near.setdefault(group, set()).update(made - {group})
                regulates.setdefault(group, set()).update(made - {group})
            # A pool made from a SOURCE is synthesised, not converted from anything on the map.
            for group in made if not touching else ():
                synthesised.add(group)
            for group in made if touching else ():
                converted.add(group)
        for group, ruled in regulates.items():
            for other in ruled:
                if other not in traded.get(group, set()) and group not in traded.get(other, set()):
                    regulators.add((group, other))
        roots = [
            moiety[node["id"]]
            for node in diagram["nodes"]
            if node["class"] == "perturbation" and node["id"] in moiety
        ]
        seen = {group: 0 for group in roots}
        queue = deque(roots)
        while queue:
            group = queue.popleft()
            for nxt in sorted(near.get(group, ())):
                if nxt not in seen:
                    seen[nxt] = seen[group] + 1
                    queue.append(nxt)
        depth = max(seen.values(), default=0) + 1
        for group in near:
            reach.setdefault(group, []).append(seen.get(group, depth))
        # A map that has BOTH moieties has the say on which comes first. Averaging distances across
        # maps does not: a map that lacks the kinase pulls its substrate forward past it.
        for group, at in seen.items():
            for other, also in seen.items():
                if at < also:
                    after.setdefault(other, set()).add(group)
    # A moiety that only ever MODULATES another — never exchanges material with it — is its
    # regulator, and a regulator belongs above the step it regulates. This OVERRIDES the distance
    # rule, which cannot see a feedback species at all: it has no material path from the stimulus,
    # so its distance is whatever the traversal happened to give it.
    for group, other in regulators:
        # A feedback regulates the step that makes it and that step regulates it back. Where both
        # directions are present the one that is only ever SYNTHESISED is the regulator: the other
        # is a step of the cascade, and a cascade step is not drawn above its own feedback.
        if (other, group) in regulators and not (group in synthesised and group not in converted):
            continue
        after.setdefault(other, set()).add(group)
        after.get(group, set()).discard(other)
    middle = {
        group: sum(reach[group]) / len(reach[group]) if reach.get(group) else 0.0
        for group in blocks
    }
    sequence, placed = [], set()
    while len(sequence) < len(blocks):
        ready = [
            group
            for group in blocks
            if group not in placed and not (after.get(group, set()) - placed)
        ]
        if not ready:  # a disagreement the maps cannot settle; fall back to the averages
            ready = [group for group in blocks if group not in placed]
        chosen = min(ready, key=lambda group: (middle[group], group))
        sequence.append(chosen)
        placed.add(chosen)
    rows, slots = {}, {}
    # Beside the column, not in it: a moiety that REGULATES another and is only ever synthesised
    # from a source — never converted from anything on the map — is not part of the material
    # cascade at all, so standing it in the cascade's own column would misread it as a step.
    aside = {
        group for group, _other in regulators if group in synthesised and group not in converted
    }
    for at, group in enumerate(sequence):
        present = [
            [node for node in ranked[group] if node in _pools(diagram)] for diagram in diagrams
        ]
        shared = {node for node in ranked[group] if sum(node in run for run in present) > 1}
        offsets, width = _pack([run for run in present if run] or [[]], shared)
        combined: dict[str, int] = {}
        for offset in offsets:
            combined |= offset
        # Every row centred on the same axis, except a pure regulator: it is not part of the
        # material cascade, so it stands beside the column rather than in it.
        middle = (width - 1) / 2 - (1.0 if group in aside else 0.0)
        for node, index in combined.items():
            # ONE ROW PER MOIETY, and the progression along it: the unmodified form on the left,
            # each further state one step right. Down the page is then one conserved pool after
            # another, which is the axis the two models can actually be compared on.
            rows[node] = float(at)
            slots[node] = float(index) - middle
    # A complex belongs to neither of the moieties it joins, so it goes on a layer of its own: half
    # a row under the earlier of its parts, which is where the two of them meet.
    combos = _combos(diagrams)
    for node, made in combos.items():
        known = [part for part in made if part in rows]
        if known:
            rows[node] = min(rows[part] for part in known) + 0.5
            slots[node] = sum(slots[part] for part in known) / len(known)
    return rows, slots


def _interpolate(marks, value):
    """``value`` on the scale the observable marks define, extrapolating past the ends by slope."""
    if len(marks) < 2:
        return float(value)
    if value <= marks[0][0]:
        low, high = marks[0], marks[1]
    elif value >= marks[-1][0]:
        low, high = marks[-2], marks[-1]
    else:
        low, high = next((a, b) for a, b in zip(marks, marks[1:]) if value <= b[0])
    if high[0] == low[0]:
        return float(low[1])
    return low[1] + (value - low[0]) * (high[1] - low[1]) / (high[0] - low[0])


def _place(diagram, grid, across=None, lean=0.30, offset=0.30):
    """``node -> position along the flow``: one row per pool, processes in the layers between.

    SBGN reads as alternating pools and processes, so a process belongs in the layer its pools
    straddle rather than on a row of its own. A source or sink rides its process for the same
    reason.

    ``across`` gives each node its sideways coordinate, and when it is known a process LEANS by the
    way it carries material: towards the earlier row if rightwards, the later row if leftwards. It
    lands 30% of a row above the pools it joins when it carries material rightwards and 30% below
    when leftwards, whether they share a row or straddle two. A pool pair joined by a forward and a reverse process would
    otherwise stack both squares, and both sets of arcs, on one point.
    """
    pools = _pools(diagram)
    columns = {node: float(grid[node]) for node in pools if node in grid}
    touching: dict[str, list[float]] = {}
    modulated: dict[str, list[float]] = {}
    attached: dict[str, list[str]] = {}
    for arc in diagram["arcs"]:
        for near, other in ((arc["source"], arc["target"]), (arc["target"], arc["source"])):
            if near not in pools and other in pools and other in columns:
                # Only what a process CONSUMES and MAKES decides which rows it sits between. A
                # modulator reaching in from another row must not drag the square off the step it
                # belongs to — the S6 conversion belongs on the S6 row, not halfway to its kinase.
                where = touching if arc["kind"] in ("consumption", "production") else modulated
                where.setdefault(near, []).append(columns[other])
            elif near not in pools and other not in pools:
                attached.setdefault(near, []).append(other)
    default = sum(columns.values()) / len(columns) if columns else 0.0
    sideways = _sides(diagram, pools)
    for process in {p["id"] for p in diagram["processes"]}:
        near = touching.get(process) or modulated.get(process, ())
        columns[process] = (min(near) + max(near)) / 2 if near else default
        if across and near:
            before, after = sideways.get(process, (None, None))
            drift = (
                0.0
                if before is None or after is None
                else across.get(after, 0.0) - across.get(before, 0.0)
            )
            # The wider the reach, the further out of the row it goes: a square sitting beside a
            # long arc would otherwise have that arc pass straight through its own row.
            reach = 1.0 + 0.35 * max(0.0, abs(drift) - 1.0)
            step = (lean if len(set(near)) > 1 else offset) * min(reach, 2.0)
            columns[process] -= step * (1 if drift > 0 else -1 if drift < 0 else 0)
    for node in diagram["nodes"]:  # the sources and sinks, on their process's layer
        if node["id"] not in columns:
            near = [columns[other] for other in attached.get(node["id"], ()) if other in columns]
            columns[node["id"]] = sum(near) / len(near) if near else default
    return columns


def _sides(diagram, pools):
    """``process -> (a pool it consumes, a pool it produces)``, for reading its direction."""
    consumed, produced = {}, {}
    for arc in diagram["arcs"]:
        if arc["kind"] == "consumption" and arc["source"] in pools:
            consumed.setdefault(arc["target"], arc["source"])
        if arc["kind"] == "production" and arc["target"] in pools:
            produced.setdefault(arc["source"], arc["target"])
    return {
        process: (consumed.get(process), produced.get(process))
        for process in {p["id"] for p in diagram["processes"]}
    }


def _geometry(diagram, grid, slots, *, entity, dx, dy, flow):
    """``(centre, xlim, ylim)``: where every glyph goes, and the box that holds them.

    Two ordering passes, because a process's lean depends on which way it carries material and that
    is only known once the pools have their sideways slots.
    """
    width, height = entity
    horizontal = flow == "horizontal"
    columns = _place(diagram, grid)
    across = _sideways(diagram, columns, slots)
    columns = _place(diagram, grid, across=across)
    across = _sideways(diagram, columns, slots)
    centre = {
        node: (
            (columns[node] * dy, across[node] * dx)
            if horizontal
            else (across[node] * dx, -columns[node] * dy)
        )
        for node in across
    }
    reach = [columns[node] for node in across] or [0.0]
    first, last = min(reach), max(reach)
    sides = [value * dx for value in across.values()] or [0.0]
    left, right = min(sides), max(sides)
    if horizontal:
        return (
            centre,
            (first * dy - width / 2 - 0.3, last * dy + width / 2 + 0.3),
            (left - height / 2 - 0.42, right + height / 2 + 0.22),
        )
    return (
        centre,
        (left - width / 2 - 0.22, right + width / 2 + 0.22),
        (-last * dy - height / 2 - 0.42, -first * dy + height / 2 + 0.22),
    )


def _sideways(diagram, columns, slots):
    """``node -> sideways slot``: the pools where their moiety puts them, the rest beside them.

    A process has no place of its own, so it takes the average of the pools it joins and is nudged
    clear of any other process in the same layer.
    """
    pools = _pools(diagram)
    across = {node: slots.get(node, 0.0) for node in pools if node in columns}
    joined: dict[str, list[str]] = {}
    near: dict[str, list[str]] = {}
    for arc in diagram["arcs"]:
        for node, other in ((arc["source"], arc["target"]), (arc["target"], arc["source"])):
            if node not in across and other in columns:
                near.setdefault(node, []).append(other)
                if arc["kind"] in ("consumption", "production") and other in pools:
                    joined.setdefault(node, []).append(other)
    for _pass in range(2):
        for node in columns:
            if node in pools and node in across:
                continue
            # Midway between the pools it CONNECTS — what it consumes and what it makes. A
            # modulator should not drag the square off the line it sits on.
            between = [across[other] for other in joined.get(node, ()) if other in across]
            beside = between or [across[o] for o in near.get(node, ()) if o in across]
            across[node] = (min(beside) + max(beside)) / 2 if beside else 0.0
    # Only what actually collides is moved, and only far enough: respacing a whole layer would
    # scramble processes that were already where their pools put them.
    crowd: dict[float, list[str]] = {}
    for node in columns:
        if node not in pools:
            crowd.setdefault(round(columns[node], 6), []).append(node)
    for layer in crowd.values():
        layer.sort(key=lambda node: (across[node], node))
        previous = None
        for node in layer:
            if previous is not None and across[node] < previous + 0.42:
                across[node] = previous + 0.42
            previous = across[node]
    return across


def process_description_limits(
    diagram,
    *,
    entity=(1.25, 0.44),
    dx=1.55,
    dy=1.05,
    flow="vertical",
    grid=None,
    slots=None,
    **_ignored,
):
    """``(xlim, ylim)`` the map needs, without drawing it.

    A caller that lays out a page has to know the shape of a diagram BEFORE it allocates the axes:
    a cascade is long and thin, and two cascades are thin in different proportions, so giving both
    the same box wastes most of the page on white.
    """
    _centre, xlim, ylim = _geometry(
        diagram, grid or {}, slots or {}, entity=entity, dx=dx, dy=dy, flow=flow
    )
    return xlim, ylim


def process_description(
    axis,
    diagram,
    *,
    entity=(1.25, 0.44),
    process=0.22,
    dx=1.55,
    dy=1.05,
    fontsize=8.0,
    label_processes=False,
    flow="vertical",
    highlight=None,
    grid=None,
    slots=None,
    along=None,
    curve=0.0,
    pad=0.0,
):
    """Draw an SBGN Process Description map, inputs first.

    ``diagram`` is the manifest's portable description: ``nodes`` (id/label/class), ``processes``
    and ``arcs`` (consumption, production, stimulation, inhibition, modulation). Layout is
    deterministic — layered breadth-first, barycentre-ordered — so the same model always draws the
    same way.

    ``flow`` puts the layers down the page (``vertical``) or across it (``horizontal``). A
    signalling cascade is long and thin, so which way it runs decides whether it fits the space at
    a readable size: on a portrait page the horizontal flow is the one that does.

    ``highlight`` maps an entity id to ``(colour, observable)``. Two models of the same system
    share no species names, so the only way to read one map against the other is by what each pool
    is MEASURED as; naming that observable on the glyph, in the colour the rest of the figure uses
    for it, is what makes the comparison possible at all.

    ``pad`` holds every arc off the glyphs at both ends. ``curve`` bows every arc by that much. On a grid of glyphs it is what keeps a reaction and its
    reverse apart, and it pulls a long arc clear of the rows it would otherwise cut straight
    through.

    ``grid`` gives every entity pool its row, shared by every map drawn with it and the same
    ``along`` limits, so the same entity sits at the same height in each. Without it two maps of
    one system are laid out independently and can only be compared by reading both end to end.
    """
    highlight = highlight or {}
    width, height = entity
    centre, xlim, ylim = _geometry(
        diagram, grid or {}, slots or {}, entity=entity, dx=dx, dy=dy, flow=flow
    )
    classes = {node["id"]: node["class"] for node in diagram["nodes"]}
    labels = {node["id"]: node["label"] for node in diagram["nodes"]}
    reversible = {p["id"]: p["reversible"] for p in diagram["processes"]}
    horizontal = flow == "horizontal"
    axis.set_aspect("equal")
    axis.axis("off")

    def extent(node):
        """Half-width and half-height of a glyph: a process is a small square and a source/sink a
        small circle, so an arc must not stop where a full entity box would end."""
        if node not in classes:
            return process / 2, process / 2
        if classes[node] in ("source", "sink"):
            return height / 2, height / 2
        return width / 2, height / 2

    def port(node, towards):
        """Where an arc meets a glyph: the edge of the box, not its centre."""
        x, y = centre[node]
        tx, ty = towards
        half_w, half_h = extent(node)
        dxa, dya = tx - x, ty - y
        if abs(dya) * half_w >= abs(dxa) * half_h:
            return (x + dxa * half_h / (abs(dya) or 1), y + math.copysign(half_h, dya))
        return (x + math.copysign(half_w, dxa), y + dya * half_w / (abs(dxa) or 1))

    for arc in diagram["arcs"]:
        source, target = arc["source"], arc["target"]
        if source not in centre or target not in centre:
            continue
        start = port(source, centre[target])
        end = port(target, centre[source])
        kind = arc["kind"]
        if kind == "consumption" and reversible.get(target):
            kind = "production"  # a reversible process produces at both ends
        _terminator(
            axis,
            start,
            end,
            kind,
            color=MUTED if kind == "consumption" else INK,
            scale=11,
            curve=curve,
            pad=pad,
        )
    for pid, rev in reversible.items():
        if pid not in centre:
            continue
        x, y = centre[pid]
        _rectangle(
            axis, x, y, process, process, facecolor="white", edgecolor=INK, linewidth=1.3, zorder=4
        )
        if label_processes:
            axis.text(x, y - process, pid, ha="center", va="top", fontsize=6.0, color=MUTED)
    for node in diagram["nodes"]:
        if node["id"] not in centre:
            continue
        x, y = centre[node["id"]]
        klass = classes[node["id"]]
        half_w, half_h = extent(node["id"])
        accent, observable = highlight.get(node["id"], (None, None))
        # A highlighted pool drops the class fill: the shape already gives the class, and a tinted
        # box under a coloured outline would put a third colour on one glyph.
        ENTITY_GLYPH.get(klass, _rounded)(
            axis,
            x,
            y,
            half_w * 2,
            half_h * 2,
            facecolor="#FFFFFF" if accent else FILLS.get(klass, "#FFFFFF"),
            edgecolor=accent or EDGES.get(klass, INK),
            linewidth=2.0 if accent else 1.3,
            zorder=5,
        )
        if labels[node["id"]]:
            # A pool's id is whatever the model called it, and a complex's is two ids joined. The
            # glyph cannot grow — the columns are a shared grid — so a long name shrinks to fit
            # rather than running over its neighbours.
            name = labels[node["id"]]
            axis.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                color=INK,
                zorder=6,
                fontsize=fontsize * max(0.68, min(1.0, 7.5 / len(name))),
            )
        if observable:
            axis.text(
                x,
                y - half_h - 0.04,
                observable,
                ha="center",
                va="top",
                fontsize=fontsize - 0.6,
                fontweight="bold",
                color=accent,
                zorder=6,
            )
    if along:  # the flow's own axis, shared so two maps line up
        xlim, ylim = (along, ylim) if horizontal else (xlim, along)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)


def process_description_key(axis, x, y, *, fontsize=7.4):
    axis.text(
        x,
        y,
        "SBGN process description: rounded = macromolecule, cut corners = complex, "
        "concave hexagon = perturbation, small square = process;\nplain line = consumption, "
        "filled arrowhead = production, open arrowhead = stimulation.",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=MUTED,
        transform=axis.transAxes,
    )
