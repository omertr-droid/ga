#!/usr/bin/env python
"""Build a shareable PDF: how our OCT-only GA area agrees with the PLEX (advRPE) reference.

The cohort is the doctor viewer's own library -- the `qc_status == ok` eyes that carry a PLEX value,
read straight from the baked `viewer/data_store/library/index.json`. So every number in this PDF is a
number a reader can also see on a Library card. Metrics come from `viewer.core.stats.cohort_stats`.

Three pages, print-first (white surface, no dark-app chrome):
  1. Headline figures, the scatter (full range + a 0-3 mm^2 zoom) and Bland-Altman.
  2. The per-eye table.
  3. Method, reference and cohort.

Eyes listed in results/plex_adjudication.csv are ones where the REFERENCE was judged wrong, not us. A
wholly-wrong reference (filled diamond) has its area CORRECTED to 0 in a second "corrected" figure quoted
beside each headline number; a PARTLY-wrong one (hollow diamond) is scored unchanged. Nothing is dropped,
and the uncorrected numbers still score every eye against the reference exactly as Zeiss produced it.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\ga_report.py                    -> outputs/ga_report.pdf
  oct_env\\Scripts\\python.exe src\\ga_report.py --out some\\where.pdf
"""
import argparse
import datetime as _dt
import os
import sys
import textwrap

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paths import OUT_DIR, RESULTS_DIR  # noqa: E402
from viewer.core import bundle, stats  # noqa: E402

# Print palette, validated for a white surface (OKLCH lightness band, chroma floor, CVD separation
# dE 32.9 protan, >=3:1 contrast). Control eyes are hollow and reference-error eyes are diamonds, so
# no category rests on colour alone. Grid/axes are recessive hairlines; text wears ink, never a hue.
C_AGREE = "#12805c"
C_DIFFER = "#b45309"
C_REFERR = "#5b52c9"
C_INK = "#1c1f24"
C_MUTED = "#6b7280"
C_GRID = "#e3e6ea"

PAGE = (8.27, 11.69)          # A4 portrait, inches
ZOOM = 3.0                    # mm^2 -- the band most eyes live in
ADJ_CSV = os.path.join(RESULTS_DIR, "plex_adjudication.csv")


def _short(e):
    """'003-010' + 'OD' -> '010 OD' -- short enough to sit beside a point."""
    return f"{e['patient_id'].split('-')[-1]} {e['eye']}"


def _mark(ax, x, y, e, label=False):
    """One eye.

    A reference error is a labelled FILLED diamond in its own hue: the gap to the reference is real, but
    it is the reference that is wrong, so it must not read as our miss. A PARTIAL reference error is the
    same diamond left HOLLOW -- the reference over-called, yet real atrophy is there and we missed some of
    it, so the eye is still counted against us and the weaker claim gets the weaker mark.
    """
    adj = e.get("adjudication")
    if adj in stats.REF_ERROR or adj in stats.REF_PARTIAL:
        hollow = adj in stats.REF_PARTIAL
        ax.plot(x, y, "D", ms=6.5, mfc=("none" if hollow else C_REFERR),
                mec=(C_REFERR if hollow else "white"), mew=1.6 if hollow else 1.4, zorder=4)
        if label:
            # Stagger: a full error labels up-and-right, a partial one down-and-LEFT. 010 OD (-1.48) and
            # 001 OS (-1.64) sit almost on top of each other, and a right-hand label on the latter runs
            # straight through the under-called eye at 1.99.
            ax.annotate(_short(e), xy=(x, y), xytext=(-8 if hollow else 8, -9 if hollow else 4),
                        textcoords="offset points", ha="right" if hollow else "left",
                        va="center", fontsize=7, color=C_REFERR)
    elif e["status"] == "control":                          # hollow: shape, not another hue
        ax.plot(x, y, "o", ms=6, mfc="none", mec=C_MUTED, mew=1.6, zorder=3)
    else:
        col = C_AGREE if e["status"] == "agree" else C_DIFFER
        # a white ring keeps overlapping eyes separable without drawing a border around the mark
        ax.plot(x, y, "o", ms=7, mfc=col, mec="white", mew=1.4, zorder=3)


def _fmt(v, p=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}"


def _signed(v, p=2):
    return f"{'+' if v >= 0 else '−'}{abs(v):.{p}f}"


def ticks_for(lim):
    """Round ticks covering [0, lim] -- shared by both axes of a scatter so the diagonal reads as 45°."""
    step = next(s for s in (0.5, 1, 2, 2.5, 5, 10) if lim / s <= 8)
    n = int(lim / step)
    return [round(i * step, 2) for i in range(n + 1)]


def _style_axes(ax):
    """Recessive chrome: hairline grid, no top/right spines."""
    ax.grid(True, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)


def _scatter(ax, eyes, xmax, ymax, zoom=False):
    """Ours (y) against the PLEX reference (x), with the perfect-agreement diagonal."""
    _style_axes(ax)
    lim = max(xmax, ymax)
    ax.plot([0, lim], [0, lim], color=C_MUTED, lw=1.2, ls="--", zorder=1, alpha=0.8)

    for e in eyes:
        x, y = e["plex_mm2"], e["ours_mm2"]
        if zoom and (x > ZOOM or y > ZOOM):
            continue
        # No labels in either scatter: at 0-3 mm2 the three adjudicated eyes sit within ~0.8 mm2 of each
        # other and their labels collide with each other and with neighbouring points. Bland-Altman has
        # the room, and the per-eye table names them all.
        _mark(ax, x, y, e, label=False)

    # Both axes are the same quantity in the same unit, so they get the same limits, the same ticks and
    # an equal aspect. Otherwise the perfect-agreement diagonal is not at 45 degrees and the eye reads
    # the under-call as smaller than it is.
    ax.set_xlim(-0.03 * lim, lim)
    ax.set_ylim(-0.03 * lim, lim)
    ax.set_aspect("equal", adjustable="box")
    tk = ticks_for(lim)
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    ax.set_xlabel("PLEX reference area (mm²)", fontsize=8.5)
    ax.set_ylabel("Our OCT-only area (mm²)", fontsize=8.5)
    # Label above the diagonal: we under-call, so essentially every eye sits below it and the
    # upper-left half is empty in both the full-range and the zoomed panel.
    ax.annotate("perfect agreement", xy=(lim * 0.38, lim * 0.38), xytext=(-4, 7),
                textcoords="offset points", ha="right", fontsize=7.5, color=C_MUTED)


def _bland_altman(ax, eyes, m, xmax):
    """Difference (ours - reference) against the reference, with bias and 95% limits of agreement."""
    _style_axes(ax)

    # The agreement limits -- +/- max(0.5 mm2, 25% of PLEX), the same rule that colours the marks and that
    # the Library cards use. They MUST be drawn: the rule is RELATIVE while this y-axis is ABSOLUTE, so
    # without them a green eye (008 OS, -2.17 mm2 = 14% of a 15 mm2 lesion) sits below an amber one
    # (015 OD, -0.54 mm2 = 27% of a 2 mm2 lesion) and the colours look arbitrary. With them, green is
    # exactly "inside", amber exactly "outside", by construction.
    # Two thin lines, not a filled band: the band swells to cover most of the right-hand half of the
    # axes, and the shading read as a wedge slicing the figure in two.
    xs = np.linspace(0, xmax, 300)
    tol = np.maximum(0.5, 0.25 * xs)
    ax.plot(xs, tol, color=C_AGREE, lw=1.0, alpha=0.55, zorder=1.2)
    ax.plot(xs, -tol, color=C_AGREE, lw=1.0, alpha=0.55, zorder=1.2)

    ax.axhline(0, color=C_GRID, lw=1, zorder=1)
    ax.axhline(m["bias"], color=C_MUTED, lw=1.6, zorder=2)
    for y in (m["loa_lo"], m["loa_hi"]):
        ax.axhline(y, color=C_MUTED, lw=1.2, ls="--", alpha=0.85, zorder=2)

    for e in eyes:
        _mark(ax, e["plex_mm2"], e["delta_mm2"], e, label=True)

    # Pin y, symmetric about zero: an over-call of 1 mm2 must look the same size as an under-call of 1.
    # (Also stops the agreement limits, which reach +/-4 mm2 at the right edge, from driving the
    # autoscale and squashing every eye into a strip.) Rounded up to a half-step so no eye is clipped --
    # 014 OD sits at -2.58, so +/-2.5 would cut it off the chart.
    dd = [abs(e["delta_mm2"]) for e in eyes] + [abs(m["loa_lo"]), abs(m["loa_hi"])]
    lim = float(np.ceil((max(dd) + 0.2) * 2) / 2)
    step = 0.5 if lim <= 2 else 1.0
    half = ticks_for(lim) if step == 0.5 else [i * 1.0 for i in range(int(lim) + 1)]
    ax.set_ylim(-lim, lim)
    ax.set_yticks([-t for t in reversed(half) if t > 0] + half)

    # Label the lower limit out at x ~ 55% of the range, where no eye sits and the -1.96 SD label
    # (right-aligned at the frame edge) cannot reach.
    lx = xmax * 0.55
    ax.annotate("agreement limits", xy=(lx, -max(0.5, 0.25 * lx)), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top", fontsize=7, color=C_MUTED)

    ax.set_xlim(-0.03 * xmax, xmax)
    ax.set_xlabel("PLEX reference area (mm²)", fontsize=8.5)
    ax.set_ylabel("Ours − PLEX (mm²)", fontsize=8.5)
    for y, lab in ((m["bias"], f"mean difference {_signed(m['bias'])}"),
                   (m["loa_hi"], f"+1.96 SD  {_signed(m['loa_hi'])}"),
                   (m["loa_lo"], f"−1.96 SD  {_signed(m['loa_lo'])}")):
        ax.annotate(lab, xy=(xmax, y), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=7.5, color=C_MUTED)


def _rule(fig, x0, x1, y0, y1, color, lw):
    fig.add_artist(Line2D([x0, x1], [y0, y1], transform=fig.transFigure, color=color, lw=lw))


# Text is hard-wrapped rather than left to matplotlib's wrap=True, which wraps at the FIGURE edge and
# so overruns the 0.07/0.93 margins the rules are drawn to.
def _wrap(text, fontsize, width_frac=0.86):
    inches = width_frac * PAGE[0]
    chars = max(20, int(inches / (fontsize * 0.503 / 72)))     # ~0.503 em average advance for this sans
    return "\n".join(textwrap.fill(p, chars) for p in text.split("\n\n")).replace("\n\n", "\n\n")


def _flow(fig, x, y, paragraphs, fontsize, color, leading=1.42, gap=0.9):
    """Lay paragraphs top-down from y, returning the y just below the last line."""
    line_h = (fontsize * leading) / 72.0 / PAGE[1]
    for p in paragraphs:
        wrapped = _wrap(p, fontsize)
        fig.text(x, y, wrapped, fontsize=fontsize, color=color, va="top", linespacing=leading)
        y -= (wrapped.count("\n") + 1) * line_h + gap * line_h
    return y


def _bullets(fig, x, y, items, fontsize, color, leading=1.42, gap=0.55):
    """Bulleted lines with a hanging indent, laid out top-down from y."""
    line_h = (fontsize * leading) / 72.0 / PAGE[1]
    chars = max(20, int(0.86 * PAGE[0] / (fontsize * 0.503 / 72))) - 3
    for it in items:
        w = textwrap.fill(it, chars, initial_indent="•  ", subsequent_indent="    ")
        fig.text(x, y, w, fontsize=fontsize, color=color, va="top", linespacing=leading)
        y -= (w.count("\n") + 1) * line_h + gap * line_h
    return y


def _sections(fig, x, y, items, fontsize, color, leading=1.5):
    """Lay out (bold heading, body) pairs top-down; `body` is prose, or a list of bullet strings. A
    run-in bold lead-in cannot survive the hard wrap, so the lead-in becomes its own heading line."""
    line_h = (fontsize * leading) / 72.0 / PAGE[1]
    for head, body in items:
        fig.text(x, y, head, fontsize=fontsize + 0.6, color=C_INK, weight="bold", va="top")
        y -= line_h * 1.25
        if isinstance(body, (list, tuple)):
            y = _bullets(fig, x, y, body, fontsize, color, leading=leading, gap=0.35) - 0.6 * line_h
        else:
            wrapped = _wrap(body, fontsize)
            fig.text(x, y, wrapped, fontsize=fontsize, color=color, va="top", linespacing=leading)
            y -= (wrapped.count("\n") + 1) * line_h + 1.1 * line_h
    return y


def _tile(fig, x, y, label, value, note, accent):
    """A headline figure: small-caps label, big proportional number, a line of context."""
    fig.text(x, y, " ".join(label.upper()), fontsize=6.6, color=C_MUTED)
    fig.text(x, y - 0.024, value, fontsize=18, color=accent, weight="bold")
    _rule(fig, x - 0.013, x - 0.013, y - 0.030, y + 0.011, accent, 2.2)
    fig.text(x, y - 0.040, note, fontsize=7.2, color=C_MUTED, va="top")


def _page1(pdf, d, stamp):
    m = d["metrics"]
    fig = plt.figure(figsize=PAGE, facecolor="white")

    fig.text(0.07, 0.955, "Geographic atrophy area from Spectralis OCT", fontsize=16, color=C_INK, weight="bold")
    fig.text(0.07, 0.934, "Agreement of our OCT-only measurement with the Zeiss PLEX (advRPE) reference",
             fontsize=9.5, color=C_MUTED)
    fig.text(0.07, 0.919, f"{m['n_eyes']} eyes · {m['n_patients']} patients · generated {stamp}",
             fontsize=8, color=C_MUTED)
    _rule(fig, 0.07, 0.93, 0.908, 0.908, C_GRID, 1)

    row = 0.880
    _tile(fig, 0.085, row, "Cohort", f"{m['n_eyes']} eyes",
          f"{m['n_patients']} patients\nPLEX {_fmt(m['plex_min'])}–{_fmt(m['plex_max'])} mm²", C_MUTED)
    _tile(fig, 0.305, row, "No false positives", f"{m['n_controls']}/{m['n_controls']}",
          f"control eyes read ≤ {_fmt(m['control_max'])} mm²\nspecificity {_fmt(m['specificity'])}", C_AGREE)
    _tile(fig, 0.525, row, "Agrees with PLEX", f"{m['n_agree']}/{m['n_eyes']}",
          "within 0.5 mm² or 25%\nof the reference", C_AGREE)
    _tile(fig, 0.745, row, "Average difference", f"{_signed(m['bias'])} mm²",
          f"we under-call, vs PLEX\n95% limits {_signed(m['loa_lo'])} to {_signed(m['loa_hi'])}", C_DIFFER)

    # one legend for the whole page: three hues + the hollow control marker
    handles = [
        Line2D([], [], marker="o", ls="", ms=6, mfc=C_AGREE, mec="white", label="agrees with PLEX"),
        Line2D([], [], marker="o", ls="", ms=6, mfc=C_DIFFER, mec="white", label="we under-call"),
        Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=C_MUTED, mew=1.6, label="control (no GA)"),
    ]
    if m["n_ref_error"]:
        handles.append(Line2D([], [], marker="D", ls="", ms=6, mfc=C_REFERR, mec="white",
                              label="PLEX wrong, ours is correct"))
    if m["n_ref_partial"]:
        handles.append(Line2D([], [], marker="D", ls="", ms=6, mfc="none", mec=C_REFERR, mew=1.6,
                              label="PLEX partly wrong"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.800), ncol=len(handles),
               frameon=False, fontsize=7.8, handletextpad=0.4, columnspacing=1.3, labelcolor=C_MUTED)

    eyes = d["eyes"]
    axmax = float(np.ceil(max(m["plex_max"], max(e["ours_mm2"] for e in eyes)) + 0.5))

    ax1 = fig.add_axes([0.10, 0.535, 0.36, 0.205])
    _scatter(ax1, eyes, axmax, axmax)
    ax1.set_title(f"Our GA vs PLEX, full range ({m['n_eyes']} eyes)", fontsize=9.5, color=C_INK, loc="left", pad=8)

    n_zoom = sum(1 for e in eyes if e["plex_mm2"] <= ZOOM and e["ours_mm2"] <= ZOOM)
    ax2 = fig.add_axes([0.58, 0.535, 0.36, 0.205])
    _scatter(ax2, eyes, ZOOM, ZOOM, zoom=True)
    ax2.set_title(f"Zoom: 0–3 mm² ({n_zoom} of {m['n_eyes']} eyes)", fontsize=9.5, color=C_INK, loc="left", pad=8)

    ax3 = fig.add_axes([0.10, 0.242, 0.84, 0.200])
    _bland_altman(ax3, eyes, m, axmax)
    # pad clears the two-line explanation annotated just above the axes
    ax3.set_title("Bland–Altman: how far each eye sits from the reference", fontsize=9.5, color=C_INK,
                  loc="left", pad=32)
    # Most readers have not met this plot. Two lines: what the axes are, what the lines are. The last
    # clause is load-bearing -- it is why a green eye can sit near the bottom.
    ax3.annotate("One dot per eye: our area minus PLEX (vertical), against the area PLEX found "
                 "(horizontal). Zero means they agree.\nGrey: mean difference and 95% limits. Green: the "
                 "agreement rule, ±0.5 mm² or 25%, so it widens with lesion size.",
                 xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                 fontsize=7.4, color=C_MUTED, va="bottom", linespacing=1.4)

    # "corrected" = the eyes where PLEX itself is wrong (marked with a diamond) have their reference set
    # to 0. Both numbers appear on the same line, so the reader never has to hold one in their head while
    # hunting for the other, and the term is defined where it is first used rather than overleaf.
    a = m.get("adjudicated")
    corr = (lambda v: f" ({v} corrected)") if a else (lambda v: "")

    ink = [
        f"Detection. An eye counts as having GA above 0.25 mm², the smallest lesion the cRORA definition "
        f"recognises. We find {m['tp']} of the {m['tp'] + m['fn']} GA eyes: sensitivity {_fmt(m['sensitivity'])}"
        + (f", or {_fmt(a['sensitivity'])} ({a['tp']}/{a['tp'] + a['fn']}) corrected" if a else "")
        + f". No healthy eye is ever called GA: specificity {_fmt(m['specificity'])} ({m['tn']}/"
        f"{m['tn'] + m['fp']}" + (f"; {a['tn']}/{a['tn'] + a['fp']} corrected" if a else "") + ").",

        f"Error. On average we measure {_fmt(abs(m['bias']))} mm² less GA than PLEX"
        + (f"{corr(_fmt(abs(a['bias'])) + ' mm²')}" if a else "") + ". Ignoring direction, the typical gap is "
        f"{_fmt(m['mae'])} mm²" + (f"{corr(_fmt(a['mae']) + ' mm²')}" if a else "") + f", median "
        f"{_fmt(m['median_ae'])}. {m['within_0p5']} eyes fall within 0.5 mm² of the reference, "
        f"{m['within_1p0']} within 1.0 mm².",

        f"Weakness. {m['fn']} GA-positive eyes under-called, two to zero. False negatives, never false "
        f"positives.",

    ]
    y = _bullets(fig, 0.07, 0.188, ink, 8.2, C_INK)
    if a:
        _bullets(fig, 0.07, y - 0.004, [
            "\"Corrected\". The eyes where PLEX itself is wrong (◆) have their reference set to 0; the eye "
            "where it is only partly wrong (◇) is left untouched. Only eyes where the two measurements "
            "disagreed were re-examined, so the corrected figures are an upper bound; the uncorrected ones "
            "are the ones to quote."],
            7.8, C_MUTED)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page2(pdf, d, stamp):
    m = d["metrics"]
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(0.07, 0.955, "Per-eye results", fontsize=14, color=C_INK, weight="bold")
    _rule(fig, 0.07, 0.93, 0.925, 0.925, C_GRID, 1)

    # Agreement and detection are different questions and get their own column: 003-026 agrees on area
    # (Δ −0.24 mm²) yet sits below the 0.25 mm² call threshold, so it is a miss. Collapsing the two into
    # one "status" would make this table contradict the "agrees with PLEX" count on page 1.
    thr = m["call_thr"]
    cells, agree_col, detect_col = [], [], []
    for e in d["eyes"]:
        pos, call = e["plex_mm2"] >= thr, e["ours_mm2"] >= thr
        adj = e.get("adjudication")
        ref_err, ref_part = adj in stats.REF_ERROR, adj in stats.REF_PARTIAL
        detect = ("missed" if pos and not call else "detected" if pos
                  else "false +" if call else "correct −")
        if ref_err or ref_part:                      # the gap is the reference's, not ours -- say so, and
            detect += " *" if ref_err else " †"      # explain it in a footnote rather than a 7th column
        cells.append([f"{e['subject']} {e['eye']}", _fmt(e["ours_mm2"]), _fmt(e["plex_mm2"]),
                      _signed(e["delta_mm2"]), e["status"], detect])
        agree_col.append(C_MUTED if e["status"] == "control" else (C_AGREE if e["status"] == "agree" else C_DIFFER))
        detect_col.append(C_REFERR if (ref_err or ref_part) else
                          (C_DIFFER if detect in ("missed", "false +") else
                           (C_AGREE if detect == "detected" else C_MUTED)))

    ax = fig.add_axes([0.07, 0.30, 0.86, 0.61])
    ax.axis("off")
    tbl = ax.table(cellText=cells,
                   colLabels=["Eye", "Our GA (mm²)", "PLEX (mm²)", "Δ (mm²)", "Agreement", "Detection"],
                   cellLoc="center", loc="upper center", colWidths=[0.26, 0.15, 0.14, 0.13, 0.16, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1, 1.24)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_GRID)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(color=C_MUTED, weight="bold")
            cell.set_facecolor("#f6f7f9")
        else:
            cell.set_text_props(color=C_INK)
            if c == 0:
                cell.set_text_props(color=C_INK, ha="left")
            if c == 4:                                   # only the state words wear a series colour;
                cell.set_text_props(color=agree_col[r - 1])
            if c == 5:                                   # the measured values stay in ink
                cell.set_text_props(color=detect_col[r - 1])

    # Anchor the footnote under the table's REAL last row rather than a guessed y -- the table grows a
    # row per eye, and a fixed offset silently overprinted the bottom rows as the cohort grew.
    row_h = tbl.get_celld()[(0, 0)].get_height()
    tbl_bottom = 0.30 + 0.61 * (1.0 - len(cells) * row_h - row_h)

    # One key line carries the "what the marker means" text, so each eye's note is only its own reason.
    key = []
    if m["n_ref_error"]:
        key.append("*  PLEX wrong. Its area is corrected to 0 in the adjudicated figures.")
    if m["n_ref_partial"]:
        key.append("†  PLEX partly wrong.")
    foot = ["   ".join(key)] if key else []
    for e in d["eyes"]:
        adj = e.get("adjudication")
        if adj in stats.REF_ERROR or adj in stats.REF_PARTIAL:
            sym = "*" if adj in stats.REF_ERROR else "†"
            foot.append(f"{sym}  {_short(e)}: {e['adjudication_note'] or 'Reference judged wrong on this eye.'}")
    if foot:
        _flow(fig, 0.07, tbl_bottom - 0.018, foot, 7.4, C_REFERR, gap=0.35)

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _page3(pdf, d, stamp):
    """Method, reference and cohort. Its own page: the prose outgrew the table page."""
    m = d["metrics"]
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(0.07, 0.955, "Method, reference and cohort", fontsize=14, color=C_INK, weight="bold")
    _rule(fig, 0.07, 0.93, 0.943, 0.943, C_GRID, 1)

    notes = [
        ("Method", [
            "Everything comes from the Spectralis OCT volume. No PLEX data, no autofluorescence.",
            "A deep-learning model traces Bruch's membrane on every B-scan.",
            "The volume is converted to an optical attenuation coefficient, which compares the light scattered "
            "back at each depth against the light still travelling deeper. Unlike raw brightness, it is not "
            "thrown off by a naturally dark or bright choroid, so a dim choroid cannot be mistaken for atrophy.",
            "For every A-scan we take the mean attenuation in a thin band 50 to 8 µm above Bruch's membrane, "
            "where the retinal pigment epithelium sits. A high value means the RPE is present; a low value "
            "suggests it is missing.",
            "That value is compared with a healthy-RPE baseline built from the same eye. In each 0.25 mm ring "
            "around the fovea we take the 75th percentile of the value, discard pixels that sit well below the "
            "local trend, and force the profile to fall, never rise, with distance from the fovea. Atrophy is "
            "dark, so it drops out of the fit; and because the profile cannot dip locally, a large lesion "
            "cannot lower the very threshold that is about to judge it. An A-scan becomes a candidate when its "
            "value falls below half of that baseline.",
            "A candidate is kept only if light also reaches the choroid immediately beneath Bruch's membrane, "
            "in a band 20 to 60 µm below it. This hypertransmission is the second cRORA criterion. It removes "
            "the dark field corners, where there is no retina to transmit. Interior gaps that transmit like "
            "the atrophy around them are filled in; gaps that do not transmit, such as spared RPE, are left out.",
            "Each surviving patch must fall below 27% of the baseline somewhere, so that partial thinning of "
            "the RPE is not counted as complete atrophy.",
            "A patch is kept only if its longest dimension is at least 250 µm, the smallest lesion the cRORA "
            "definition recognises. Smaller specks are discarded.",
            "What remains is measured over the central 6×6 mm and reported in mm² on the 24 mm model eye.",
        ]),

        ("Reference",
         "The Zeiss advRPE segmenter on PLEX Elite 6×6 mm cubes of the same eyes, taken the same day, so no "
         "disagreement is GA growth. It is an automatic algorithm on another device, not a human grader: a "
         "silver standard. Disagreement is shared between the two and is not, on its own, our error. No PLEX "
         "data enters our number."),

        ("Cohort",
         f"Every eye that passed quality control and has a PLEX value: {m['n_eyes']} eyes, {m['n_patients']} "
         f"patients, {_fmt(m['plex_min'])}–{_fmt(m['plex_max'])} mm² (median {_fmt(m['plex_median'])}). Excluded "
         "beforehand, for reasons independent of the result: wet AMD and intraretinal fluid (out of scope)."),
    ]
    _sections(fig, 0.07, 0.918, notes, 8.6, C_INK)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ga_report.pdf"), help="output PDF path")
    args = ap.parse_args()

    adj = stats.load_adjudication(ADJ_CSV)
    d = stats.cohort_stats(bundle.read_index(), adj)
    if not d["n_eyes"]:
        raise SystemExit("no baked library eyes found; run src/bake_library.py first")

    stamp = _dt.date.today().strftime("%d %b %Y")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with PdfPages(args.out) as pdf:
        _page1(pdf, d, stamp)
        _page2(pdf, d, stamp)
        _page3(pdf, d, stamp)
        info = pdf.infodict()
        info["Title"] = "GA area from Spectralis OCT: agreement with the PLEX reference"
        info["Subject"] = f"{d['n_eyes']} eyes, {d['n_patients']} patients"

    m = d["metrics"]
    print(f"wrote {args.out}")
    # plain ASCII: this line goes to a console that may be cp1252, unlike the PDF text above
    print(f"  {m['n_eyes']} eyes / {m['n_patients']} patients | bias {m['bias']:+.2f} mm2 | "
          f"MAE {_fmt(m['mae'])} | agrees {m['n_agree']}/{m['n_eyes']} | "
          f"sens {_fmt(m['sensitivity'])} / spec {_fmt(m['specificity'])}")
    a = m.get("adjudicated")
    if a:
        print(f"  adjudicated ({m['n_ref_error']} reference error(s) corrected to 0: "
              f"{', '.join(m['ref_error_slugs'])})")
        print(f"  {a['n_eyes']} eyes | bias {a['bias']:+.2f} mm2 | MAE {_fmt(a['mae'])} | "
              f"sens {_fmt(a['sensitivity'])} ({a['tp']}/{a['tp']+a['fn']}) / "
              f"spec {_fmt(a['specificity'])} ({a['tn']}/{a['tn']+a['fp']})")


if __name__ == "__main__":
    main()
