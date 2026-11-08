#!/usr/bin/env python3
"""Draws the figures of the rigid-water template (Figures 4.1 to 4.3 of
Section 4.7 of the white paper; D[rigid-water-template], #774): which
distances the structure tests, the work for one pair of waters, and the
lists and the launch. The geometry of OPC is to scale; the counts of
steps and the times of Figure 4.3 are those of the stage 2 prototype of
#774 (RTX 4090, mixed precision), to be replaced by the implementation's.

    plot-rigid-water.py OUTDIR

Writes OUTDIR/rigid-water-{members,pair-work,lists}.svg and converts each
to a PNG at twice its size with rsvg-convert (librsvg), which must be on
the path. The paper takes the PNGs (docs/paper/figures), as the build of
the PDF copies only those.
"""
import math
import os
import shutil
import subprocess
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
OUT = sys.argv[1]
INK, MUTED, FAINT, ACC, ACC2, BG = "#1f2328", "#6e7781", "#d0d7de", "#0b62c4", "#c2410c", "#ffffff"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
# OPC: r_OH 0.8724 A, HOH 103.6 deg, extra point 0.1594 A along the bisector
ROH, ANG, DM = 0.8724, math.radians(103.6), 0.1594

def water(cx, cy, s, rot):
    """Site positions (px) of one OPC water with O at (cx, cy), bisector at angle rot."""
    h = ANG / 2
    p = {"O": (cx, cy)}
    for name, sg in (("H1", -1), ("H2", 1)):
        a = rot + sg * h
        p[name] = (cx + s * ROH * math.cos(a), cy + s * ROH * math.sin(a))
    p["M"] = (cx + s * DM * math.cos(rot), cy + s * DM * math.sin(rot))
    return p

def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" font-size="13" fill="{INK}">\n<rect width="{w}" height="{h}" fill="{BG}"/>\n'
            + "\n".join(body) + "\n</svg>\n")

def t(x, y, s, size=13, anchor="middle", fill=INK, weight="normal", style="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')

def line(a, b, col=INK, w=1.2, dash=None, op=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{col}" stroke-width="{w}"{d} opacity="{op}"/>'

def mol(p, s):
    out = [line(p["O"], p["H1"], MUTED, 2.2), line(p["O"], p["H2"], MUTED, 2.2)]
    out.append(f'<circle cx="{p["O"][0]:.1f}" cy="{p["O"][1]:.1f}" r="{0.30*s:.1f}" fill="#e5484d" stroke="{INK}" stroke-width="1"/>')
    for k in ("H1", "H2"):
        out.append(f'<circle cx="{p[k][0]:.1f}" cy="{p[k][1]:.1f}" r="{0.17*s:.1f}" fill="#ffffff" stroke="{INK}" stroke-width="1"/>')
    out.append(f'<circle cx="{p["M"][0]:.1f}" cy="{p["M"][1]:.1f}" r="{0.07*s:.1f}" fill="{INK}"/>')
    return out

def sub(base, s):
    return f'{base}<tspan baseline-shift="sub" font-size="9">{s}</tspan>'

# ---------- Figure 1: which distances are tested ----------
def fig1():
    W, H, s = 860, 360, 62
    body = []
    for k, (x0, title, sub_) in enumerate(((0, "General structure: every site is a member", "16 tests per pair of OPC waters"),
                                         (430, "Template kernel: the oxygen is the only member", "1 test per pair of waters"))):
        body.append(t(x0 + 215, 28, title, 15, weight="bold"))
        body.append(t(x0 + 215, 48, sub_, 13, fill=MUTED))
        A = water(x0 + 95, 190, s, math.radians(200))
        B = water(x0 + 320, 200, s, math.radians(-15))
        if k == 0:
            for a in ("O", "H1", "H2", "M"):
                for b in ("O", "H1", "H2", "M"):
                    body.append(line(A[a], B[b], ACC, 1.0, "4 3", 0.75))
        else:
            rm = ROH * s
            for P, lab in ((A, "A"), (B, "B")):
                body.append(f'<circle cx="{P["O"][0]:.1f}" cy="{P["O"][1]:.1f}" r="{rm:.1f}" fill="none" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
            body.append(line(A["O"], B["O"], ACC, 2.4))
            mx, my = (A["O"][0] + B["O"][0]) / 2, (A["O"][1] + B["O"][1]) / 2
            body.append(t(mx, my - 12, "d(O, O) &lt; R + 2" + sub("r", "m"), 13, fill=ACC, weight="bold"))
            # radius label on B
            ang = math.radians(-60)
            e = (B["O"][0] + rm * math.cos(ang), B["O"][1] + rm * math.sin(ang))
            body.append(line(B["O"], e, MUTED, 1))
            body.append(t(e[0] + 18, e[1] - 2, sub("r", "m"), 13, fill=MUTED))
            # a guaranteed site pair
            body.append(line(A["H1"], B["H2"], ACC2, 1.4, "5 3"))
            body.append(t(x0 + 215, 318, "every site lies within " + sub("r", "m") + " of its oxygen, so", 12, fill=MUTED))
            body.append(t(x0 + 215, 338, "|x" + "<tspan baseline-shift='sub' font-size='9'>a</tspan> − x<tspan baseline-shift='sub' font-size='9'>b</tspan>| ≥ d(O, O) − 2" + sub("r", "m") + ": no site pair within r<tspan baseline-shift='sub' font-size='9'>c</tspan> is missed", 12, fill=ACC2))
        body += mol(A, s) + mol(B, s)
        if k == 0:
            body.append(t(x0 + 215, 318, "each site is sorted, tested for displacement,", 12, fill=MUTED))
            body.append(t(x0 + 215, 338, "and paired with every site of the other molecule", 12, fill=MUTED))
    body.append(line((430, 20), (430, 345), FAINT, 1))
    # legend
    lx, ly = 300, 82
    return svg(W, H, body)

# ---------- Figure 2: work per pair of molecules ----------
def grid(x0, y0, sites, cells, cs=34, title="", note=""):
    out = [t(x0 + cs * len(sites) / 2 + 20, y0 - 34, title, 14, weight="bold")]
    for i, a in enumerate(sites):
        out.append(t(x0 + 10, y0 + cs * i + cs / 2 + 5, a, 12, "end", MUTED))
        out.append(t(x0 + 20 + cs * i + cs / 2, y0 - 8, a, 12, "middle", MUTED))
    for i, a in enumerate(sites):
        for j, b in enumerate(sites):
            kind = cells(a, b)
            x, y = x0 + 20 + cs * j, y0 + cs * i
            fill = {"both": "#cfe0f5", "coul": "#cfe0f5", "lj": "#fde2cf", "none": "#ffffff", "full": "#e6e9ed"}[kind]
            out.append(f'<rect x="{x}" y="{y}" width="{cs-2}" height="{cs-2}" rx="3" fill="{fill}" stroke="{FAINT}"/>')
            lab = {"both": "C+LJ", "coul": "C", "lj": "LJ", "none": "", "full": "C+LJ"}[kind]
            col = {"both": ACC, "coul": ACC, "lj": ACC2, "none": MUTED, "full": MUTED}[kind]
            if kind == "both":
                out.append(t(x + cs / 2 - 1, y + cs / 2 + 4, "C", 11, fill=ACC, weight="bold"))
                out.append(t(x + cs / 2 - 1, y + cs / 2 + 14, "LJ", 9, fill=ACC2, weight="bold"))
            elif lab:
                out.append(t(x + cs / 2 - 1, y + cs / 2 + 4, lab, 10 if kind == "full" else 11, fill=col, weight="bold"))
    for k, part in enumerate(note.split("|")):
        out.append(t(x0 + cs * len(sites) / 2 + 20, y0 + cs * len(sites) + 22 + 17 * k, part, 12, fill=MUTED))
    return out

def fig2():
    W, H = 860, 360
    body = [t(430, 26, "Work for one pair of waters (rows: sites of A, columns: sites of B)", 15, weight="bold")]
    S4, S3 = ["O", "H1", "H2", "M"], ["O", "H1", "H2"]
    body += grid(40, 90, S4, lambda a, b: "full", title="OPC, general",
                 note="16 general pairs|charges and types read per pair")
    def opc(a, b):
        if a == "O" and b == "O": return "lj"
        if a == "O" or b == "O": return "none"
        return "coul"
    body += grid(320, 90, S4, opc, title="OPC, template kernel", note="9 Coulomb + 1 LJ|no parameter reads, no intramolecular exclusion checks")
    def tip(a, b):
        return "both" if a == b == "O" else "coul"
    body += grid(610, 90, S3, tip, title="TIP3P, template kernel", note="9 Coulomb + 1 LJ|no parameter reads, no intramolecular exclusion checks")
    # key
    for i, (fill, lab) in enumerate((("#cfe0f5", "Coulomb (constant charges)"), ("#fde2cf", "Lennard-Jones (O–O only)"),
                                     ("#ffffff", "skipped: the term is zero"), ("#e6e9ed", "general pair: both terms, parameters loaded"))):
        x, ky = 140 + (i % 2) * 330, 308 + (i // 2) * 22
        body.append(f'<rect x="{x}" y="{ky-11}" width="14" height="14" rx="2" fill="{fill}" stroke="{FAINT}"/>')
        body.append(t(x + 20, ky, lab, 11, "start", MUTED))
    return svg(W, H, body)

# ---------- Figure 3: lists and one launch ----------
def box(x, y, w, h, fill, stroke=FAINT, rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'

def arrow(*pts, col=MUTED):
    """An orthogonal connector through the given points, with a head at the last."""
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#ah)"/>'

def fig3():
    W, H = 860, 400
    body = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" markerUnits="strokeWidth" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']
    body.append(t(430, 26, "One sort, two lists, one launch (ubiquitin, OPC, 0.15 M KCl; stage 2 prototype)", 15, weight="bold"))
    # column 1: one sort into two kinds of cluster
    body.append(box(30, 60, 220, 300, "#f6f8fa"))
    body.append(t(140, 82, "one spatial sort", 13, weight="bold"))
    body.append(box(50, 100, 180, 110, "#ffffff"))
    body.append(t(140, 120, "water clusters", 12, weight="bold", fill=ACC))
    body.append(t(140, 138, "members: oxygens (anchors)", 11, fill=MUTED))
    for i in range(6):
        cx, cy = 75 + i * 26, 172
        p = water(cx, cy, 13, math.radians(-90 + 37 * i))
        body += mol(p, 13)
    body.append(box(50, 225, 180, 115, "#ffffff"))
    body.append(t(140, 245, "solute clusters", 12, weight="bold", fill=ACC2))
    body.append(t(140, 263, "members: atoms and ions", 11, fill=MUTED))
    pts = [(70, 300), (92, 288), (114, 304), (136, 290), (158, 306), (180, 292), (202, 308)]
    body.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in pts) + f'" fill="none" stroke="{MUTED}" stroke-width="2"/>')
    for x, y in pts:
        body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#ffffff" stroke="{INK}"/>')
    body.append(f'<circle cx="205" cy="325" r="5" fill="#8b5cf6" stroke="{INK}"/>')
    body.append(t(190, 329, "K⁺", 10, "end", MUTED))
    # column 2: two lists
    body.append(box(300, 60, 250, 300, "#f6f8fa"))
    body.append(t(425, 82, "two lists", 13, weight="bold"))
    body.append(box(320, 100, 210, 110, "#ffffff"))
    body.append(t(425, 120, "water list (WW)", 12, weight="bold", fill=ACC))
    body.append(t(425, 140, "tested on the oxygens only", 11, fill=MUTED))
    body.append(t(425, 165, "built every ~55 steps", 12))
    body.append(t(425, 184, "pruned every ~8.5 steps", 12))
    body.append(box(320, 225, 210, 115, "#ffffff"))
    body.append(t(425, 245, "solute lists (SW, SS)", 12, weight="bold", fill=ACC2))
    body.append(t(425, 265, "tested on oxygens and atoms", 11, fill=MUTED))
    body.append(t(425, 290, "built every ~40 steps", 12))
    body.append(t(425, 309, "pruned every ~3 steps", 12))
    body.append(t(425, 380, "general groups: built every 20.7, pruned every 2.8", 11, fill=MUTED))
    body.append(arrow((230, 155), (318, 155)))
    body.append(f'<circle cx="270" cy="155" r="2.5" fill="{MUTED}"/>')
    body.append(arrow((270, 155), (270, 262), (318, 262)))
    body.append(arrow((230, 300), (318, 300)))
    # column 3: one launch
    body.append(box(600, 60, 230, 300, "#f6f8fa"))
    body.append(t(715, 82, "one launch", 13, weight="bold"))
    body.append(box(620, 110, 190, 120, "#ffffff", ACC))
    body.append(t(715, 135, "fused kernel", 12, weight="bold"))
    body.append(t(800, 148, "share of pairs", 9, "end", MUTED))
    for i, (lab, col, frac) in enumerate((("WW water–water", ACC, "93.3%"), ("SW solute–water", ACC2, "4.1%"), ("SS solute–solute", ACC2, "2.3%"))):
        body.append(t(632, 166 + 22 * i, lab, 11, "start", col, "bold"))
        body.append(t(800, 166 + 22 * i, frac, 11, "end", MUTED))
    body.append(arrow((530, 162), (618, 162)))
    body.append(arrow((530, 282), (575, 282), (575, 205), (618, 205)))
    body.append(t(715, 262, "short-range path per step (prototype)", 11, fill=MUTED))
    body.append(t(715, 284, "one launch: 68.0 µs", 13, fill=ACC, weight="bold"))
    body.append(t(715, 304, "three launches: 87.8 µs", 12, fill=MUTED))
    body.append(t(715, 324, "(small launches wait on latency)", 11, fill=MUTED))
    return svg(W, H, body)

converter = shutil.which("rsvg-convert")
if converter is None:
    sys.exit("rsvg-convert (librsvg) is not on the path")
os.makedirs(OUT, exist_ok=True)
for name, f in (("rigid-water-members", fig1), ("rigid-water-pair-work", fig2), ("rigid-water-lists", fig3)):
    svg_path = os.path.join(OUT, name + ".svg")
    with open(svg_path, "w", encoding="utf-8") as out:
        out.write(f())
    subprocess.run([converter, "--zoom=2", "-o", os.path.join(OUT, name + ".png"), svg_path], check=True)
    print(os.path.join(OUT, name + ".png"))
