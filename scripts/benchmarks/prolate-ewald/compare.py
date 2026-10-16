"""Markdown table: per system and target, the best Gaussian (MDIR's layouts:
brick or direct), the best Gaussian with tiles (issue #336), and the best
ESP (any layout), by the time of md.reciprocal (sequence, f32).
compare.py TAG [prec]  (reads proto-*-TAG.jsonl and proto2-*-TAG.jsonl)"""
import json, glob, sys, os
D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
tag = sys.argv[1]
prec = sys.argv[2] if len(sys.argv) > 2 else 'f32'
rows = []
for f in glob.glob(f'{D}proto-*-{tag}.jsonl') + glob.glob(f'{D}proto2-*-{tag}.jsonl'):
    rows += [json.loads(l) for l in open(f)]
rows = [r for r in rows if r['prec'] == prec]
print(f'| System | Δ ≤ | Gaussian, bricks (order, grid, µs) | Gaussian, tiles (#336) | ESP (P, grid, layout, µs) | ESP / Gaussian bricks |')
print('|---|---|---|---|---|---|')
for name in ['jac', 'cel', 'stmv']:
    for t in [0.001, 0.0003, 0.0001, 1e-05]:
        sel = [r for r in rows if r['case']['name'] == name and r['case']['target'] == t]
        if not sel:
            continue
        g = [r for r in sel if r['case']['key'].startswith('gauss') and r['layout'] in ('brick', 'direct')]
        gt = [r for r in sel if r['case']['key'].startswith('gauss')]
        e = [r for r in sel if r['case']['key'].startswith('esp')]
        bg = min(g, key=lambda r: r['sequence_us']); bgt = min(gt, key=lambda r: r['sequence_us']); be = min(e, key=lambda r: r['sequence_us'])
        fmt = lambda r: f"{r['P']}, {'×'.join(map(str, r['K']))}, {r['sequence_us']:.0f}"
        print(f"| {name} | {t:g} | {fmt(bg)} | {bgt['sequence_us']:.0f} ({bgt['P']}, {bgt['layout']}) | {be['P']}, {'×'.join(map(str, be['K']))}, {be['layout']}, {be['sequence_us']:.0f} | {be['sequence_us'] / bg['sequence_us']:.2f} |")
