"""GPU job: time the prototype on the cases of a system, every variant.
proto_job.py CASES_JSON OUT_JSONL [quick]"""
import json, subprocess, sys, os

D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
cases = json.load(open(sys.argv[1]))
out = open(sys.argv[2], 'w')
quick = len(sys.argv) > 3 and sys.argv[3] == 'quick'
layouts = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != 'quick' else ['brick', 'direct']
precs = sys.argv[4].split(',') if len(sys.argv) > 4 else ['f', 'd']
subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.sm,power.limit', '--format=csv'], check=True)
fdir = D + 'forces/'
os.makedirs(fdir, exist_ok=True)
fail = 0
for c in cases:
    P = c.get('P', c.get('order'))
    lanes = [4, 8, 16] if P == 4 else [8, 16, 32]
    variants = []
    for prec in precs:
        for layout in layouts:
            for order in ['input', 'sorted']:
                for gl in lanes:
                    if quick and (layout == 'direct' or gl != lanes[1]):
                        continue
                    if prec == 'd' and order == 'input':
                        continue
                    if layout.startswith('tiled') and order == 'input':
                        continue  # the bins order the particles themselves
                    variants.append((prec, layout, gl, order))
    for prec, layout, gl, order in variants:
        tag = os.path.basename(c['file'])[:-4]
        fo = f'{fdir}{tag}-{prec}-{layout}.f64' if (order == 'sorted' and gl == lanes[1]) else '-'
        r = subprocess.run([D + os.environ.get('BENCH', 'esp_bench'), c['file'], prec, '200', layout, str(gl), order, fo], capture_output=True, text=True)
        if r.returncode != 0:
            print('FAILED', c['file'], prec, layout, gl, order, r.stderr, flush=True)
            fail += 1
            continue
        d = json.loads(r.stdout)
        d.update(case=c, forces=fo)
        out.write(json.dumps(d) + '\n')
        out.flush()
        print(tag, prec, layout, gl, order, 'seq', d['sequence_us'], 'spread', d['spread_us'], 'gather', d['gather_us'], 'fft', d['fft_fwd_us'] + d['fft_bwd_us'], flush=True)
sys.exit(1 if fail else 0)
