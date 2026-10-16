"""The whole step from the logs of mdir_job.py (D266, #334):
the mean and spread of ms/step of each run over the rounds after the
first, from the lines `round K TAG MS`, so that a job that ran out of time
before its profiled runs still counts the rounds it finished.

ws_table.py LOG...    one table a log: tag, rounds, mean, standard deviation
"""
import re, statistics, sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = defaultdict(list)
    device = ''
    for line in open(path):
        m = re.match(r'round (\d+) (\S+) ([0-9.]+)$', line.strip())
        if m and int(m[1]) > 0:
            rows[m[2]].append(float(m[3]))
        g = re.search(r'gpu-run: job \d+ on \S+ at .*: (.*), 0 %', line)
        if g:
            device = g[1]
    print(f'## {path}: {device}')
    for tag, v in rows.items():
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f'{tag} {len(v)} {statistics.mean(v):.4f} {sd:.4f}')
