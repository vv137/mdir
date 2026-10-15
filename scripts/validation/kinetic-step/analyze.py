#!/usr/bin/env python3
"""The means of the logs that run.py writes, as a Markdown table.

    scripts/validation/kinetic-step/analyze.py LOG... [--skip 0.02]
        [--reference LOG]

For each log: the means of the temperature, the potential energy, the
pressure, and the volume of the log's rows past the first --skip of the
run (0.02 = 100 ps of 5 ns), each with the standard error of the mean from
its integrated autocorrelation time (Sokal's window, c = 5); the drift of
the conserved energy, the slope of a straight line through it, in
kcal/mol/ns; and the temperatures of the solvent and the solute that the
end of the log reports (D203). With --reference, the differences of the
potential energy and the volume from those of that log, with the error of
the difference.

Needs numpy.
"""
import argparse
import math
import os
import re

import numpy as np


def read(path):
    header, rows = None, []
    summary = ''
    for line in open(path):
        if line.startswith('INFO:') and 'STEP' in line:
            header = line.split()[1:]
        elif line.startswith('INFO:') and header:
            fields = line.split()[1:]
            if len(fields) == len(header):
                try:
                    rows.append([float(v) for v in fields])
                except ValueError:
                    pass
        elif 'the temperatures over' in line:
            summary = line.strip()
    data = np.array(rows)
    return {name: data[:, k] for k, name in enumerate(header)}, summary


def mean_error(x):
    """The mean and its standard error, from the integrated
    autocorrelation time with Sokal's window."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return float(np.mean(x)), math.nan
    d = x - x.mean()
    f = np.fft.rfft(d, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for m in range(1, n):
        tau += acf[m]
        if m >= 5 * tau:
            break
    variance = d.var()
    return float(x.mean()), math.sqrt(2 * tau * variance / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--skip', type=float, default=0.02)
    parser.add_argument('--reference')
    args = parser.parse_args()

    def stats(path):
        columns, summary = read(path)
        start = int(len(columns['STEP']) * args.skip)
        out = {}
        for name in ('TEMPERATURE', 'POTENTIAL_ENE', 'PRESSURE', 'VOLUME'):
            if name in columns:
                out[name] = mean_error(columns[name][start:])
        if 'CONSERVED' in columns:
            t = columns['TIME'][start:] * 1e-3  # ns
            e = columns['CONSERVED'][start:]
            out['DRIFT'] = float(np.polyfit(t, e, 1)[0]) if len(t) > 2 \
                else math.nan
        solvent = re.search(r'of the solute ([0-9.]+) ± ([0-9.]+), of the '
                            r'solvent ([0-9.]+) ± ([0-9.]+); from the '
                            r'velocities of the steps alone, ([0-9.]+) ± '
                            r'[0-9.]+ and ([0-9.]+)', summary)
        if solvent:
            out['SOLUTE'] = (float(solvent.group(1)), float(solvent.group(2)))
            out['SOLVENT'] = (float(solvent.group(3)),
                              float(solvent.group(4)))
            out['SOLUTE_STEP'] = float(solvent.group(5))
            out['SOLVENT_STEP'] = float(solvent.group(6))
        return out

    reference = stats(args.reference) if args.reference else None
    print('| Run | T (K) | ⟨U⟩ (kcal/mol) | ⟨P⟩ (atm) | ⟨V⟩ (Å³) | '
          'conserved drift (kcal/mol/ns) | solvent T | solute T | '
          'ΔU | ΔV |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for path in args.logs:
        s = stats(path)

        def show(key, digits=2):
            if key not in s:
                return ''
            m, e = s[key]
            return f'{m:.{digits}f} ± {e:.{digits}f}'

        def difference(key, digits=2):
            if not reference or key not in s or key not in reference:
                return ''
            m, e = s[key]
            r, f = reference[key]
            return f'{m - r:+.{digits}f} ± {math.hypot(e, f):.{digits}f}'

        drift = s.get('DRIFT', math.nan)
        print(f"| {os.path.basename(path)[:-4]} | {show('TEMPERATURE')} | "
              f"{show('POTENTIAL_ENE')} | {show('PRESSURE', 1)} | "
              f"{show('VOLUME', 1)} | "
              f"{'' if math.isnan(drift) else f'{drift:.3g}'} | "
              f"{show('SOLVENT')} | {show('SOLUTE')} | "
              f"{difference('POTENTIAL_ENE')} | {difference('VOLUME', 1)} |")


if __name__ == '__main__':
    main()
