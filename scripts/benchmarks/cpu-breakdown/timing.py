"""Read reference-engine timing quantities without rounded-rate conversion."""

import re


def pmemd_all_step_ms(output):
    """Return pmemd's reported time per step for the complete production run."""
    rates = re.findall(
        r"Average timings for all steps:.*?Per Step\(ms\)\s*=\s*([0-9.]+)",
        output, flags=re.S,
    )
    if len(rates) != 1:
        raise ValueError("pmemd did not report one all-steps time per step")
    return float(rates[0])


def gromacs_step_ms(output):
    """Return the ms/step column of GROMACS's Performance line."""
    rates = re.findall(
        r"^\s*Performance:\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
        output, flags=re.M,
    )
    if len(rates) != 1:
        raise ValueError("GROMACS did not report one Performance ms/step")
    return float(rates[0])
