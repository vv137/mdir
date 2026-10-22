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
