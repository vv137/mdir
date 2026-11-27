### Fixed

- With `[execution] step_graph = "GRAPH"`, a run continued from a checkpoint at a step that is not a multiple of an interval writes its energy rows, frames, checkpoints and thermostat couplings at the steps of `step_graph = "LAUNCH"`, counted from the start step. Graph replay had aligned them to multiples of the absolute step, so the log dropped every row and the frames, checkpoints and couplings fell on other steps (#1014).
