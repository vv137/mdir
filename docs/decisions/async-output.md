# D[async-output]

Recorded at its merge.

**The frames of a trajectory are copied from the device and written while the steps go on: the call of the host stages the positions and the numbers of the particles into a ring of buffers of the host, by copies on a stream of their own on a device, and one thread writes the frames from them in the order of the steps. Every write is done before a checkpoint is staged, before the files close, at a stop on a signal or the wall time, before a failure ends the run, at the end of each part of a Python simulation, and at the exit of the process.** #113. *Draft: the implementation and its measurements follow.*
