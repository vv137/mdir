### Performance

- **Frames of a trajectory no longer stall the steps** (D400, #113). A run of dynamics, in `mdir run` and in a Python simulation, stages each frame's copy on a stream of its own and writes the frames on a thread beside the steps; the files are the same, byte for byte, and checkpoints, stops, and the end of a run wait for the frames queued before them. On JAC on an RTX 4090 a DCD frame every 10 steps now adds about 1% to the run, where it added 21%, and an XTC frame every 10 steps about 1%, where it added 48%.
