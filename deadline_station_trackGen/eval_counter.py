"""Machine-independent compute counter (model evaluations).

One unit = one one-step prediction of the train's motion (neural emulator, point-mass, MPC horizon step, or
one QP-planner iteration per horizon node), one whole-trip arrival-time prediction, one safety check of a
candidate command, or one query of the pilot (network, table, or rule). The benchmark resets N before each
journey and divides by the number of control steps."""
N = [0]          # running count
QP_NODES = [0]   # horizon nodes of the QP currently being solved (set by the planner before solving)
