# D[distribute-pme]

Recorded at its merge.

Electrostatic PME on a fixed orthorhombic CPU domain grid uses a second,
independent two-dimensional grid of mesh ranks. The first `pme_ranks`
transport ranks own the mesh. The default is the full communicator.
An explicit `pme_decomposition = [a, b]` sets both factors and implies
their product. When both keys are present their counts must agree.
Setup chooses feasible factors that minimize the total real-pencil
boundary length and logs the mesh dimensions, factors and rank count.
Local transport has one mesh rank.

The direct term follows the pair traversal. Each excluded unordered pair
takes one reciprocal correction by topology identity. Each particle domain stores a contiguous subgrid. Its extent along
each axis spans all charged stencil starts, including constraint-group
members, and the following window-order-minus-one grid intervals.
The subgrid can cross the periodic face. The ordinary CPU spread loop
runs over the domain's charged particles and adds to this subgrid in
floating point. Mixed precision rounds each stencil product to binary32
and accumulates in binary64 before rounding the completed mesh point.
Gather reuses the cached weights and their derivatives.

In deterministic mode each individual assignment is rounded to the
nearest even signed Q40 word. Each output plane is assigned to a single
worker, which visits the particles whose stencils reach it. Signed
128-bit intermediate sums preserve cancellation. Each individual word
and the completed point must fit in $[-2^{23},2^{23})$ elementary charges.
Domains exchange their subgrid bounds once per mesh evaluation. Both
directions pack contiguous rows of each subgrid–pencil intersection.
Rows owned by the same rank are copied or accumulated locally.

Forward transforms run along $z$, then $y$, then $x$. Two all-to-all
transposes deliver complete pencils to the next axis. The influence,
energy and virial use the resulting $x$ pencils, including the Hermitian
multiplicity. The double inverse reverses these stages. Mixed precision
transforms $y$, then $x$, then $z$ on the inverse, redistributing the
spectrum first so each axis rounds in ordinary CPU order. Returning the
potential along the reversed subgrid exchange supplies every stencil of
an owned particle. Mixed assignment computes the fractional coordinate from
the input image, avoiding a second periodic subtraction at a binary32
rounding boundary. The distribution log counts logical bytes sent to other ranks, including
other ranks on the same node. The overlap counter includes the subgrid
bounds and the charge and potential rows. The transpose counter includes
the complex pencil intersections sent during forward and inverse transforms.
Local copies contribute zero bytes. An interval reports totals across
ranks and steps.

In deterministic mode each Fourier mode contributes a Q32 energy and
virial word per observable component. Mesh ranks sum in signed 128-bit
integers and check the completed word. Each gathered force component is
rounded to Q40. Rank zero computes the self and neutralizing-background
constants in canonical charge order and broadcasts them during setup.
