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
takes one reciprocal correction by topology identity. A particle owner
spreads onto a sparse subgrid covering the union of its assignment stencils.
The extra support is the window order minus one mesh intervals. Owners
exchange contributions by periodic mesh index. Each stencil contribution
is rounded to the nearest even signed Q40 word in units of elementary
charge. Both the individual word and the completed point must fit in
$[-2^{23},2^{23})$ elementary charges. Signed 128-bit intermediate sums
preserve cancellation.

Forward transforms run along $z$, then $y$, then $x$. Two all-to-all
transposes deliver complete pencils to the next axis. The influence,
energy and virial use the resulting $x$ pencils, including the Hermitian
multiplicity. The double inverse reverses these stages. Mixed precision
transforms $y$, then $x$, then $z$ on the inverse, redistributing the
spectrum first so each axis rounds in ordinary CPU order. Returning the
potential along the reversed subgrid exchange supplies every stencil of
an owned particle. Mixed assignment computes the fractional coordinate from
the input image, avoiding a second periodic subtraction at a binary32
rounding boundary. Only pencil data and overlap points move between mesh owners.

In deterministic mode each Fourier mode contributes a Q32 energy and
virial word per observable component. Mesh ranks sum in signed 128-bit
integers and check the completed word. Each gathered force component is
rounded to Q40. Rank zero computes the self and neutralizing-background
constants in canonical charge order and broadcasts them during setup.
