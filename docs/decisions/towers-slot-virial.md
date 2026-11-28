# D[towers-slot-virial]

Recorded at its merge.

**Compute the TOWERS direct virial from staged slot forces.** Issue #1048, following [D[towers-virial]](towers-virial.md).

Use the single sum over staged slots in the fused moving pressure traversal.
NT ownership assigns each pair to one interaction box. Distribution of its
displacement outer product puts equal and opposite force words on the two
slots, each with its cell's shift into the box. Repeated staging in tiled
work carries the same image and a disjoint portion of the slot force.
Exclusion blocks subtract the endpoint products with those images. The
pair traversal and the exclusion blocks keep their original force words.

The staged coordinate has 32 fractional bits in nm and the force word has
18 fractional bits in kJ/(mol nm). The two signed 64-bit limbs represent a
Q32 virial with 18 additional fractional bits, which carry through the tile,
warp and grid sums. Rounding each slot separately would break
distributivity. The complete sum rounds once to Q32, ties to even.

The coordinate certificate bounds each shifted component by 256 nm in
magnitude, and therefore each coordinate word by $2^{40}$. The narrow
force word has magnitude at most $2^{31}$. With fewer than $2^{26}$ atoms,
there are fewer than $2^{52}$ pair endpoints,
and exclusion debits can at most double that count. Their combined
absolute sum is below $2^{124}$ in the guarded representation. Wide replay
visits physical pairs and adds only the products missing from the narrow
traversal. For $N$ atoms, each wide endpoint product is bounded by
$2^{126-2\lceil\log_2 N\rceil}$. The two bounds together fit in signed
128 bits. The force certificate rejects an invalid coordinate or wide
product before publication.

A CPU binary64 check on the captured initial frames uses the periodic
direct pair displacements and groups their forces by the NT owner and
staged image. Every pair has one owner. The relative RMS difference
between the pair and slot tensors was $5.09\times 10^{-16}$ on JAC,
$7.02\times 10^{-16}$ on Factor IX, $3.54\times 10^{-16}$ on TIP3P 10k,
and $4.66\times 10^{-16}$ on triclinic Trp-cage. The maximum modeled
absolute staged coordinate was 2.492 nm and the maximum narrow slot
product was 9,246 kJ/mol.

The control keys, defaults, file formats and overwrite behavior are unchanged.
