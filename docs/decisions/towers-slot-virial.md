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

Each staged coordinate uses Q32 in nm. A fused pressure step keeps both
the original Q18 force word and the Q28 virial force in kJ/(mol nm). It
packs the signed Q18 word into the low 43 bits and the signed difference
between the Q28 word and the scaled Q18 word into the high bits. At most
1,024 opposite slots contribute to a tile slot. Each contributes at most
$2^{31}$ base units and $2^9$ residual units, so both accumulated fields
fit. The fused kernel uses the same shared-memory footprint as the
force-only kernel. At the slot flush, it publishes the original Q18 force
and uses the Q28 word for the outer product.

Three 64-bit limbs hold the signed Q60 virial through the tile, warp, and
grid sums. The complete sum rounds once to Q32, ties to even. For a
shifted coordinate of magnitude below 256 nm, its word has magnitude
below $2^{40}$. With fewer than $2^{26}$ atoms, there are fewer than
$2^{52}$ narrow pair endpoints. Each carries at most $2^{41} + 2^9$ force
units, and exclusion debits at most double the endpoint count. The
combined absolute sum is below $2^{135}$. Wide replay visits physical
pairs and adds the products skipped by the narrow traversal. For $N$
atoms, the original Q50 wide endpoint product is bounded by
$2^{126-2\lceil\log_2 N\rceil}$. The kernel shifts that product by ten in
the accumulator and adds its fine residual. The complete sum fits in
signed 128 bits after dropping the 28 guard bits. The force certificate
rejects an invalid coordinate or wide product before publication.

A CPU binary64 check on the captured initial frames uses the periodic
direct pair displacements and groups their forces by the NT owner and
staged image. Every pair has one owner. The relative RMS difference
between the pair and slot tensors was $5.09\times 10^{-16}$ on JAC,
$7.02\times 10^{-16}$ on Factor IX, $3.54\times 10^{-16}$ on TIP3P 10k,
and $4.66\times 10^{-16}$ on triclinic Trp-cage. The maximum modeled
absolute staged coordinate was 2.492 nm and the maximum narrow slot
product was 9,246 kJ/mol.

The control keys, defaults, file formats and overwrite behavior are unchanged.
