# D[towers-slot-virial]

Recorded at its merge.

**Compute the TOWERS direct virial from staged slot forces.** Issue #1048, following [D455](D455.md).

Use the single sum over staged slots in the fused moving pressure traversal.
NT ownership assigns each pair to one interaction box. Distribution of its
displacement outer product puts equal and opposite force words on the two
slots, each with its cell's shift into the box. Repeated staging in tiled
work carries the same image and a disjoint portion of the slot force.
Exclusion blocks subtract the endpoint products with those images. The
pair traversal and the exclusion blocks keep their original force words.

Each staged coordinate uses Q32 in nm. A fused pressure step keeps both
the original Q18 force word and the Q28 virial force in kJ/(mol nm). It
accumulates the Q18 word in the force-only kernel's low and coarse planes,
and the signed difference between Q28 and scaled Q18 in three separate
32-bit shared planes. Each lane holds its target residual in registers
until the target changes. At most 1,024 opposite slots contribute to a
tile slot. Each contributes at most $2^{31}$ base units and $2^9$
residual units, so the residual sum fits in a signed 32-bit word.
When the extra planes exceed the tile's shared-memory budget, the kernel
uses a 64-bit packed word per slot and component. The low 43 bits recover
the signed Q18 sum. Subtracting it from the packed word before extracting
the high field recovers the signed residual. Both layouts publish the
same Q18 force and reconstruct the Q28 word for the outer product.

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
product was 9,246 kJ/mol. Modeling Q28 force quantization on the same
frames with binary64 pair evaluation gave relative RMS errors between
$1.19\times 10^{-11}$ and $1.02\times 10^{-10}$. This isolates fixed-point
rounding from the mixed-precision pair evaluation.

The acceptance tolerance for the full direct tensor is $10^{-6}$ relative
RMS against the independent CPU binary64 pair sum. On RTX 4090, the
maximum relative RMS errors over the sampled frames were
$8.24\times 10^{-7}$ (JAC), $2.41\times 10^{-7}$ (Factor IX),
$3.00\times 10^{-7}$ (TIP3P 10k), $7.29\times 10^{-7}$ (triclinic Trp-cage),
and $7.62\times 10^{-7}$ (0.05 Å exclusion stress). Three of these maxima
exceed the initial $5\times 10^{-7}$ target for a twofold margin.

To select the force scale, compare the error norm introduced by the slot
sum with one tenth of the error norm introduced by the mixed-precision
pair evaluation. Both differences use the same frame. The intermediate
reference accumulates in binary64 the per-pair products of the staged
displacements and the forces evaluated in binary32. The 0.05 Å fixture
retains one Lennard-Jones pair. Factoring its captured rank-one Q60 tensor
bounded the relative RMS slot error on one frame at
$[1.92,1.95]\times 10^{-7}$ for Q18 and at most $1.07\times 10^{-9}$ for
Q28. The mixed-precision pair error on that frame was
$[4.90,5.10]\times 10^{-8}$. This is sufficient to reject Q18 under the
one-tenth criterion. The three-limb Q60 tensor also matched bit for bit
on all 11 stress frames when the excluded sites were disabled.

The control keys, defaults, file formats and overwrite behavior are unchanged.
