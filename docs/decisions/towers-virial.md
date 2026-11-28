# D[towers-virial]

Recorded at its merge.

**Reduce the cost of the fused TOWERS force and direct virial traversal on
pressure-coupled steps.** Issue #1039.

The fused traversal must retain the certified integer force words. Its
virial contains each nonexcluded pair within the cutoff once, and the
topology terms supply their own contributions. The control file, output
formats, and pressure coupling schedule retain their existing behavior.

## Exclusions and reduction

Each target atom has an undirected exclusion window anchored at the least
of its own ID and its excluded partners' IDs. The first 31 partner IDs in
that window have exact bits. The high bit records whether there are
excluded partners beyond the window. A worker loads the anchor and mask
alongside the target fields it already carries across a record. A source
in the window needs a bit test. Outside the window, a clear high bit
proves that the source is physical. A set high bit uses the existing
forward index and sorted exclusion list. Both paths test membership before
adding the pair's virial. This preserves the exclusion of even very large
or nonfinite provisional interactions.

With $\mathbf d$ the periodic displacement and $\mathbf F$ the pair
force, the direct term is the sum of $\mathbf d \otimes \mathbf F$
over physical pairs inside the cutoff. The reduction keeps the binary64
thread sums, reduces each warp, then reduces the warp sums in one warp.
The block can reuse the tile's shared force planes after their last use.
It publishes one atomic addition per populated tensor component. The
exclusion blocks continue to cancel integer force words and conversion
votes in their original arithmetic.

## Accumulator range

An independent binary64 direct sum on the four initial pressure frames
found a largest physical diagonal term of $751$ kJ/mol and a largest
excluded term of $4.31\times10^6$ kJ/mol. The greatest component sum of
absolute terms was $2.52\times10^7$ kJ/mol over physical pairs and
$2.10\times10^9$ kJ/mol over excluded pairs. An accumulator for
add-then-subtract must cover the large excluded terms and partial sums.
Allowing both signs, a signed 64-bit Q32 pair word covers magnitudes below
$2^{31}$ kJ/mol. Two 64-bit limbs cover the sum of all such words, including
one cancellation per excluded pair. TOWERS admits fewer than $2^{26}$
atoms, so the number of additions and cancellations to each component is
less than $2^{52}$ and even the sum of their absolute word magnitudes is
less than $2^{115}$. Round-to-nearest Q32 gives an absolute quantization
bound of $2^{-33}$ kJ/mol per included pair.

## Cancellation measurements and choice

The subtraction prototypes feed each accepted direct pair into the
accumulator before the exclusion blocks subtract their recomputed terms.
The Q32 prototype rounds each component to nearest, with ties to even,
and compares the rounded word against the signed range before adding it.
An out-of-range or nonfinite conversion contributes zero and casts a
signed vote. A matching invalid exclusion cancels the vote. The remaining
vote must be zero before the prototype publishes the virial. Warp and
block additions propagate the carry between the two limbs. The low limb
of a global atomic addition returns its previous value, which supplies
the carry to the corresponding high-limb addition.

The RTX 4090 timings are mean fused-kernel durations in µs over nine
launches for each input, with 21 launches for triclinic Trp-cage. Q32
subtraction also launches a decode kernel (1.30 µs on JAC).

| System | Cached lookup | FP64 subtraction | Q32 subtraction |
| --- | ---: | ---: | ---: |
| JAC | 67.67 | 54.08 | 82.07 |
| Factor IX | 218.30 | 178.28 | 292.82 |
| TIP3P 10k | 50.86 | 47.65 | 73.36 |
| Triclinic Trp-cage | 135.60 | 98.49 | 172.61 |

The residue columns give the largest absolute component and relative RMS
differences from FP64 lookup over the four initial RTX 5090 pressure frames.
The last column gives the physical diagonal component from the RTX 4090
stress case with a same-tower excluded Lennard-Jones pair at 0.05 Å. The
analytic value is −0.53154494 kcal/mol with an absolute tolerance of
$2\times10^{-5}$ kcal/mol.

| Variant | Component residue (kJ/mol) | Relative RMS residue | 0.05 Å virial (kcal/mol) |
| --- | ---: | ---: | ---: |
| Cached lookup (reference) | 0 | 0 | −0.531545 |
| FP64 subtraction | $2.28\times10^{-6}$ | $1.78\times10^{-12}$ | 0.000000 |
| Q32 subtraction | $1.20\times10^{-7}$ | $7.37\times10^{-13}$ | −0.531545 |

The initial-frame virials meet the $10^{-6}$ relative RMS tolerance against
the independent CPU binary64 direct sum. Q32 subtraction and Q32 lookup
produce identical integer words on all four frames on both GPU models, with
zero outstanding range votes. Both subtraction variants preserve the
provisional and published force words.

The implementation keeps cached lookup and the FP64 block reduction. FP64
subtraction is faster on these four inputs, but it loses the entire physical
contribution in the 0.05 Å case and can propagate a nonfinite provisional
term. Lookup excludes those terms before summation. Q32 preserves the
physical contribution in the stress case, but per-pair quantization and
carry arithmetic make it slower overall even though it reduces the median
JAC block flush from 6,211 to 4,056 SM cycles. The JAC fused kernel remains
above the 1.2-times force-only target.
