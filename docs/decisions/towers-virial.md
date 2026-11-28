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

On the four initial pressure frames on the RTX 5090, the greatest
FP64 subtraction residue relative to exact lookup was
$2.28\times10^{-6}$ kJ/mol, or $1.79\times10^{-12}$ relative RMS on
Factor IX. The four frames meet the $10^{-6}$ relative RMS bound against
the independent binary64 direct sum. The Q32 sums obtained by subtracting
exclusions agree word for word with the Q32 sums obtained by looking up
exclusions before adding. The force words also agree through publication.

On RTX 4090 JAC, the mean fused kernel over nine launches was
$67.67$ µs with lookup, $54.08$ µs with FP64 subtraction, and $82.07$ µs
with Q32 subtraction. The Q32 decode took another $1.30$ µs. Median
instrumented flush costs were $6{,}211$ cycles for FP64 subtraction and
$4{,}056$ for Q32 subtraction. The cheaper integer flush did not offset
per-pair rounding and carry arithmetic. Q32 would make this direct sum
independent of accumulation order, but its total cost exceeded both
FP64 variants on this frame. The implementation keeps the exact cached
lookup and the FP64 block reduction. Lookup also keeps arbitrarily
large or nonfinite excluded terms out of the accumulation, which FP64
add-then-subtract cannot guarantee. The JAC fused kernel remains above
the 1.2-times force-only target.
