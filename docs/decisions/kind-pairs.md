# D[kind-pairs]

Recorded at its merge.

**The pair work of the rigid kinds of D403 is divided into segments by one rule, R3: the $K$ kinds that a consumer takes and the rest are kinds $0, \dots, K$, and the segment $s(f, g) = f(f+1)/2 + g$, $g \le f$, holds the pairs of molecules of kinds $f$ and $g$; the `md` ops `md.kind_pairs`, `md.sum_segment_window`, and `md.gather_segment_window` name their segment and take the weights of each site from the kind, and one generator in the driver derives each segment's site pairs from the kinds' per-site data.** Issue #859, the Interaction IR layer of #795 made general before it lands. No run reads these ops yet.

Status: draft; the design is on the pull request.
