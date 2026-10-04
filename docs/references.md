# References

Last updated: 2026-10-03.

This document lists the literature behind the methods, algorithms,
derivations, force fields, and formats that the MDIR documentation and
source code name.

**How to cite.** A document cites an entry by its key in brackets, linked
to the entry: [[Bussi2007]](references.md#bussi2007). A comment in the
source code cites the key or the short form "Bussi et al., J. Chem. Phys.
126, 014101 (2007)". A key is usually the family name of the first author
and the year of publication, written in ASCII; a book by two authors is
keyed by both names, and a manual by the name of the program.

**Verification.** The bibliographic data of every entry was checked on
2026-09-29, or on the day a later entry was added, against a primary
source: the Crossref record of the DOI, the DataCite record for a Zenodo
DOI, or the page of the publisher. An entry
whose data could not be checked is marked **UNVERIFIED**, with the reason.
Authors are listed in full up to six; beyond six, the first three are
followed by "et al."

Entries are sorted by key.

### Abraham2015

M. J. Abraham, T. Murtola, R. Schulz, S. Páll, J. C. Smith, B. Hess,
E. Lindahl, "GROMACS: High performance molecular simulations through
multi-level parallelism from laptops to supercomputers," *SoftwareX*
**1–2**, 19–25 (2015).
[doi:10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001)

Used for: identifying GROMACS, a reference engine for validation and a
system discussed in [prior-art.md](prior-art.md).

### AllenTildesley2017

M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.
(Oxford University Press, Oxford, 2017).
[doi:10.1093/oso/9780198803195.001.0001](https://doi.org/10.1093/oso/9780198803195.001.0001)

Used for: textbook treatment of periodic boundaries and the minimum image
convention, cell (linked) lists, Verlet neighbor lists and the half-skin
displacement criterion for rebuilding them, the long-range correction for
dispersion, and velocities drawn from the Maxwell–Boltzmann distribution.

### Andersen1983

H. C. Andersen, "Rattle: A 'velocity' version of the shake algorithm for
molecular dynamics calculations," *J. Comput. Phys.* **52**, 24–34 (1983).
[doi:10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991(83)90014-1)

Used for: RATTLE, the velocity constraints under velocity Verlet (M1).

### Anderson2020

J. A. Anderson, J. Glaser, S. C. Glotzer, "HOOMD-blue: A Python package
for high-performance molecular dynamics and hard particle Monte Carlo
simulations," *Comput. Mater. Sci.* **173**, 109363 (2020).
[doi:10.1016/j.commatsci.2019.109363](https://doi.org/10.1016/j.commatsci.2019.109363)

Used for: identifying HOOMD-blue, a system discussed in
[prior-art.md](prior-art.md).

### Ballenegger2012

V. Ballenegger, J. J. Cerdà, C. Holm, "How to convert SPME to P3M:
influence functions and error estimates," *J. Chem. Theory Comput.* **8**,
936–947 (2012).
[doi:10.1021/ct2001792](https://doi.org/10.1021/ct2001792)

Used for: influence functions of mesh Ewald other than that of smooth
particle mesh Ewald, of which `influence = "OPTIMAL"` takes a factor
for each edge (pme-m1.md, Section 1.1).

### Barker1973

J. A. Barker, R. O. Watts, "Monte Carlo studies of the dielectric
properties of water-like models," *Mol. Phys.* **26**, 789–792 (1973).
[doi:10.1080/00268977300102101](https://doi.org/10.1080/00268977300102101)

Used for: the reaction field of a dielectric continuum beyond the cutoff
(D140).

### Berendsen1984

H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola,
J. R. Haak, "Molecular dynamics with coupling to an external bath,"
*J. Chem. Phys.* **81**, 3684–3690 (1984).
[doi:10.1063/1.448118](https://doi.org/10.1063/1.448118)

Used for: the Berendsen barostat, which stochastic cell rescaling extends
with a noise term.

### Bernetti2020

M. Bernetti, G. Bussi, "Pressure control using stochastic cell
rescaling," *J. Chem. Phys.* **153**, 114107 (2020).
[doi:10.1063/5.0020514](https://doi.org/10.1063/5.0020514)

Used for: stochastic cell rescaling, the barostat of M1 (D50, D72); its
integrators, the cost of evaluating after a scaling, the effective energy,
and the dependence on the period of coupling (D77).

### Berthelot1898

D. Berthelot, "Sur le mélange des gaz," *C. R. Hebd. Séances Acad. Sci.*
**126**, 1703–1706 (1898). No DOI.

**UNVERIFIED**: the volume (Gallica, ark:/12148/bpt6k3082d) was located,
but the pages could not be opened to check the title and page range.

Used for: the geometric mean of the energy parameter in the
Lorentz–Berthelot mixing rule.

### Beutler1994

T. C. Beutler, A. E. Mark, R. C. van Schaik, P. R. Gerber,
W. F. van Gunsteren, "Avoiding singularities and numerical instabilities
in free energy calculations based on molecular simulations," *Chem. Phys.
Lett.* **222**(6), 529–539 (1994).
[doi:10.1016/0009-2614(94)00397-1](https://doi.org/10.1016/0009-2614(94)00397-1)

Used for: the soft-core Lennard-Jones of the pairs that [free_energy] decouples (D161).

### Blackman2021

D. Blackman, S. Vigna, "Scrambled linear pseudorandom number generators,"
*ACM Trans. Math. Softw.* **47**(4), 1–32 (2021).
[doi:10.1145/3460772](https://doi.org/10.1145/3460772)

Used for: xoshiro256\*\*, the generator of the initial velocities in the
driver.

### Blondel1996

A. Blondel, M. Karplus, "New formulation for derivatives of torsion
angles and improper torsion angles in molecular mechanics: Elimination of
singularities," *J. Comput. Chem.* **17**, 1132–1141 (1996).
[doi:10.1002/(SICI)1096-987X(19960715)17:9&lt;1132::AID-JCC5&gt;3.0.CO;2-T](https://doi.org/10.1002/(SICI)1096-987X(19960715)17:9%3C1132::AID-JCC5%3E3.0.CO;2-T)

Used for: the derivative of a dihedral with respect to the positions of
its four particles, which has no singularity where three particles are
in line.

### Bondi1964

A. Bondi, "van der Waals volumes and radii," *J. Phys. Chem.* **68**,
441–451 (1964).
[doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001)

### BoxMuller1958

G. E. P. Box, M. E. Muller, "A note on the generation of random normal
deviates," *Ann. Math. Stat.* **29**, 610–611 (1958).
[doi:10.1214/aoms/1177706645](https://doi.org/10.1214/aoms/1177706645)

Used for: normal deviates from uniform ones, for the initial velocities
and the thermostat.

### Brunken2025

C. Brunken, O. Peltre, H. Chomet, et al., "Machine learning interatomic
potentials: library for efficient training, model development and
simulation of molecular systems," arXiv:2505.22397 (2025).
[doi:10.48550/arXiv.2505.22397](https://doi.org/10.48550/arXiv.2505.22397)

Used for: related work, a library of learned potentials (MACE, NequIP,
ViSNet) with wrappers for ASE and JAX MD.


### Brooks1983

B. R. Brooks, R. E. Bruccoleri, B. D. Olafson, D. J. States,
S. Swaminathan, M. Karplus, "CHARMM: A program for macromolecular energy,
minimization, and dynamics," *J. Comput. Chem.* **4**, 187–217 (1983).
[doi:10.1002/jcc.540040211](https://doi.org/10.1002/jcc.540040211)

Used for: the squared-distance switching potential (VSWITCH).

### Bussi2007

G. Bussi, D. Donadio, M. Parrinello, "Canonical sampling through velocity
rescaling," *J. Chem. Phys.* **126**, 014101 (2007).
[doi:10.1063/1.2408420](https://doi.org/10.1063/1.2408420)

Used for: stochastic velocity rescaling, the thermostat of M1 (D50).

### Bussi2009

G. Bussi, T. Zykova-Timan, M. Parrinello, "Isothermal-isobaric molecular
dynamics using stochastic velocity rescaling," *J. Chem. Phys.* **130**,
074101 (2009).
[doi:10.1063/1.3073889](https://doi.org/10.1063/1.3073889)

Used for: the second-order barostat that M1 considered and did not take
(D50).

### Chodera2007

J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, K. A. Dill, "Use of the
weighted histogram analysis method for the analysis of simulated and
parallel tempering simulations," *J. Chem. Theory Comput.* **3**(1),
26–41 (2007).
[doi:10.1021/ct0502864](https://doi.org/10.1021/ct0502864)

Used for: the statistical inefficiency by which `scripts/free-energy.py` spaces its samples (D161).

### Darden1993

T. Darden, D. York, L. Pedersen, "Particle mesh Ewald: An N⋅log(N)
method for Ewald sums in large systems," *J. Chem. Phys.* **98**,
10089–10092 (1993).
[doi:10.1063/1.464397](https://doi.org/10.1063/1.464397)

Used for: particle mesh Ewald (M1).

### Daw1984

M. S. Daw, M. I. Baskes, "Embedded-atom method: Derivation and
application to impurities, surfaces, and other defects in metals,"
*Phys. Rev. B* **29**, 6443–6453 (1984).
[doi:10.1103/PhysRevB.29.6443](https://doi.org/10.1103/PhysRevB.29.6443)

Used for: the embedded-atom method (EAM), the example of a many-body
potential with several stages.

### deBuyl2014

P. de Buyl, P. H. Colberg, F. Höfling, "H5MD: A structured, efficient, and
portable file format for molecular data," *Comput. Phys. Commun.* **185**,
1546–1553 (2014).
[doi:10.1016/j.cpc.2014.01.018](https://doi.org/10.1016/j.cpc.2014.01.018)

Used for: H5MD, the format of checkpoints (D26).

### deJong2013

D. H. de Jong, G. Singh, W. F. D. Bennett, et al., "Improved parameters
for the Martini coarse-grained protein force field," *J. Chem. Theory
Comput.* **9**, 687–697 (2013).
[doi:10.1021/ct300646g](https://doi.org/10.1021/ct300646g)

Used for: Martini 2.2, the force field of the first target of M1 (D46),
now deferred (D53).

### Dickson2022

C. J. Dickson, R. C. Walker, I. R. Gould, "Lipid21: Complex lipid membrane
simulations with AMBER," *J. Chem. Theory Comput.* **18**, 1726–1736
(2022).
[doi:10.1021/acs.jctc.1c01217](https://doi.org/10.1021/acs.jctc.1c01217)

Used for: Lipid21, the force field of the POPC bilayer on which the
semi-isotropic barostat is compared with GROMACS (D119).

### Eastman2017

P. Eastman, J. Swails, J. D. Chodera, et al., "OpenMM 7: Rapid
development of high performance algorithms for molecular dynamics,"
*PLoS Comput. Biol.* **13**, e1005659 (2017).
[doi:10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

Used for: identifying OpenMM, whose custom forces define the syntax of
energy expressions (D22) and which is discussed in
[prior-art.md](prior-art.md).

### Ermak1978

D. L. Ermak, J. A. McCammon, "Brownian dynamics with hydrodynamic
interactions," *J. Chem. Phys.* **69**, 1352–1360 (1978).
[doi:10.1063/1.436761](https://doi.org/10.1063/1.436761)

Used for: the step of Brownian dynamics without hydrodynamic
interactions, `integrator = "BROWNIAN"` (D163b).

### Essmann1995

U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee, L. G. Pedersen,
"A smooth particle mesh Ewald method," *J. Chem. Phys.* **103**,
8577–8593 (1995).
[doi:10.1063/1.470117](https://doi.org/10.1063/1.470117)

Used for: smooth particle mesh Ewald, with B-spline charge spreading, the
influence function, and the reciprocal virial (M1); its sum for the
dispersion, 1/r⁶ (Section 5 of the paper), for `lennard_jones = "PME"`
(D162).

### Ewald1921

P. P. Ewald, "Die Berechnung optischer und elektrostatischer
Gitterpotentiale," *Ann. Phys.* **369**, 253–287 (1921).
[doi:10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)

Used for: the Ewald sum, of which particle mesh Ewald is a fast
evaluation, and against which it is validated.

### Fennell2006

C. J. Fennell, J. D. Gezelter, "Is the Ewald summation still necessary?
Pairwise alternatives to the accepted standard for long-range
electrostatics," *J. Chem. Phys.* **124**, 234104 (2006).
[doi:10.1063/1.2206581](https://doi.org/10.1063/1.2206581)

Used for: damped shifted force electrostatics, a cutoff-based
alternative named in the design review.

### Fuchs2025

P. Fuchs, W. Chen, S. Thaler, J. Zavadlav, "chemtrain-deploy: A parallel
and scalable framework for machine learning potentials in million-atom
MD simulations," *J. Chem. Theory Comput.* **21**(15), 7550–7560 (2025).
[doi:10.1021/acs.jctc.5c00996](https://doi.org/10.1021/acs.jctc.5c00996)

Used for: related work, the deployment of potentials defined in JAX
inside LAMMPS on several GPUs.

### Fukunishi2002

H. Fukunishi, O. Watanabe, S. Takada, "On the Hamiltonian replica
exchange method for efficient sampling of biomolecular systems:
Application to protein structure prediction," *J. Chem. Phys.* **116**,
9058–9067 (2002).
[doi:10.1063/1.1472510](https://doi.org/10.1063/1.1472510)

Used for: Hamiltonian replica exchange, a protocol of `ensemble`.

### Galtsov2025

I. S. Galtsov, R. V. Muratov, G. V. Vyskvarko, S. A. Murzov, S. A.
Dyachkov, P. R. Levashov, "MDcraft -- a modern molecular dynamics
simulation package with machine learning potentials support,"
arXiv:2511.22951 (2025).
[doi:10.48550/arXiv.2511.22951](https://doi.org/10.48550/arXiv.2511.22951)

Used for: related work, an engine with a Python interface over a core in
C++, parallel with MPI, that runs learned potentials.

### Gomez2022

Y. K. Gomez, A. M. Natale, J. Lincoff, C. W. Wolgemuth, J. M. Rosenberg,
M. Grabe, "Taking the Monte-Carlo gamble: How not to buckle under the
pressure!," *J. Comput. Chem.* **43**, 431–434 (2022).
[doi:10.1002/jcc.26798](https://doi.org/10.1002/jcc.26798)

Used for: a Monte Carlo barostat accepts a scaling by the change of the
energy, which jumps where a pair crosses a truncated cutoff, and the
pressure of the virial does not see the jump; the volumes of the two
differ unless the correction for the dispersion accounts for it.

### Gratl2022

F. A. Gratl, S. Seckler, H.-J. Bungartz, P. Neumann, "N ways to simulate
short-range particle systems: Automated algorithm selection with the
node-level library AutoPas," *Comput. Phys. Commun.* **273**, 108262
(2022).
[doi:10.1016/j.cpc.2021.108262](https://doi.org/10.1016/j.cpc.2021.108262)

Used for: identifying AutoPas, a model for the joint planner.

### GromacsManual2025

M. Abraham, A. Alekseenko, B. Andrews, et al., *GROMACS 2025.4 Manual*
(Zenodo, 2025).
[doi:10.5281/zenodo.17671776](https://doi.org/10.5281/zenodo.17671776)

Used for: the definitions of the `potential-shift`, `potential-switch`
(the fifth-degree switching polynomial), and `force-switch` modifiers,
which the MDIR truncations `shift`, `switch`, and `force_switch` match
and against which the tests check them; the sign convention of the
GROMACS virial; the formats `.top`, `.itp`, and `.gro`; the update groups of
the domain decomposition of GROMACS (Section "Domain decomposition"),
which the disjoint union of the groups of the constraints parallels (D83).

### Hairer2003

E. Hairer, C. Lubich, G. Wanner, "Geometric numerical integration
illustrated by the Störmer–Verlet method," *Acta Numer.* **12**, 399–450
(2003).
[doi:10.1017/S0962492902000144](https://doi.org/10.1017/S0962492902000144)

Used for: the properties `symplectic` and `time_reversible` of velocity
Verlet and leapfrog, and why neither conserves energy exactly (B9).

### Hawkins1996

G. D. Hawkins, C. J. Cramer, D. G. Truhlar, "Parametrized models of
aqueous free energies of solvation based on pairwise descreening of solute
atomic charges from a dielectric medium," *J. Phys. Chem.* **100**,
19824–19839 (1996).
[doi:10.1021/jp961710n](https://doi.org/10.1021/jp961710n)

Used for: the integral of the descreening of a charge by its neighbors in
generalized Born (D143).

### He2024

Y. He, A. Podobas, S. Markidis, "Leveraging MLIR for loop vectorization
and GPU porting of FFT libraries," in *Euro-Par 2023: Parallel Processing
Workshops*, Lecture Notes in Computer Science (Springer, Cham, 2024),
pp. 207–218.
[doi:10.1007/978-3-031-50684-0_16](https://doi.org/10.1007/978-3-031-50684-0_16)

Used for: related work, FFTc, a language for FFTs on dialects of MLIR;
MDIR calls cuFFT and pocketfft for the transforms of PME.

### Hess2008

B. Hess, C. Kutzner, D. van der Spoel, E. Lindahl, "GROMACS 4:
Algorithms for highly efficient, load-balanced, and scalable molecular
simulation," *J. Chem. Theory Comput.* **4**, 435–447 (2008).
[doi:10.1021/ct700301q](https://doi.org/10.1021/ct700301q)

Used for: eighth-shell domain decomposition with dynamic load balancing
and separate PME ranks, discussed in [prior-art.md](prior-art.md).

### HockneyEastwood1988

R. W. Hockney, J. W. Eastwood, *Computer Simulation Using Particles*
(IOP Publishing, Bristol, 1988).
[doi:10.1887/0852743920](https://doi.org/10.1887/0852743920)

Used for: the leapfrog integrator, and the particle–particle
particle–mesh (PPPM) method.

### Hoogerbrugge1992

P. J. Hoogerbrugge, J. M. V. A. Koelman, "Simulating microscopic
hydrodynamic phenomena with dissipative particle dynamics," *Europhys.
Lett.* **19**, 155–160 (1992).
[doi:10.1209/0295-5075/19/3/001](https://doi.org/10.1209/0295-5075/19/3/001)

Used for: dissipative particle dynamics (DPD), the example of a pairwise
thermostat that needs a neighborhood.

### Hoover1985

W. G. Hoover, "Canonical dynamics: Equilibrium phase-space
distributions," *Phys. Rev. A* **31**, 1695–1697 (1985).
[doi:10.1103/PhysRevA.31.1695](https://doi.org/10.1103/PhysRevA.31.1695)

Used for: the Nosé–Hoover equations, the chain of one thermostat (D163a).

### Hub2014

J. S. Hub, B. L. de Groot, H. Grubmüller, G. Groenhof, "Quantifying
artifacts in Ewald simulations of inhomogeneous systems with a net
charge," *J. Chem. Theory Comput.* **10**, 381–390 (2014).
[doi:10.1021/ct400626b](https://doi.org/10.1021/ct400626b)

Used for: the energy of the uniform background that neutralizes a system
with a net charge under Ewald summation, $-\pi Q^2 / (2V\beta^2)$ in units of
$f$ (pme-m1.md).

### Izadi2014

S. Izadi, R. Anandakrishnan, A. V. Onufriev, "Building water models: A
different approach," *J. Phys. Chem. Lett.* **5**, 3863–3871 (2014).
[doi:10.1021/jz501780a](https://doi.org/10.1021/jz501780a)

Used for: OPC, a water model with four sites, whose extra point is a
virtual site (design-m1.md, Section 19).

### Jones1924

J. E. Jones, "On the determination of molecular fields. —II. From the
equation of state of a gas," *Proc. R. Soc. Lond. A* **106**, 463–477
(1924).
[doi:10.1098/rspa.1924.0082](https://doi.org/10.1098/rspa.1924.0082)

Used for: the Lennard-Jones potential, the pair term of M0.

### Jorgensen1983

W. L. Jorgensen, J. Chandrasekhar, J. D. Madura, R. W. Impey, M. L. Klein,
"Comparison of simple potential functions for simulating liquid water,"
*J. Chem. Phys.* **79**, 926–935 (1983).
[doi:10.1063/1.445869](https://doi.org/10.1063/1.445869)

Used for: TIP3P, the water model of M1 (D53); also TIP4P, which M1 cannot
run.

### Jung2018

J. Jung, C. Kobayashi, Y. Sugita, "Kinetic energy definition in velocity
Verlet integration for accurate pressure evaluation," *J. Chem. Phys.*
**148**, 164109 (2018).
[doi:10.1063/1.5008438](https://doi.org/10.1063/1.5008438)

Used for: the kinetic energy from the half steps, which the pressure
takes (D45).

### Jung2019

J. Jung, C. Kobayashi, Y. Sugita, "Optimal temperature evaluation in
molecular dynamics simulations with a large time step," *J. Chem. Theory
Comput.* **15**, 84–94 (2019).
[doi:10.1021/acs.jctc.8b00874](https://doi.org/10.1021/acs.jctc.8b00874)

Used for: the mean of the kinetic energies at the step and at the half
steps, which the temperature takes (D45).

### Jung2026

J. Jung, D. Ugarte La Torre, C. Kobayashi, K. Ozaki, Y. Sugita,
"Optimized M-SHAKE constraint implementations for GPU-accelerated
molecular dynamics: Balancing precision and performance across
architectures," *J. Comput. Chem.* **47**(25), e70492 (2026).
[doi:10.1002/jcc.70492](https://doi.org/10.1002/jcc.70492)

Used for: the error of the velocities that the rounding of a constraint
in single precision makes, the change of a step divided by the step, as
the source of the drift of the energy, and M-SHAKE for the waters in
mixed precision, whose change is a small number (D112).

### Kelley2025

B. Kelley, S. Rajamanickam, "LAPIS: A performance portable, high
productivity compiler framework," arXiv:2509.25605 (2025).
[doi:10.48550/arXiv.2509.25605](https://doi.org/10.48550/arXiv.2509.25605)

Used for: related work, a compiler on MLIR for sparse and dense linear
algebra that emits Kokkos.

### Kirkwood1935

J. G. Kirkwood, "Statistical mechanics of fluid mixtures," *J. Chem.
Phys.* **3**(5), 300–313 (1935).
[doi:10.1063/1.1749657](https://doi.org/10.1063/1.1749657)

Used for: thermodynamic integration, dF/dλ = ⟨∂H/∂λ⟩ (D161).

### Krautler2001

V. Kräutler, W. F. van Gunsteren, P. H. Hünenberger, "A fast SHAKE
algorithm to solve distance constraint equations for small molecules in
molecular dynamics simulations," *J. Comput. Chem.* **22**(5), 501–508
(2001).
[doi:10.1002/1096-987X(20010415)22:5<501::AID-JCC1021>3.0.CO;2-V](https://doi.org/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V)

Used for: M-SHAKE, the iterations of Newton on all the bonds of a group
at once: the groups of SHAKE, and the rigid waters below double
precision (D112).

### Laio2002

A. Laio, M. Parrinello, "Escaping free-energy minima," *Proc. Natl. Acad.
Sci. USA* **99**, 12562–12566 (2002).
[doi:10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399)

Used for: metadynamics, the example of a history-dependent bias in the
design review.

### Lattner2021

C. Lattner, M. Amini, U. Bondhugula, et al., "MLIR: Scaling compiler
infrastructure for domain specific computation," in *2021 IEEE/ACM
International Symposium on Code Generation and Optimization (CGO)*
(IEEE, 2021), pp. 2–14.
[doi:10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308)

Used for: MLIR, the compiler infrastructure that MDIR is built on.

### LeGrand2013

S. Le Grand, A. W. Götz, R. C. Walker, "SPFP: Speed without
compromise—A mixed precision model for GPU accelerated molecular dynamics
simulations," *Comput. Phys. Commun.* **184**, 374–380 (2013).
[doi:10.1016/j.cpc.2012.09.022](https://doi.org/10.1016/j.cpc.2012.09.022)

Used for: accumulation in fixed point with integer atomic additions,
whose result does not depend on the order of the threads.

### Leimkuhler2013

B. Leimkuhler, C. Matthews, "Rational construction of stochastic numerical
methods for molecular sampling," *Appl. Math. Res. Express* **2013**(1), 34
(2013).
[doi:10.1093/amrx/abs010](https://doi.org/10.1093/amrx/abs010)

Used for: the splitting BAOAB of Langevin dynamics, whose positions sample
the canonical distribution to second order in the step (F1).

### Leimkuhler2016

B. Leimkuhler, C. Matthews, "Efficient molecular dynamics using geodesic
integration and solvent–solute splitting," *Proc. R. Soc. A* **472**,
20160138 (2016).
[doi:10.1098/rspa.2016.0138](https://doi.org/10.1098/rspa.2016.0138)

Used for: BAOAB with holonomic constraints, the projection of the
velocities after each part of the step (F1).

### Lekien2005

F. Lekien, J. Marsden, "Tricubic interpolation in three dimensions,"
*Int. J. Numer. Methods Eng.* **63**, 455–471 (2005).
[doi:10.1002/nme.1296](https://doi.org/10.1002/nme.1296)

Used for: the tricubic patch of a tabulated function of three arguments,
which matches the values, the first derivatives, and the mixed ones at the
corners of its cell (D165).

### Li2014

P. Li, K. M. Merz, "Taking into account the ion-induced dipole
interaction in the nonbonded model of ions," *J. Chem. Theory Comput.*
**10**, 289–297 (2014).
[doi:10.1021/ct400751u](https://doi.org/10.1021/ct400751u)

Used for: the 12-6-4 model of ions, which the readers of M1 reject.

### Lorentz1881

H. A. Lorentz, "Ueber die Anwendung des Satzes vom Virial in der
kinetischen Theorie der Gase," *Ann. Phys.* **248**, 127–136 (1881).
[doi:10.1002/andp.18812480110](https://doi.org/10.1002/andp.18812480110)

Used for: the arithmetic mean of the size parameter in the
Lorentz–Berthelot mixing rule.

### Louwerse2006

M. J. Louwerse, E. J. Baerends, "Calculation of pressure in case of
periodic boundary conditions," *Chem. Phys. Lett.* **421**, 138–141
(2006).
[doi:10.1016/j.cplett.2006.01.087](https://doi.org/10.1016/j.cplett.2006.01.087)

Used for: the virial of pair forces with minimum-image displacements
under periodic boundaries, from which the pressure follows (B8).

### Lyubartsev1992

A. P. Lyubartsev, A. A. Martsinovski, S. V. Shevkunov,
P. N. Vorontsov-Velyaminov, "New approach to Monte Carlo calculation of
the free energy: Method of expanded ensembles," *J. Chem. Phys.* **96**,
1776–1783 (1992).
[doi:10.1063/1.462133](https://doi.org/10.1063/1.462133)

Used for: expanded ensembles, a protocol of `ensemble`.

### MacKerell2004

A. D. MacKerell, M. Feig, C. L. Brooks, "Extending the treatment of
backbone energetics in protein force fields: Limitations of gas-phase
quantum mechanics in reproducing protein conformational distributions in
molecular dynamics simulations," *J. Comput. Chem.* **25**, 1400–1415
(2004).
[doi:10.1002/jcc.20065](https://doi.org/10.1002/jcc.20065)

Used for: CMAP, the correction map of two backbone dihedrals, a bicubic
patch in each cell of a grid (design-m1.md, Section 20).

### LuoRoux2010

Y. Luo, B. Roux, "Simulation of osmotic pressure in concentrated aqueous
salt solutions," *J. Phys. Chem. Lett.* **1**, 183–189 (2010).
[doi:10.1021/jz900079w](https://doi.org/10.1021/jz900079w)

Used for: the osmotic pressure read from the force on walls that act on
the ions only, the example of `observe` (D189).

### Maier2015

J. A. Maier, C. Martinez, K. Kasavajhala, L. Wickstrom, K. E. Hauser,
C. Simmerling, "ff14SB: Improving the accuracy of protein side chain and
backbone parameters from ff99SB," *J. Chem. Theory Comput.* **11**,
3696–3713 (2015).
[doi:10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255)

Used for: ff14SB, the protein force field of M1 (D53).

### Marrink2007

S. J. Marrink, H. J. Risselada, S. Yefimov, D. P. Tieleman, A. H. de Vries,
"The MARTINI force field: Coarse grained model for biomolecular
simulations," *J. Phys. Chem. B* **111**, 7812–7824 (2007).
[doi:10.1021/jp071097f](https://doi.org/10.1021/jp071097f)

Used for: the Martini 2 force field, including its lipids, deferred
(D53).

### Martinez2009

L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez, "PACKMOL: A package
for building initial configurations for molecular dynamics simulations,"
*J. Comput. Chem.* **30**, 2157–2164 (2009).
[doi:10.1002/jcc.21224](https://doi.org/10.1002/jcc.21224)

Used for: packing the POPC bilayer of the comparison of D119, whose close
contacts led to D120.

### Marsaglia2000

G. Marsaglia, W. W. Tsang, "A simple method for generating gamma
variables," *ACM Trans. Math. Softw.* **26**, 363–372 (2000).
[doi:10.1145/358407.358414](https://doi.org/10.1145/358407.358414)

Used for: the sum of the squares of normal deviates that the thermostat
takes, drawn as a gamma deviate (`runtime/mdrt.c`).

### Martyna1992

G. J. Martyna, M. L. Klein, M. Tuckerman, "Nosé–Hoover chains: The
canonical ensemble via continuous dynamics," *J. Chem. Phys.* **97**,
2635–2643 (1992).
[doi:10.1063/1.463940](https://doi.org/10.1063/1.463940)

Used for: the Nosé–Hoover chain, its conserved energy and the masses of
its thermostats, `method = "NOSE-HOOVER"` (D163a).

### Martyna1996

G. J. Martyna, M. E. Tuckerman, D. J. Tobias, M. L. Klein, "Explicit
reversible integrators for extended systems dynamics," *Mol. Phys.*
**87**, 1117–1157 (1996).
[doi:10.1080/00268979600100761](https://doi.org/10.1080/00268979600100761)

Used for: the factorization of the action of a Nosé–Hoover chain by the
Suzuki–Yoshida weights (D163a).

### McGibbon2015

R. T. McGibbon, K. A. Beauchamp, M. P. Harrigan, et al., "MDTraj: A modern
open library for the analysis of molecular dynamics trajectories,"
*Biophys. J.* **109**, 1528–1532 (2015).
[doi:10.1016/j.bpj.2015.08.015](https://doi.org/10.1016/j.bpj.2015.08.015)

Used for: MDTraj, one of the programs of analysis compared in the roadmap
(Section 8).

### Merz2018

P. T. Merz, M. R. Shirts, "Testing for physical validity in molecular
simulations," *PLOS ONE* **13**, e0202764 (2018).
[doi:10.1371/journal.pone.0202764](https://doi.org/10.1371/journal.pone.0202764)

Used for: the tests of physical validity that the roadmap plans for the
mixed precision and the couplings: the convergence of the fluctuation of
the energy with the step, the distribution of the kinetic energy and its
equipartition, and the ensemble tests at two states, which that paper
applies to isotropic coupling only.

### Metropolis1953

N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller,
E. Teller, "Equation of state calculations by fast computing machines,"
*J. Chem. Phys.* **21**, 1087–1092 (1953).
[doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)

Used for: the Metropolis acceptance test (`dyn.metropolis`).

### MichaudAgrawal2011

N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, O. Beckstein, "MDAnalysis: A
toolkit for the analysis of molecular dynamics simulations," *J. Comput.
Chem.* **32**, 2319–2327 (2011).
[doi:10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787)

Used for: MDAnalysis, one of the programs of analysis compared in the
roadmap (Section 8).

### Miyamoto1992

S. Miyamoto, P. A. Kollman, "Settle: An analytical version of the SHAKE
and RATTLE algorithm for rigid water models," *J. Comput. Chem.* **13**,
952–962 (1992).
[doi:10.1002/jcc.540130805](https://doi.org/10.1002/jcc.540130805)

Used for: SETTLE, the constraints of rigid water in double precision
(M1; below it, M-SHAKE, D112).

### Molinero2009

V. Molinero, E. B. Moore, "Water modeled as an intermediate element
between carbon and silicon," *J. Phys. Chem. B* **113**, 4008–4016 (2009).
[doi:10.1021/jp805227c](https://doi.org/10.1021/jp805227c)

Used for: the mW water, the Stillinger-Weber form with the parameters of
water, whose two-body term has a pole at its cutoff (D159) and whose
three-body term is a term over triplets (D160).

### Moses2020

W. S. Moses, V. Churavy, "Instead of rewriting foreign code for machine
learning, automatically synthesize fast gradients," in *Advances in
Neural Information Processing Systems 33 (NeurIPS 2020)* (Curran
Associates, 2020), pp. 12472–12485. No DOI;
[proceedings page](https://proceedings.neurips.cc/paper_files/paper/2020/hash/9332c513ef44b682e9347822c2e457ac-Abstract.html).

Used for: Enzyme, automatic differentiation at the level of LLVM, which
is not the default path of MDIR (D2).

### Mudalige2012

G. R. Mudalige, M. B. Giles, I. Reguly, C. Bertolli, P. H. J. Kelly,
"OP2: An active library framework for solving unstructured mesh-based
applications on multi-core and many-core architectures," in *2012
Innovative Parallel Computing (InPar)* (IEEE, 2012), pp. 1–12.
[doi:10.1109/InPar.2012.6339594](https://doi.org/10.1109/InPar.2012.6339594)

Used for: OP2, the origin of loops with access descriptors.

### Musaelian2023

A. Musaelian, S. Batzner, A. Johansson, et al., "Learning local
equivariant representations for large-scale atomistic dynamics," *Nat.
Commun.* **14**, 579 (2023).
[doi:10.1038/s41467-023-36329-y](https://doi.org/10.1038/s41467-023-36329-y)

Used for: Allegro, the example of a strictly local learned potential.

### Nose1984

S. Nosé, "A molecular dynamics method for simulations in the canonical
ensemble," *Mol. Phys.* **52**, 255–268 (1984).
[doi:10.1080/00268978400101201](https://doi.org/10.1080/00268978400101201)

Used for: the extended system of a thermostat (D163a).

### Onufriev2004

A. Onufriev, D. Bashford, D. A. Case, "Exploring protein native states and
large-scale conformational changes with a modified generalized Born
model," *Proteins* **55**, 383–394 (2004).
[doi:10.1002/prot.20033](https://doi.org/10.1002/prot.20033)

Used for: the Born radii of OBC from the integral of the descreening, and
its parameters α, β, γ (D143).

### Pall2013

S. Páll, B. Hess, "A flexible algorithm for calculating pair interactions
on SIMD architectures," *Comput. Phys. Commun.* **184**, 2641–2650
(2013).
[doi:10.1016/j.cpc.2013.06.003](https://doi.org/10.1016/j.cpc.2013.06.003)

Used for: cluster pair lists (clusters of a fixed number of particles,
lists of cluster pairs searched by bounding boxes, masks of interactions
and exclusions, the 8 × 4 layout of the CUDA kernels), discussed in
[prior-art.md](prior-art.md); the tile structure of MDIR (D82) follows
them, see [tiles-m1.md](tiles-m1.md).

### Pall2020

S. Páll, A. Zhmurov, P. Bauer, et al., "Heterogeneous parallelization
and acceleration of molecular dynamics simulations in GROMACS," *J. Chem.
Phys.* **153**, 134110 (2020).
[doi:10.1063/5.0018516](https://doi.org/10.1063/5.0018516)

Used for: the dual pair list with dynamic (rolling) pruning, the pruning
of cluster pairs on the GPU, and the fixed lifetime of the pair list with
a buffer estimated from a tolerance of the energy drift, which the tile
structure of MDIR (D82) follows or departs from; see
[tiles-m1.md](tiles-m1.md).

### Park2024

Y. Park, J. Kim, S. Hwang, S. Han, "Scalable parallel algorithm for graph
neural network interatomic potentials in molecular dynamics
simulations," *J. Chem. Theory Comput.* **20**, 4857–4868 (2024).
[doi:10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190)

Used for: SevenNet, the example of a message-passing learned potential
with per-layer communication.

### Peng2025

M. J. Peng, W. S. Moses, O. Zinenko, C. Dubach, "Sound and modular
activity analysis for automatic differentiation in MLIR," *Proc. ACM
Program. Lang.* **9**(OOPSLA2), 2087–2114 (2025).
[doi:10.1145/3763125](https://doi.org/10.1145/3763125)

Used for: related work, an activity analysis for automatic
differentiation on MLIR, proved sound as an abstract interpretation.

### Press2007

W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
*Numerical Recipes: The Art of Scientific Computing*, 3rd ed. (Cambridge
University Press, Cambridge, 2007), Section 3.6.

Used for: the bicubic patch of a tabulated function of two arguments from
the values and the derivatives at the corners of its cell (D165).

### Quentrec1973

B. Quentrec, C. Brot, "New method for searching for neighbors in
molecular dynamics computations," *J. Comput. Phys.* **13**, 430–432
(1973).
[doi:10.1016/0021-9991(73)90046-6](https://doi.org/10.1016/0021-9991(73)90046-6)

Used for: binning particles into cells to search for neighbors (cell
lists), the first stage of every neighbor build.

### Rathgeber2012

F. Rathgeber, G. R. Markall, L. Mitchell, et al., "PyOP2: A high-level
framework for performance-portable simulations on unstructured meshes,"
in *2012 SC Companion: High Performance Computing, Networking Storage and
Analysis* (IEEE, 2012), pp. 1116–1123.
[doi:10.1109/SC.Companion.2012.134](https://doi.org/10.1109/SC.Companion.2012.134)

Used for: PyOP2, with OP2 the origin of loops with access descriptors.

### ReactantJl

EnzymeAD, "Reactant.jl," a compiler of Julia functions to MLIR, with
automatic differentiation by EnzymeMLIR and executables for CPUs, GPUs,
and TPUs through XLA, under the MIT license. No paper;
[github.com/EnzymeAD/Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
(read 2026-10-02).

Used for: related work, a compiler on MLIR with differentiation at the
level of MLIR.

### Reinecke2019

M. Reinecke, "pocketfft," a library of fast Fourier transforms in C,
Max-Planck-Society (2010–2019), under the 3-clause BSD license.
[gitlab.mpcdf.mpg.de/mtr/pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft)

Used for: the FFT of particle mesh Ewald on the host (D64, pme-m1.md).

### Roe2013

D. R. Roe, T. E. Cheatham III, "PTRAJ and CPPTRAJ: Software for processing
and analysis of molecular dynamics trajectory data," *J. Chem. Theory
Comput.* **9**, 3084–3095 (2013).
[doi:10.1021/ct400341p](https://doi.org/10.1021/ct400341p)

Used for: cpptraj, the program of analysis of the comparisons of
`scripts/benchmarks/mdbench/structure.py`, and one of those compared in the
roadmap (Section 8).

### Ryckaert1977

J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, "Numerical integration
of the cartesian equations of motion of a system with constraints:
molecular dynamics of n-alkanes," *J. Comput. Phys.* **23**, 327–341
(1977).
[doi:10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991(77)90098-5)

Used for: SHAKE, the constraints of the bonds of hydrogen (M1).

### Salmon2011

J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel random
numbers: As easy as 1, 2, 3," in *Proceedings of 2011 International
Conference for High Performance Computing, Networking, Storage and
Analysis (SC '11)* (ACM, 2011), pp. 1–12.
[doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405)

Used for: counter-based random numbers (Random123) and the Philox
generator, Philox 4×32 with 10 rounds in M1 (`runtime/mdrt.c`), checked
against the known answers of Random123 (`test/Runtime/random.test`).

### SalomonFerrer2013

R. Salomon-Ferrer, A. W. Götz, D. Poole, S. Le Grand, R. C. Walker,
"Routine microsecond molecular dynamics simulations with AMBER on GPUs.
2. Explicit solvent particle mesh Ewald," *J. Chem. Theory Comput.* **9**,
3878–3888 (2013).
[doi:10.1021/ct400314y](https://doi.org/10.1021/ct400314y)

Used for: how pmemd.cuda, the reference of the goal of performance, builds
its neighbor list (Section 3.3): particles sorted into boxes of the
cutoff plus the skin and along a Hilbert curve of 4 × 4 × 4 within each,
then taken in groups of 16 or 32 that share the particles within the
extended cutoff of any of them, from the box and its 13 neighbors on the
leading edge (a half shell); the direct sum iterates over that list a warp
at a time, with the exclusions masked. Measured against in
`scripts/experiments/neighbor-structures` (2026-10-01).

### Saunders2018

W. R. Saunders, J. Grant, E. H. Müller, "A domain specific language for
performance portable molecular dynamics algorithms," *Comput. Phys.
Commun.* **224**, 119–135 (2018).
[doi:10.1016/j.cpc.2017.11.006](https://doi.org/10.1016/j.cpc.2017.11.006)

Used for: PPMD, the origin of the particle and pair loops of `md_exec`
(P11 to P18); see [prior-art.md](prior-art.md).

### Schoenholz2020

S. S. Schoenholz, E. D. Cubuk, "JAX MD: A framework for differentiable
physics," in *Advances in Neural Information Processing Systems 33
(NeurIPS 2020)* (Curran Associates, 2020), pp. 11428–11441. No DOI;
[proceedings page](https://proceedings.neurips.cc/paper_files/paper/2020/hash/83d3d4b6c9579515e1679aca8cbc8033-Abstract.html).

Used for: identifying JAX-MD, a system discussed in
[prior-art.md](prior-art.md).

### SchottVerdugo2019

S. Schott-Verdugo, H. Gohlke, "PACKMOL-Memgen: A simple-to-use,
generalized workflow for membrane-protein–lipid-bilayer system building,"
*J. Chem. Inf. Model.* **59**, 2522–2528 (2019).
[doi:10.1021/acs.jcim.9b00269](https://doi.org/10.1021/acs.jcim.9b00269)

Used for: the input of PACKMOL for the POPC bilayer of D119.

### Shirts2007

M. R. Shirts, D. L. Mobley, J. D. Chodera, V. S. Pande, "Accurate and
efficient corrections for missing dispersion interactions in molecular
simulations," *J. Phys. Chem. B* **111**, 13052–13063 (2007).
[doi:10.1021/jp0735987](https://doi.org/10.1021/jp0735987)

Used for: the correction for the dispersion beyond the cutoff, to the
energy and the pressure (M1).

### Shirts2013

M. R. Shirts, "Simple quantitative tests to validate sampling from
thermodynamic ensembles," *J. Chem. Theory Comput.* **9**(2), 909–926
(2013).
[doi:10.1021/ct300688p](https://doi.org/10.1021/ct300688p)

Used for: the tests of the ensembles of the thermostat and the barostat,
from the ratio of the distributions of two runs at different temperatures
or pressures (`scripts/validation/ensembles`).

### ShirtsChodera2008

M. R. Shirts, J. D. Chodera, "Statistically optimal analysis of samples
from multiple equilibrium states," *J. Chem. Phys.* **129**(12), 124105
(2008).
[doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177)

Used for: MBAR, the estimator of `scripts/free-energy.py`, and the energies of every state that [free_energy] writes for it (D161).

### Slattery2022

S. Slattery, S. T. Reeve, C. Junghans, et al., "Cabana: A performance
portable library for particle-based simulations," *J. Open Source Softw.*
**7**(72), 4115 (2022).
[doi:10.21105/joss.04115](https://doi.org/10.21105/joss.04115)

Used for: identifying Cabana, a model for the runtime.

### Srinivasan1999

J. Srinivasan, M. W. Trevathan, P. Beroza, D. A. Case, "Application of a
pairwise generalized Born model to proteins and nucleic acids: inclusion
of salt effects," *Theor. Chem. Acc.* **101**, 426–434 (1999).
[doi:10.1007/s002140050460](https://doi.org/10.1007/s002140050460)

### Steele2014

G. L. Steele, D. Lea, C. H. Flood, "Fast splittable pseudorandom number
generators," in *Proceedings of the 2014 ACM International Conference on
Object Oriented Programming Systems Languages & Applications (OOPSLA '14)*
(ACM, 2014), pp. 453–472.
[doi:10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195)

Used for: SplitMix, whose 64-bit variant (splitmix64) seeds the state of
the generator of the initial velocities. The mixing constants in the code
are those of the common 64-bit variant, not of the paper.

### Steinbach1994

P. J. Steinbach, B. R. Brooks, "New spherical-cutoff methods for
long-range forces in macromolecular simulation," *J. Comput. Chem.* **15**,
667–683 (1994).
[doi:10.1002/jcc.540150702](https://doi.org/10.1002/jcc.540150702)

Used for: force switching and potential switching as truncations of
nonbonded terms at a cutoff.

### StillingerWeber1985

F. H. Stillinger, T. A. Weber, "Computer simulation of local order in
condensed phases of silicon," *Phys. Rev. B* **31**, 5262–5271 (1985).
[doi:10.1103/PhysRevB.31.5262](https://doi.org/10.1103/PhysRevB.31.5262)

Used for: the three-body term over the triplets centered on each
particle, each center with each unordered pair of its neighbors within
$a\sigma$, and its form, which vanishes at the cutoff with all its
derivatives (D160).

### Stoddard1973

S. D. Stoddard, J. Ford, "Numerical experiments on the stochastic
behavior of a Lennard-Jones gas system," *Phys. Rev. A* **8**, 1504–1512
(1973).
[doi:10.1103/PhysRevA.8.1504](https://doi.org/10.1103/PhysRevA.8.1504)

Used for: the shifted-force potential, `u(r) − u(r_c) − (r − r_c) u'(r_c)`,
the truncation `force_shift`.

### Sugita1999

Y. Sugita, Y. Okamoto, "Replica-exchange molecular dynamics method for
protein folding," *Chem. Phys. Lett.* **314**, 141–151 (1999).
[doi:10.1016/S0009-2614(99)01123-9](https://doi.org/10.1016/S0009-2614(99)01123-9)

Used for: temperature replica exchange, a protocol of `ensemble`.

### Swope1982

W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, "A computer
simulation method for the calculation of equilibrium constants for the
formation of physical clusters of molecules: Application to small water
clusters," *J. Chem. Phys.* **76**, 637–649 (1982).
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716)

Used for: the velocity Verlet integrator (D21).

### Tersoff1988

J. Tersoff, "New empirical approach for the structure and energy of
covalent systems," *Phys. Rev. B* **37**, 6991–7000 (1988).
[doi:10.1103/PhysRevB.37.6991](https://doi.org/10.1103/PhysRevB.37.6991)

Used for: the Tersoff potential, an example of a many-body term.

### Thaler2026

F. Thaler, S. Keller, "GPU-native compressed neighbor lists with a
space-filling-curve data layout," arXiv:2602.19873 (2026).
[doi:10.48550/arXiv.2602.19873](https://doi.org/10.48550/arXiv.2602.19873)

Used for: lists of clusters along a Hilbert curve whose indices are
stored as deltas in nibbles, 3.6 bytes a particle against 12 for GROMACS;
weighed against the lists of groups in the roadmap (Section 2).

### Theobald2005

D. L. Theobald, "Rapid calculation of RMSDs using a quaternion-based
characteristic polynomial," *Acta Crystallogr. A* **61**, 478–480 (2005).
[doi:10.1107/S0108767305015266](https://doi.org/10.1107/S0108767305015266)

Used for: the superposition of RMSD (QCP) that MDTraj takes, and the method
for the matrix of RMSDs between frames on a device (roadmap, Section 8).

### Thompson2009

A. P. Thompson, S. J. Plimpton, W. Mattson, "General formulation of
pressure and stress tensor for arbitrary many-body interaction potentials
under periodic boundary conditions," *J. Chem. Phys.* **131**, 154107
(2009).
[doi:10.1063/1.3245303](https://doi.org/10.1063/1.3245303)

Used for: the virial and the pressure of terms of more than two particles
under periodic boundaries, from the forces and the displacements within
a tuple (M1, Section 4.1 of [design-m1.md](design-m1.md)).

### Thompson2022

A. P. Thompson, H. M. Aktulga, R. Berger, et al., "LAMMPS - a flexible
simulation tool for particle-based materials modeling at the atomic,
meso, and continuum scales," *Comput. Phys. Commun.* **271**, 108171
(2022).
[doi:10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171)

Used for: identifying LAMMPS, a system discussed in
[prior-art.md](prior-art.md).

### Tian2020

C. Tian, K. Kasavajhala, K. A. A. Belfon, L. Raguette, H. Huang,
A. N. Migues, J. Bickel, Y. Wang, J. Pincay, Q. Wu, C. Simmerling, "ff19SB:
Amino-acid-specific protein backbone parameters trained against quantum
mechanics energy surfaces in solution," *J. Chem. Theory Comput.* **16**,
528–552 (2020).
[doi:10.1021/acs.jctc.9b00591](https://doi.org/10.1021/acs.jctc.9b00591)

Used for: ff19SB, the protein force field of the target of M1 (D65), whose
CMAP terms are corrections of $\phi$ and $\psi$ for each amino acid.

### Tiesinga2021

E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, "CODATA recommended
values of the fundamental physical constants: 2018," *Rev. Mod. Phys.*
**93**, 025010 (2021).
[doi:10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)

Used for: the Boltzmann and Avogadro constants, from which the driver
takes the Boltzmann constant in kJ/(mol K) and the factor of the unit of
pressure.

### Tironi1995

I. G. Tironi, R. Sperb, P. E. Smith, W. F. van Gunsteren, "A generalized
reaction field method for molecular dynamics simulations," *J. Chem.
Phys.* **102**, 5451–5459 (1995).
[doi:10.1063/1.469273](https://doi.org/10.1063/1.469273)

Used for: the reaction field with its potential shifted to zero at the
cutoff, `electrostatics = "REACTION_FIELD"` (D140).

### Tuckerman1992

M. Tuckerman, B. J. Berne, G. J. Martyna, "Reversible multiple time
scale molecular dynamics," *J. Chem. Phys.* **97**, 1990–2001 (1992).
[doi:10.1063/1.463137](https://doi.org/10.1063/1.463137)

Used for: multiple time steps (r-RESPA), considered for D163.

### Tuckerman1999

M. E. Tuckerman, C. J. Mundy, G. J. Martyna, "On the classical
statistical mechanics of non-Hamiltonian systems," *Europhys. Lett.*
**45**, 149–155 (1999).
[doi:10.1209/epl/i1999-00139-0](https://doi.org/10.1209/epl/i1999-00139-0)

Used for: the invariant measure of the Nosé–Hoover chain, from the
compressibility of its flow (Section 6.3 of the paper).

### Verlet1967

L. Verlet, "Computer 'experiments' on classical fluids. I.
Thermodynamical properties of Lennard-Jones molecules," *Phys. Rev.*
**159**, 98–103 (1967).
[doi:10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)

Used for: the Verlet integrator, of which velocity Verlet and leapfrog
are forms, and the neighbor list built with a cutoff plus a skin (the
Verlet list).

### VijayKumar1987

S. Vijay-Kumar, C. E. Bugg, W. J. Cook, "Structure of ubiquitin refined at
1.8 Å resolution," *J. Mol. Biol.* **194**, 531–544 (1987).
[doi:10.1016/0022-2836(87)90679-6](https://doi.org/10.1016/0022-2836(87)90679-6)

Used for: PDB entry 1UBQ, the protein of the comparison with GROMACS
(white paper, Section 10.6) and of the tutorial `examples/ubiquitin`.

### Wang2011

L. Wang, R. A. Friesner, B. J. Berne, "Replica exchange with solute
scaling: A more efficient version of replica exchange with solute
tempering (REST2)," *J. Phys. Chem. B* **115**, 9431–9438 (2011).
[doi:10.1021/jp204407d](https://doi.org/10.1021/jp204407d)

Used for: REST2, a protocol of `ensemble`.

### Wennberg2013

C. L. Wennberg, T. Murtola, B. Hess, E. Lindahl, "Lennard-Jones lattice
summation in bilayer simulations has critical effects on surface tension
and lipid properties," *J. Chem. Theory Comput.* **9**, 3527–3537 (2013).
[doi:10.1021/ct400140n](https://doi.org/10.1021/ct400140n)

Used for: particle mesh Ewald for the dispersion with force fields whose
pairs do not follow the geometric rule (D162): the grid sums the geometric
C6, and the direct terms give each pair within the cutoff its own
Lennard-Jones.

### Zhang2019

Z. Zhang, X. Liu, K. Yan, M. E. Tuckerman, J. Liu, "Unified efficient
thermostat scheme for the canonical ensemble with holonomic or isokinetic
constraints via molecular dynamics," *J. Phys. Chem. A* **123**, 6056–6079
(2019).
[doi:10.1021/acs.jpca.9b02771](https://doi.org/10.1021/acs.jpca.9b02771)

Used for: the middle scheme of Langevin dynamics with constraints, the
thermostat between the two halves of the drift (F1).

### Zwanzig1954

R. W. Zwanzig, "High-temperature equation of state by a perturbation
method. I. Nonpolar gases," *J. Chem. Phys.* **22**(8), 1420–1426 (1954).
[doi:10.1063/1.1740409](https://doi.org/10.1063/1.1740409)

Used for: the exponential average of the energy difference between two states (D161).

