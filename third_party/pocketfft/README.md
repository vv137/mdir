# pocketfft

The C version of pocketfft by Martin Reinecke, from
https://gitlab.mpcdf.mpg.de/mtr/pocketfft (branch `master`, commit
81d171a6d5562e3aaa2c73489b70f564c633ff81, 2019-05-10), unchanged. It is
under the 3-clause BSD license in `LICENSE.md`.

MDIR uses it for the FFT of particle mesh Ewald on the host
(`runtime/mdrt.c`, docs/pme-m1.md).
