/* Device argument shared by the plan builder and the cuFFT load callback. */
typedef struct {
  const float *tables;
  unsigned nx, ny, nz;
} PMECallbackInfo;
