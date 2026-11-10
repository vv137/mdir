/* Test observer for successful refreshes of the template route only. */
#ifndef MDIR_RUNTIME_TEMPLATEREFRESHOBSERVER_H
#define MDIR_RUNTIME_TEMPLATEREFRESHOBSERVER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* active has water bit 0 and atom bit 1. The nine doubles are the actual
 * refresh input cell (ax, by, cz, bx, cx, cy; zero tilts when rectangular)
 * followed by the exact WW, SW, SS reaches. The pointer is valid only for
 * this callback. The callback may inspect the owner but must not mutate it.
 * Register before running the simulation and unregister before data expires;
 * registration and simulation execution must not run concurrently. */
typedef void (*MdrtTemplateRefreshObserver)(int64_t handle, int64_t path,
    int64_t active, const double *cellAndReaches, void *data);

/* Passing a null callback disables observation. No environment key or
 * device operation is introduced; tests load this route-private symbol. */
void mdrtTemplatePairsSetRefreshObserver(MdrtTemplateRefreshObserver callback,
                                        void *data);
#ifdef __cplusplus
}
#endif
#endif
