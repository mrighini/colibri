/* test_v41_amx.c -- v41_amx.h's tile GEMM against a double reference (needs an AMX-BF16 CPU; skips otherwise).
 * Weights are bf16 (exact in both), activations random floats: V41_AMX=1 within bf16 rounding of the activations,
 * V41_AMX=2 near float32. Shapes cover a partial row tile (S % 16), a partial row pair (S % 32), partial output slabs
 * (O % 32, O % 16), the row-group split (few output slabs), and a strided y / x.
 * Build: gcc -O2 -fopenmp -march=x86-64-v4 -I.. test_v41_amx.c -o test_v41_amx -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#include "../v41_amx.h"

#if !V41_AMX_HAVE
int main(void) { printf("SKIP: not x86-64 Linux\n"); return 0; }
#else
typedef struct { const uint16_t *w; } Wb;
static void pack(const void *ctx, int o0, int O, int I, uint16_t *dst) {
    const Wb *w = ctx;
    for (int n = 0; n < 32; n++) {
        const int o = o0 + n;
        for (int i = 0; i < I; i++) dst[v41_amx_bidx(n, i)] = o < O ? w->w[(size_t)o * I + i] : 0;
    }
}
static uint32_t rs = 12345;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) / 16777216.0f) * 2.0f - 1.0f; }

static int one(int mode, int S, int I, int O, int xs, int ys) {
    uint16_t *w = malloc((size_t)O * I * 2);
    float *x = malloc((size_t)S * xs * 4), *y = malloc((size_t)S * ys * 4);
    for (size_t k = 0; k < (size_t)O * I; k++) w[k] = v41_f2bf(rnd() * 0.5f);
    for (size_t k = 0; k < (size_t)S * xs; k++) x[k] = rnd();
    for (size_t k = 0; k < (size_t)S * ys; k++) y[k] = -12345.0f;
    Wb wb = { w };
    g_amx_mode = -1; setenv("V41_AMX", mode == 1 ? "1" : "2", 1);
    if (!v41_amx_on()) { printf("SKIP: no AMX-BF16\n"); exit(0); }
    if (!v41_amx_gemm(y, ys, x, xs, S, I, O, pack, &wb)) { printf("FAIL: gemm refused S %d I %d O %d\n", S, I, O); return 1; }
    double worst = 0; int bad = 0;
    for (int r = 0; r < S; r++) {
        for (int o = 0; o < O; o++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < I; i++) { double p = (double)v41_bf2f(w[(size_t)o * I + i]) * x[(size_t)r * xs + i]; ref += p; mag += fabs(p); }
            const double err = fabs(y[(size_t)r * ys + o] - ref) / (mag > 0 ? mag : 1);
            if (err > worst) worst = err;
        }
        for (int o = O; o < ys; o++) if (y[(size_t)r * ys + o] != -12345.0f) bad++;   /* the stride's gap untouched */
    }
    const double lim = mode == 1 ? 8e-3 : 1e-4;   /* relative to sum |w x|: bf16 activations' rounding vs ~16 bits */
    const int ok = worst < lim && !bad;
    printf("%s mode %d S %4d I %5d O %5d xs %5d ys %5d: worst rel err %.2e (limit %.0e)%s\n", ok ? "ok  " : "FAIL",
           mode, S, I, O, xs, ys, worst, lim, bad ? ", wrote outside O" : "");
    free(w); free(x); free(y);
    return !ok;
}
int main(void) {
    int f = 0;
    const int sh[][5] = { {16, 64, 32, 64, 32}, {17, 96, 40, 96, 48}, {33, 128, 1000, 160, 1000}, {100, 5120, 48, 5120, 64},
                          {512, 1024, 576, 1024, 576}, {31, 5120, 4096, 5120, 4096}, {3000, 256, 77, 256, 80} };
    for (int mode = 1; mode <= 2; mode++)
        for (size_t k = 0; k < sizeof sh / sizeof *sh; k++) f |= one(mode, sh[k][0], sh[k][1], sh[k][2], sh[k][3], sh[k][4]);
    /* throughput: a prompt chunk's trunk-sized product */
    {
        const int S = 2048, I = 5120, O = 4096;
        uint16_t *w = malloc((size_t)O * I * 2); float *x = malloc((size_t)S * I * 4), *y = malloc((size_t)S * O * 4);
        for (size_t k = 0; k < (size_t)O * I; k++) w[k] = v41_f2bf(rnd());
        for (size_t k = 0; k < (size_t)S * I; k++) x[k] = rnd();
        Wb wb = { w };
        for (int mode = 1; mode <= 2; mode++) {
            g_amx_mode = -1; setenv("V41_AMX", mode == 1 ? "1" : "2", 1); v41_amx_on();
            v41_amx_gemm(y, O, x, I, S, I, O, pack, &wb);
            const double t0 = now_s(); v41_amx_gemm(y, O, x, I, S, I, O, pack, &wb); const double t = now_s() - t0;
            printf("perf mode %d: S %d I %d O %d in %.2f ms, %.0f GFLOP/s (%d threads)\n", mode, S, I, O, t * 1e3,
                   2.0 * S * I * (double)O / t / 1e9, omp_get_max_threads());
        }
        free(w); free(x); free(y);
    }
    printf(f ? "FAILED\n" : "PASS\n");
    return f;
}
#endif
