/* v41_amx.h -- V41_AMX: deepseek_v41's prompt-pass matrix products (rows >= V41_AMX_MIN, default 16) on the CPU's
 * AMX-BF16 tiles (Sapphire Rapids and later). Off by default. NOT the AVX2 path's bits: the products are summed in the
 * tile unit's order, and the activations go in as bf16 -- so a run with it is judged by a quality gate, not by
 * identity.
 *   V41_AMX=1  activations rounded to bf16 (round to nearest even): ~3 significant digits an activation.
 *   V41_AMX=2  activations split x = hi + lo, both bf16, two tile products: ~16 bits of mantissa, near float32.
 * The weights are exact in bf16 with their scales folded in: an e4m3 value times its ue8m0 tile scale, an e2m1 value
 * times its group's 2^(e-127), a bf16 weight as it is (powers of two only move the exponent, which bf16 shares with
 * float32). Decode-sized calls (rows below the threshold) and calls from inside an OpenMP region keep the AVX2 path.
 * Layout: C tiles 0..3 (2 row tiles x 2 output slabs of 16, float32), A tiles 4,5 (16 rows x 32 bf16 of the
 * activations), B tiles 6,7 (16 pairs of K x 16 outputs x 2 bf16: the VNNI pairing TDPBF16PS reads). */
#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <cpuid.h>
#include <sys/syscall.h>
#include <unistd.h>
#define V41_AMX_HAVE 1

typedef void (*v41_amx_pack_fn)(const void *ctx, int o0, int O, int I, uint16_t *dst);
typedef struct { uint8_t palette, start_row; uint8_t rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } V41TileCfg;

static int g_amx_mode = -1, g_amx_min = 16;
static unsigned long long g_amx_calls, g_amx_rows;
static double g_amx_s, g_amx_flop;

static void v41_amx_report(void) {
    if (g_amx_mode > 0)
        fprintf(stderr, "[v41] V41_AMX=%d: %llu products, %llu rows, %.2f s, %.1f GFLOP/s\n", g_amx_mode, g_amx_calls,
                g_amx_rows, g_amx_s, g_amx_s > 0 ? g_amx_flop / g_amx_s / 1e9 : 0.0);
}
static int v41_amx_on(void) {
    if (g_amx_mode >= 0) return g_amx_mode;
    const char *e = getenv("V41_AMX");
    int m = e ? atoi(e) : 0;
    const char *mn = getenv("V41_AMX_MIN");
    if (mn && atoi(mn) > 0) g_amx_min = atoi(mn);
    if (m > 0) {
        unsigned a = 0, b = 0, c = 0, d = 0;
        /* CPUID.7.0: EDX bit 24 AMX-TILE, bit 22 AMX-BF16; EBX bit 16 AVX512F */
        int ok = __get_cpuid_count(7, 0, &a, &b, &c, &d) && (d & (1u << 24)) && (d & (1u << 22)) && (b & (1u << 16));
        /* the tile data state needs the kernel's permission, process-wide (ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) */
        if (ok) ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0;
        if (!ok) { fprintf(stderr, "[v41] V41_AMX=%d: no AMX-BF16 here, or the kernel refused the tile state: off\n", m); m = 0; }
        else {
            if (m > 2) m = 2;
            fprintf(stderr, "[v41] V41_AMX=%d: prompt-pass products on AMX-BF16 tiles (rows >= %d; activations %s)\n", m,
                    g_amx_min, m == 2 ? "split hi+lo bf16" : "bf16");
            static int registered; if (!registered) { registered = 1; atexit(v41_amx_report); }
        }
    }
    g_amx_mode = m;
    return m;
}
static inline uint16_t v41_f2bf(float f) {   /* round to nearest even; a NaN stays a (quiet) NaN */
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline uint16_t v41_f2bf_exact(float f) {   /* a value bf16 holds exactly: the top half */
    uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16);
}
static inline float v41_bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

__attribute__((target("amx-tile"), noinline)) static void v41_amx_cfg(void) {
    V41TileCfg cfg; memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    for (int t = 0; t < 8; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    _tile_loadconfig(&cfg);
}
__attribute__((target("amx-tile"), noinline)) static void v41_amx_release(void) { _tile_release(); }

/* rows [rb, re) (multiples of 16, re <= the padded row count) of one 32-output slab bp, all of K */
__attribute__((target("amx-tile,amx-bf16"), noinline))
static void v41_amx_task(float *y, int ystride, const uint16_t *xh, const uint16_t *xl, int S, int I, int O,
                         const uint16_t *bp, int o0, int rb, int re) {
    float c[4][256] __attribute__((aligned(64)));
    const long as = (long)I * 2;
    const int kbs = I / 32;
    for (int r0 = rb; r0 < re; r0 += 32) {
        const int two = r0 + 16 < re;
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        const uint16_t *ah = xh + (size_t)r0 * I, *al = xl ? xl + (size_t)r0 * I : NULL;
        for (int kb = 0; kb < kbs; kb++) {
            const uint16_t *b = bp + (size_t)kb * 1024;
            _tile_loadd(6, b, 64);
            _tile_loadd(7, b + 512, 64);
            _tile_loadd(4, ah + kb * 32, as);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            if (two) {
                _tile_loadd(5, ah + (size_t)16 * I + kb * 32, as);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            if (al) {
                _tile_loadd(4, al + kb * 32, as);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                if (two) {
                    _tile_loadd(5, al + (size_t)16 * I + kb * 32, as);
                    _tile_dpbf16ps(2, 5, 6);
                    _tile_dpbf16ps(3, 5, 7);
                }
            }
        }
        _tile_stored(0, c[0], 64);
        _tile_stored(1, c[1], 64);
        if (two) { _tile_stored(2, c[2], 64); _tile_stored(3, c[3], 64); }
        for (int rt = 0; rt <= two; rt++)
            for (int rr = 0; rr < 16; rr++) {
                const int r = r0 + rt * 16 + rr;
                if (r >= S) break;
                float *yr = y + (size_t)r * ystride;
                for (int half = 0; half < 2; half++) {
                    const float *cs = c[rt * 2 + half] + rr * 16;
                    const int ob = o0 + half * 16, n = O - ob < 16 ? O - ob : 16;
                    for (int k = 0; k < n; k++) yr[ob + k] = cs[k];
                }
            }
    }
}

/* y[r][o] = sum_i W(o,i) x[r][i] for S rows; pack(ctx, o0, O, I, dst) writes the 32-output slab at o0 (zeros past O)
 * in the B layout. Returns 0 (nothing written) where the call stays the AVX2 path's. */
static int v41_amx_gemm(float *y, int ystride, const float *x, int xstride, int S, int I, int O,
                        v41_amx_pack_fn pack, const void *ctx) {
    const int mode = v41_amx_on();
    if (!mode || S < g_amx_min || I < 32 || (I & 31) || O < 1 || omp_in_parallel()) return 0;
    const double t0 = now_s();
    const int Sp = (S + 15) & ~15;
    uint16_t *xh = aligned_alloc(64, ((size_t)Sp * I * 2 + 63) & ~(size_t)63);
    uint16_t *xl = mode == 2 ? aligned_alloc(64, ((size_t)Sp * I * 2 + 63) & ~(size_t)63) : NULL;
    if (!xh || (mode == 2 && !xl)) { free(xh); free(xl); return 0; }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < Sp; r++) {
        uint16_t *h = xh + (size_t)r * I, *l = xl ? xl + (size_t)r * I : NULL;
        if (r >= S) { memset(h, 0, (size_t)I * 2); if (l) memset(l, 0, (size_t)I * 2); continue; }
        const float *xr = x + (size_t)r * xstride;
        for (int i = 0; i < I; i++) {
            const uint16_t b = v41_f2bf(xr[i]);
            h[i] = b;
            if (l) { const float hv = v41_bf2f(b); l[i] = isfinite(hv) ? v41_f2bf(xr[i] - hv) : 0; }
        }
    }
    const int nob = (O + 31) / 32, nth = omp_get_max_threads();
    int nrg = 1;   /* row groups (multiples of 32 rows) when the output slabs alone would leave threads idle */
    while (nob * nrg < 2 * nth && (nrg + 1) * 32 <= Sp) nrg++;
    const int rows_per = ((Sp + nrg - 1) / nrg + 31) & ~31;
    const int ntask = nob * nrg;
    int failed = 0;
    #pragma omp parallel
    {
        uint16_t *bp = aligned_alloc(64, (size_t)I * 64);
        if (!bp) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp barrier
        if (!failed) {
            v41_amx_cfg();
            int last = -1;
            #pragma omp for schedule(static)
            for (int t = 0; t < ntask; t++) {
                const int ob = t / nrg, rg = t % nrg;
                const int rb = rg * rows_per;
                int re = rb + rows_per; if (re > Sp) re = Sp;
                if (rb >= re) continue;
                if (ob != last) { pack(ctx, ob * 32, O, I, bp); last = ob; }
                v41_amx_task(y, ystride, xh, xl, S, I, O, bp, ob * 32, rb, re);
            }
            v41_amx_release();
        }
        free(bp);
    }
    free(xh); free(xl);
    if (failed) return 0;
    g_amx_calls++; g_amx_rows += (unsigned long long)S;
    g_amx_flop += 2.0 * S * (double)I * O * (mode == 2 ? 2 : 1);
    g_amx_s += now_s() - t0;
    return 1;
}
/* the B slab's index of weight (output n of the slab, column i) */
static inline size_t v41_amx_bidx(int n, int i) {
    return (size_t)(i >> 5) * 1024 + (size_t)(n >> 4) * 512 + (size_t)((i & 31) >> 1) * 32 + (size_t)(n & 15) * 2 + (i & 1);
}
#else
#define V41_AMX_HAVE 0
#endif
