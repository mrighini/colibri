/* bench_mv8_512.c -- v41_mv8_512.h against deepseek_v41.c's AVX2 mv8_rows_cpu loop (copied below verbatim in its
 * arithmetic): byte-for-byte identity over trunk-shaped FP8 matrices (rows 1, 2, 4, 7; NaN weights included), then the
 * time of a "token" -- every matrix once, rows 1 and 4, an OpenMP parallel loop over outputs a call as the engine does
 * -- for the AVX2 loop, the AVX-512 pairs, and a plain read of the same bytes (the memory bound).
 * Build as the engine: gcc -O3 -march=x86-64-v4 -fopenmp bench_mv8_512.c -o bench_mv8_512 -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include "../v41_mv8_512.h"
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- the engine's pieces (quant.h e4m3_decode8, idot.h hsum256, deepseek_v41.c ue8m0 / mv_block_rows / mv8_rows_cpu) */
static inline __m256 e4m3_decode8(const uint8_t *p) {
    __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i low = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));
    __m256i bits = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24), _mm256_slli_epi32(low, 20));
    __m256 value = _mm256_mul_ps(_mm256_castsi256_ps(bits), _mm256_set1_ps(0x1p120f));
    __m256i nan = _mm256_cmpeq_epi32(low, _mm256_set1_epi32(0x7F));
    return _mm256_blendv_ps(value, _mm256_set1_ps(NAN), _mm256_castsi256_ps(nan));
}
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
static inline float ue8m0(uint8_t byte) { union { uint32_t u; float f; } v; v.u = (uint32_t)byte << 23; return v.f; }
#define FP8_TILE 32
#define MV_ROWS_MAX 32
static int mv_block_rows(int I) { int block = (int)((192 * 1024) / ((size_t)I * sizeof(float) + 1)); if (block > MV_ROWS_MAX) block = MV_ROWS_MAX; return block < 1 ? 1 : block; }
typedef struct { uint8_t *q, *s; int O, I; } W8;
static void mv8_avx2(float *y, int ystride, const W8 *w, const float *x, int xstride, int rows) {
    int I = w->I, tiles_i = (I + FP8_TILE - 1) / FP8_TILE;
    int block = mv_block_rows(I);
    for (int r0 = 0; r0 < rows; r0 += block) {
        int nr = rows - r0 < block ? rows - r0 : block;
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < w->O; o++) {
            const uint8_t *row = w->q + (size_t)o * I;
            const uint8_t *scale = w->s + (size_t)(o / FP8_TILE) * tiles_i;
            float sum[MV_ROWS_MAX];
            for (int r = 0; r < nr; r++) sum[r] = 0.0f;
            for (int base = 0; base < I; base += FP8_TILE) {
                float tile = ue8m0(scale[base / FP8_TILE]);
                __m256 w0 = e4m3_decode8(row + base), w1 = e4m3_decode8(row + base + 8);
                __m256 w2 = e4m3_decode8(row + base + 16), w3 = e4m3_decode8(row + base + 24);
                for (int r = 0; r < nr; r++) {
                    const float *xr = x + (size_t)(r0 + r) * xstride + base;
                    __m256 acc = _mm256_mul_ps(w0, _mm256_loadu_ps(xr));
                    acc = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr + 8), acc);
                    acc = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr + 16), acc);
                    acc = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr + 24), acc);
                    sum[r] += hsum256(acc) * tile;
                }
            }
            for (int r = 0; r < nr; r++) y[(size_t)(r0 + r) * ystride + o] = sum[r];
        }
    }
}
/* the candidate: pairs on AVX-512, an odd last output on the AVX2 loop's code */
static void mv8_512(float *y, int ystride, const W8 *w, const float *x, int xstride, int rows) {
    int I = w->I, block = mv_block_rows(I), pairs = w->O / 2;
    for (int r0 = 0; r0 < rows; r0 += block) {
        int nr = rows - r0 < block ? rows - r0 : block;
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < pairs; p++) v41_mv8_pair(y, ystride, w->q, w->s, I, 2 * p, x, xstride, r0, nr);
    }
    if (w->O & 1) {   /* the last output alone, the AVX2 loop's arithmetic */
        W8 last = *w; last.q = w->q + (size_t)(w->O - 1) * I; last.O = 1;
        const int tiles_i = I / 32;
        last.s = w->s + (size_t)((w->O - 1) / 32) * tiles_i;
        float *tmp = malloc((size_t)rows * sizeof(float));
        mv8_avx2(tmp, 1, &last, x, xstride, rows);
        for (int r = 0; r < rows; r++) y[(size_t)r * ystride + w->O - 1] = tmp[r];
        free(tmp);
    }
}
/* the memory bound: every weight byte read once, the same parallel loop */
__attribute__((target("avx512f,avx512bw")))
static float readall(const W8 *w) {
    double t = 0;
    #pragma omp parallel for schedule(static) reduction(+:t)
    for (int o = 0; o < w->O; o++) {
        const uint8_t *row = w->q + (size_t)o * w->I;
        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i + 64 <= w->I; i += 64) acc = _mm512_add_epi8(acc, _mm512_loadu_si512(row + i));
        t += _mm512_reduce_add_epi32(acc);
    }
    return (float)t;
}

static uint32_t rs = 777;
static uint32_t rnu(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static void wmake(W8 *w, int O, int I, int nans) {
    w->O = O; w->I = I;
    w->q = aligned_alloc(64, ((size_t)O * I + 63) & ~(size_t)63);
    w->s = malloc((size_t)((O + 31) / 32) * (I / 32));
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        uint32_t st = 0x9e3779b9u * (uint32_t)(o + 1);
        for (int i = 0; i < I; i++) { st = st * 1664525u + 1013904223u; uint8_t b = (uint8_t)(st >> 24); if (!nans && (b & 0x7f) == 0x7f) b ^= 1; if (nans == 0 && !(b & 0x78)) b |= 0x08; w->q[(size_t)o * I + i] = b; }
    }
    for (size_t k = 0; k < (size_t)((O + 31) / 32) * (I / 32); k++) w->s[k] = (uint8_t)(112 + rnu() % 12);
}

int main(void) {
    /* identity: shapes with odd O, a single output, NaNs, several row counts */
    int bad = 0; long long vals = 0;
    const int shp[][2] = { {1024, 5120}, {577, 5120}, {33, 1024}, {1, 4096}, {5120, 8192}, {64, 32} };
    for (size_t k = 0; k < sizeof shp / sizeof *shp; k++) {
        W8 w; wmake(&w, shp[k][0], shp[k][1], k == 2 ? 1 : 2);   /* raw bytes: subnormals (and for k 2 NaNs) included */
        for (int rows = 1; rows <= 7; rows += (rows < 2 ? 1 : rows == 2 ? 2 : 3)) {
            const int xs = w.I + 32;
            float *x = malloc((size_t)rows * xs * 4), *a = malloc((size_t)rows * w.O * 4), *b = malloc((size_t)rows * w.O * 4);
            for (size_t i = 0; i < (size_t)rows * xs; i++) x[i] = ((int)(rnu() % 2001) - 1000) / 337.0f;
            mv8_avx2(a, w.O, &w, x, xs, rows); mv8_512(b, w.O, &w, x, xs, rows);
            for (size_t i = 0; i < (size_t)rows * w.O; i++) { if (memcmp(a + i, b + i, 4)) { if (bad++ < 5) printf("DIFF O %d I %d rows %d at %zu: %a vs %a\n", w.O, w.I, rows, i, a[i], b[i]); } vals++; }
            free(x); free(a); free(b);
        }
        free(w.q); free(w.s);
    }
    printf("identity: %lld values compared, %d differ\n", vals, bad);

    /* a token: a decoder layer's trunk shapes x 40; normal e4m3 only (trained weights almost never hold the subnormals
     * whose float decode takes a microcode assist on every vector that has one) */
    const int lay[][2] = { {1024, 5120}, {32768, 1024}, {576, 5120}, {8192, 4096}, {5120, 8192}, {2048, 5120}, {2048, 5120}, {5120, 2048} };
    const int nl = 40, nm = (int)(sizeof lay / sizeof *lay);
    W8 *ws = malloc(sizeof(W8) * nl * nm); double gb = 0;
    for (int l = 0; l < nl; l++) for (int m = 0; m < nm; m++) { wmake(&ws[l * nm + m], lay[m][0], lay[m][1], 0); gb += (double)lay[m][0] * lay[m][1] / 1e9; }
    float *x = malloc(4 * 32768 * 4), *y = malloc(4 * 32768 * 4);
    for (int i = 0; i < 4 * 32768; i++) x[i] = ((int)(rnu() % 2001) - 1000) / 337.0f;
    printf("token: %d matrices, %.2f GB of FP8, %d threads\n", nl * nm, gb, omp_get_max_threads());
    for (int rows = 1; rows <= 4; rows += 3)
        for (int v = 0; v < 3; v++) {
            double best = 1e9;
            for (int rep = 0; rep < 4; rep++) {
                const double t0 = now_s(); volatile float sink = 0;
                for (int k = 0; k < nl * nm; k++) {
                    const W8 *w = &ws[k];
                    if (v == 0) mv8_avx2(y, w->O, w, x, w->I, rows);
                    else if (v == 1) mv8_512(y, w->O, w, x, w->I, rows);
                    else sink += readall(w);
                }
                const double t = now_s() - t0; if (t < best) best = t;
                (void)sink;
            }
            static const char *vn[] = {"AVX2 (engine)", "AVX-512 pairs", "plain read"};
            printf("rows %d %-14s %7.2f ms a token, %6.1f GB/s\n", rows, vn[v], best * 1e3, gb / best);
        }
    printf(bad ? "FAILED\n" : "PASS\n");
    return bad != 0;
}
