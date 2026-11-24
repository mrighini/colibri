/* coli_cuda_bf16t_* (deepseek_v41's V41_HEAD_GPU / V41_BF16_GPU): every output must be, byte for byte, deepseek_v41.c's
 * mvb_cpu on an AVX2 build: per output and row, lanes j = 0..7 take columns j, j+8, ... in order, acc = fma(w, x, acc)
 * from +0; y = ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). The reference replays that with std::fma, each operation a
 * statement. Both kernels are held to it: bf16t2_kernel (the default) and bf16t_kernel (COLI_SYCL_BF16T_V1=1). Shapes:
 * DeepSeek V4.1's lm head (129280 x 5120), its router and compressor shapes, and odd ones (O not a multiple of 256, I a
 * multiple of 8 only), 1..9 rows; then the head timed at 1 and 4 rows for both kernels (BF16T-TIME, GB/s of weights).
 * "BF16T SUITE PASS" at the end. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails;
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL BF16T: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x2545f491u;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float bf(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void reference(float *y, const uint16_t *w, const float *x, int S, int I, int O) {
    for (int r = 0; r < S; r++)
        for (int o = 0; o < O; o++) {
            const uint16_t *row = w + (size_t)o * I;
            const float *xr = x + (size_t)r * I;
            float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < I; i += 8)
                for (int j = 0; j < 8; j++) a[j] = std::fma(bf(row[i + j]), xr[i + j], a[j]);
            const float s04 = a[0] + a[4], s26 = a[2] + a[6], s15 = a[1] + a[5], s37 = a[3] + a[7];
            const float lo = s04 + s26, hi = s15 + s37;
            y[(size_t)r * O + o] = lo + hi;
        }
}
static uint16_t wrand() {   /* a finite bf16, subnormals included, no Inf/NaN */
    uint16_t b;
    do b = (uint16_t)rnd(); while ((b & 0x7f80) == 0x7f80);
    if ((rnd() & 7) == 0) b &= 0x807f;   /* a subnormal now and then */
    else b = (uint16_t)((b & 0x807f) | ((0x70 + (rnd() % 24)) << 7));   /* exponents near 1 */
    return b;
}
static void shape(int dev, int O, int I, int S, int v1) {
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    std::vector<float> x((size_t)S * I), ref((size_t)S * O), got((size_t)S * O, NAN);
    for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) / 7919.0f;
    reference(ref.data(), w.data(), x.data(), S, I, O);
    void *h = coli_cuda_bf16t_upload(dev, w.data(), O, I);
    require(h != nullptr, "upload");
    if (v1) setenv("COLI_SYCL_BF16T_V1", "1", 1);
    const int ok = coli_cuda_bf16t_mv(h, got.data(), O, x.data(), I, S);
    unsetenv("COLI_SYCL_BF16T_V1");
    size_t diff = 0;
    for (size_t i = 0; i < got.size(); i++) diff += std::memcmp(&got[i], &ref[i], 4) != 0;
    std::printf("  %s %s O %d I %d S %d: %zu of %zu values differ\n", ok && !diff ? "ok  " : "FAIL", v1 ? "v1" : "v2", O, I, S, diff, got.size());
    if (!ok || diff) fails++;
    coli_cuda_bf16t_free(h);
}
static void timed(int dev) {
    const int O = 129280, I = 5120;
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    std::vector<float> x((size_t)4 * I, 0.5f), y((size_t)4 * O);
    void *h = coli_cuda_bf16t_upload(dev, w.data(), O, I);
    require(h != nullptr, "upload (timing)");
    for (int v1 = 0; v1 < 2; v1++)
        for (int S : {1, 4}) {
            if (v1) setenv("COLI_SYCL_BF16T_V1", "1", 1);
            for (int k = 0; k < 3; k++) require(coli_cuda_bf16t_mv(h, y.data(), O, x.data(), I, S), "warm");
            const int n = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < n; k++) require(coli_cuda_bf16t_mv(h, y.data(), O, x.data(), I, S), "timed");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
            unsetenv("COLI_SYCL_BF16T_V1");
            std::printf("BF16T-TIME %s O %d I %d S %d: %.3f ms a call (upload, kernel, download), %.0f GB/s of weights\n",
                        v1 ? "v1" : "v2", O, I, S, ms, (double)O * I * 2 / (ms * 1e6));
        }
    coli_cuda_bf16t_free(h);
}
int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    require(coli_cuda_init(dev, n), "init");
    const int sh[][3] = { {64, 8, 1}, {384, 5120, 1}, {384, 5120, 4}, {129280, 5120, 1}, {129280, 5120, 4}, {513, 5120, 3},
                          {1000, 1032, 9}, {257, 4096, 5}, {200, 64, 8}, {33, 520, 2} };
    for (int v1 = 0; v1 < 2; v1++)
        for (auto &p : sh) shape(dev[0], p[0], p[1], p[2], v1);
    timed(dev[0]);
    std::printf(fails ? "BF16T SUITE FAIL\n" : "BF16T SUITE PASS\n");
    return fails ? 1 : 0;
}
