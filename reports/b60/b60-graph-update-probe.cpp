// Check graph update and stable-USM parameters on one B60, not model inference.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace sycl_exp = sycl::ext::oneapi::experimental;
using clock_type = std::chrono::steady_clock;
struct Parameters { int offset; int value; };

int main(int argc, char **argv) try {
    const int ordinal = argc > 1 ? std::atoi(argv[1]) : 0;
    constexpr int iterations = 32;
    std::vector<sycl::device> devices;
    for (const auto &p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
            for (const auto &d : p.get_devices(sycl::info::device_type::gpu)) devices.push_back(d);
    if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= devices.size()) return 2;
    const auto &d = devices[ordinal];
    std::printf("device=%d pci=%s graph_full=%d graph_limited=%d\n", ordinal,
                d.get_info<sycl::ext::intel::info::device::pci_address>().c_str(),
                d.has(sycl::aspect::ext_oneapi_graph), d.has(sycl::aspect::ext_oneapi_limited_graph));
    std::fflush(stdout);
    if (!d.has(sycl::aspect::ext_oneapi_graph)) { std::puts("updatable_graph_unsupported"); return 0; }
    sycl::queue q(d, sycl::property::queue::in_order{});
    int *out = sycl::malloc_device<int>(iterations, q);
    Parameters *param = sycl::malloc_device<Parameters>(1, q);
    if (!out || !param) return 1;
    std::vector<int> host(iterations);
    auto reset = [&] { q.memset(out, 0, iterations * sizeof(int)).wait_and_throw(); };
    auto valid = [&] {
        q.memcpy(host.data(), out, iterations * sizeof(int)).wait_and_throw();
        for (int i = 0; i < iterations; i++) if (host[i] != i + 101) return false;
        return true;
    };

    // Whole-graph update changes captured per-replay scalar kernel arguments.
    auto record = [&](int offset, int value) {
        sycl_exp::command_graph g(q.get_context(), q.get_device());
        g.begin_recording(q);
        q.single_task([=] { out[offset] = value; });
        g.end_recording();
        return g;
    };
    reset();
    auto original = record(0, 101);
    auto executable = original.finalize({sycl_exp::property::graph::updatable{}});
    q.ext_oneapi_graph(executable).wait_and_throw();
    const auto start = clock_type::now();
    for (int i = 1; i < iterations; i++) {
        auto replacement = record(i, i + 101);
        executable.update(replacement);
        q.ext_oneapi_graph(executable).wait_and_throw();
    }
    const double update_us = std::chrono::duration<double, std::micro>(clock_type::now() - start).count() / (iterations - 1);
    if (!valid()) { std::fprintf(stderr, "whole-graph update result mismatch\n"); return 1; }
    std::printf("whole_graph_update=pass iterations=%d us_per_update_and_replay=%.3f\n", iterations, update_us);

    // Same graph each time; change its arguments indirectly through stable USM.
    reset();
    sycl_exp::command_graph indirect(q.get_context(), q.get_device());
    indirect.begin_recording(q);
    q.single_task([=] { out[param->offset] = param->value; });
    indirect.end_recording();
    auto replay = indirect.finalize();
    const auto start2 = clock_type::now();
    for (int i = 0; i < iterations; i++) {
        const Parameters p{i, i + 101};
        q.memcpy(param, &p, sizeof p).wait_and_throw();
        q.ext_oneapi_graph(replay).wait_and_throw();
    }
    const double indirect_us = std::chrono::duration<double, std::micro>(clock_type::now() - start2).count() / iterations;
    if (!valid()) { std::fprintf(stderr, "indirect parameter result mismatch\n"); return 1; }
    std::printf("stable_usm_parameter=pass iterations=%d us_per_copy_and_replay=%.3f\n", iterations, indirect_us);
    sycl::free(param, q);
    sycl::free(out, q);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "graph update probe failed: %s\n", e.what());
    return 1;
}
