// B60 command-graph replay smoke test and dispatch microbenchmark.
// This does not execute the Qwen model or implement its graph path.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace sycl_exp = sycl::ext::oneapi::experimental;
using clock_type = std::chrono::steady_clock;

static void enqueue_add(sycl::queue &q, std::uint32_t *data, std::size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { data[i] += 1; });
}

static double micros(clock_type::time_point t) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - t).count();
}

int main(int argc, char **argv) try {
    const int ordinal = argc > 1 ? std::atoi(argv[1]) : 0;
    constexpr int nodes = 64, repetitions = 40;
    constexpr std::size_t elements = 1024;
    std::vector<sycl::device> devices;
    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto &device : platform.get_devices(sycl::info::device_type::gpu))
            devices.push_back(device);
    }
    if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= devices.size()) {
        std::fprintf(stderr, "invalid GPU ordinal %d (found %zu)\n", ordinal, devices.size());
        return 2;
    }
    const auto &device = devices[ordinal];
    const bool full = device.has(sycl::aspect::ext_oneapi_graph);
    const bool limited = device.has(sycl::aspect::ext_oneapi_limited_graph);
    std::printf("device=%d name=%s pci=%s graph_full=%d graph_limited=%d\n", ordinal,
                device.get_info<sycl::info::device::name>().c_str(),
                device.get_info<sycl::ext::intel::info::device::pci_address>().c_str(), full, limited);
    std::fflush(stdout);
    if (!full && !limited) { std::puts("graph_unsupported"); return 0; }
    sycl::queue q(device, sycl::property::queue::in_order{});
    auto *data = sycl::malloc_device<std::uint32_t>(elements, q);
    if (!data) { std::fprintf(stderr, "device allocation failed\n"); return 1; }
    std::vector<std::uint32_t> out(elements);
    auto reset = [&] { q.memset(data, 0, elements * sizeof(*data)).wait_and_throw(); };
    auto verify = [&] {
        q.memcpy(out.data(), data, elements * sizeof(*data)).wait_and_throw();
        for (auto value : out) if (value != nodes * repetitions) return false;
        return true;
    };
    reset();
    enqueue_add(q, data, elements);
    q.wait_and_throw();  // Load the kernel before timing either path.
    reset();
    auto t = clock_type::now();
    for (int r = 0; r < repetitions; ++r) {
        for (int k = 0; k < nodes; ++k) enqueue_add(q, data, elements);
        q.wait_and_throw();
    }
    const double direct_us = micros(t) / repetitions;
    if (!verify()) { std::fprintf(stderr, "direct result mismatch\n"); return 1; }

    t = clock_type::now();
    sycl_exp::command_graph graph(q.get_context(), q.get_device());
    graph.begin_recording(q);
    for (int k = 0; k < nodes; ++k) enqueue_add(q, data, elements);
    graph.end_recording();
    auto executable = graph.finalize();
    const double setup_us = micros(t);
    q.ext_oneapi_graph(executable).wait_and_throw();  // Warm graph replay.
    reset();
    t = clock_type::now();
    for (int r = 0; r < repetitions; ++r) q.ext_oneapi_graph(executable).wait_and_throw();
    const double graph_us = micros(t) / repetitions;
    if (!verify()) { std::fprintf(stderr, "graph result mismatch\n"); return 1; }
    std::printf("nodes=%d repetitions=%d elements=%zu direct_us_per_frame=%.3f graph_us_per_frame=%.3f graph_setup_us=%.3f correctness=pass\n",
                nodes, repetitions, elements, direct_us, graph_us, setup_us);
    sycl::free(data, q);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "graph probe failed: %s\n", e.what());
    return 1;
}
