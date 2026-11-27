#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() try {
    std::vector<sycl::device> dev;
    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto &d : platform.get_devices(sycl::info::device_type::gpu)) dev.push_back(d);
    }
    if (dev.size() != 2) throw std::runtime_error("expected exactly two visible Level Zero GPUs");
    for (int i=0;i<2;i++) {
        if (!dev[i].ext_oneapi_can_access_peer(dev[1-i])) throw std::runtime_error("peer access unavailable");
        dev[i].ext_oneapi_enable_peer_access(dev[1-i]);
    }
    sycl::context ctx(dev);
    sycl::queue q0(ctx, dev[0], sycl::property::queue::in_order{});
    sycl::queue q1(ctx, dev[1], sycl::property::queue::in_order{});
    constexpr size_t max_bytes=64u*1024*1024;
    auto *a=sycl::malloc_device<std::uint32_t>(max_bytes/4,dev[0],ctx);
    auto *b=sycl::malloc_device<std::uint32_t>(max_bytes/4,dev[1],ctx);
    if(!a || !b) throw std::runtime_error("peer buffers unavailable");
    std::vector<std::uint32_t> in(max_bytes/4), out(max_bytes/4);
    for(size_t i=0;i<in.size();i++) in[i]=(std::uint32_t)(i*2654435761u)^0x5a31b7cdu;
    q0.memcpy(a,in.data(),max_bytes).wait_and_throw();
    for(size_t n : {size_t(8192),size_t(65536),size_t(1048576),max_bytes}) {
        for(int dir=0;dir<2;dir++) {
            auto &q=dir ? q0 : q1;
            auto *dst=dir ? a : b;
            auto *src=dir ? b : a;
            if(dir) q1.memcpy(b,in.data(),n).wait_and_throw();
            std::vector<double> times;
            for(int i=0;i<5;i++) {
                auto start=std::chrono::steady_clock::now();
                q.memcpy(dst,src,n).wait_and_throw();
                auto end=std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double>(end-start).count());
            }
            q.memcpy(out.data(),dst,n).wait_and_throw();
            if(!std::equal(in.begin(),in.begin()+n/4,out.begin())) throw std::runtime_error("peer copy mismatch");
            std::sort(times.begin(),times.end());
            std::printf("%d->%d bytes=%zu median_us=%.3f GB_per_s=%.3f verified=yes\n",
                        dir?1:0,dir?0:1,n,times[2]*1e6,n/times[2]/1e9);
        }
    }
    sycl::free(a,ctx); sycl::free(b,ctx);
    return 0;
} catch(const std::exception &e) {
    std::fprintf(stderr,"P2P probe failed: %s\n",e.what());
    return 1;
}
