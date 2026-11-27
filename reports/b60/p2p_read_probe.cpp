#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() try {
    std::vector<sycl::device> devices;
    for(const auto &p:sycl::platform::get_platforms())
        if(p.get_backend()==sycl::backend::ext_oneapi_level_zero)
            for(const auto &d:p.get_devices(sycl::info::device_type::gpu))devices.push_back(d);
    if(devices.size()!=2)throw std::runtime_error("expected exactly two Level Zero GPUs");
    for(int d=0;d<2;d++){
        if(!devices[d].ext_oneapi_can_access_peer(devices[1-d]))throw std::runtime_error("peer access unavailable");
        devices[d].ext_oneapi_enable_peer_access(devices[1-d]);
    }
    sycl::context ctx(devices);
    sycl::queue q0(ctx,devices[0],sycl::property::queue::in_order{});
    sycl::queue q1(ctx,devices[1],sycl::property::queue::in_order{});
    sycl::queue *q[]={&q0,&q1};
    constexpr size_t max_bytes=64u*1024*1024, count=max_bytes/sizeof(uint32_t);
    uint32_t *source[2],*dest[2];
    for(int d=0;d<2;d++){
        source[d]=sycl::malloc_device<uint32_t>(count,devices[d],ctx);
        dest[d]=sycl::malloc_device<uint32_t>(count,devices[d],ctx);
        if(!source[d]||!dest[d])throw std::runtime_error("device allocation failed");
    }
    std::vector<uint32_t> in(count),out(count);
    for(size_t i=0;i<count;i++)in[i]=(uint32_t)(i*2654435761u)^0x5a31b7cdu;
    for(int d=0;d<2;d++)q[d]->memcpy(source[d],in.data(),max_bytes).wait_and_throw();
    for(size_t bytes:{size_t(8192),size_t(1048576),max_bytes})for(int d=0;d<2;d++)for(int remote=0;remote<2;remote++){
        auto *src=source[remote?1-d:d],*dst=dest[d];size_t n=bytes/4;
        std::vector<double> times;
        for(int rep=0;rep<5;rep++){
            auto start=std::chrono::steady_clock::now();
            q[d]->parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){dst[i]=src[i]^0x013579bdu;}).wait_and_throw();
            auto stop=std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(stop-start).count());
        }
        q[d]->memcpy(out.data(),dst,bytes).wait_and_throw();
        for(size_t i=0;i<n;i++)if(out[i]!=(in[i]^0x013579bdu))throw std::runtime_error("kernel output mismatch");
        std::sort(times.begin(),times.end());
        std::printf("execute=%d source=%d bytes=%zu median_us=%.3f GB_per_s=%.3f verified=yes\n",
                    d,remote?1-d:d,bytes,times[2]*1e6,bytes/times[2]/1e9);
    }
    for(int d=0;d<2;d++){sycl::free(source[d],ctx);sycl::free(dest[d],ctx);}
    std::puts("peer-read kernel PASS");return 0;
}catch(const std::exception &e){std::fprintf(stderr,"peer-read kernel FAILED: %s\n",e.what());return 1;}
