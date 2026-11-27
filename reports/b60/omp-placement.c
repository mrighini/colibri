#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>

int main(void) {
    #pragma omp parallel
    {
        cpu_set_t set; CPU_ZERO(&set); sched_getaffinity(0,sizeof(set),&set);
        #pragma omp critical
        {
            printf("thread=%d place=%d cpus=",omp_get_thread_num(),omp_get_place_num());
            for(int i=0;i<CPU_SETSIZE;i++)if(CPU_ISSET(i,&set))printf("%d,",i);
            putchar('\n');
        }
    }
    return 0;
}
