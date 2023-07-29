#include <stdio.h>
#include "hello_from_gpu.cuh"

int main(void)
{
    hello_from_gpu<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}