#include <stdio.h>
#include <cuda.h>

__global__ void print_thread_index() {
    printf("threadIdx.x: %d\n", threadIdx.x);
    printf("blockIdx.x: %d\n", blockIdx.x);
}

int main() {
    // print the order of the thread and block
    print_thread_index<<<5, 15>>>();
    cudaDeviceSynchronize();
    return 0;
}