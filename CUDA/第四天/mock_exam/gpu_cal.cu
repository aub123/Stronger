#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void findMax(int *data, int *max1, int *max2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每个线程读取一个元素
    int value = data[i];

    // 使用并行归约算法找到全局最大值和次大值
    __shared__ int s_max1;
    __shared__ int s_max2;

    // 将全局最大值和次大值初始化为数组的前两个元素
    if (tid == 0) {
        s_max1 = data[0];
        s_max2 = data[1];
    }
    __syncthreads();

    // 使用并行归约算法找到全局最大值和次大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (value > data[i + s]) {
                value = data[i + s];
            }
            int tmp_max1 = s_max1;
            int tmp_max2 = s_max2;
            if (tmp_max1 < tmp_max2) {
                int tmp = tmp_max1;
                tmp_max1 = tmp_max2;
                tmp_max2 = tmp;
            }
            if (value > tmp_max1) {
                s_max2 = s_max1;
                s_max1 = value;
            } else if (value > tmp_max2) {
                s_max2 = value;
            }
        }
        __syncthreads();
    }

    // 将每个块的最大值和次大值更新到全局变量
    if (tid == 0) {
        atomicMax(max1, s_max1);
        atomicMax(max2, s_max2);
    }
}

int main() {
    int data[N];
    int max1 = 0, max2 = 0;
    int *d_data, *d_max1, *d_max2;

    // 初始化数组
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100000;
    }

    // 在设备上分配内存并将数组复制到设备内存
    cudaMalloc((void **)&d_data, N * sizeof(int));
    cudaMalloc((void **)&d_max1, sizeof(int));
    cudaMalloc((void **)&d_max2, sizeof(int));
    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max1, &max1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max2, &max2, sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 调用核函数
    findMax<<<numBlocks, threadsPerBlock>>>(d_data, d_max1, d_max2);

    // 将最大值和次大值从设备内存复制回主机内存
    cudaMemcpy(&max1, d_max1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max2, d_max2, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印最大值和次大值
    printf("Max1: %d\nMax2: %d\n", max1, max2);

    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_max1);
    cudaFree(d_max2);

    return 0;
}