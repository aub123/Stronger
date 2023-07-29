#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 16
#define BLOCK_SIZE 16

__global__ void transpose(int* A, int* B) {
    __shared__ int tile[BLOCK_SIZE][BLOCK_SIZE+1];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int index_in = y * N + x;
    int index_out = x * N + y;

    // 将输入矩阵元素复制到共享内存中
    tile[threadIdx.y][threadIdx.x] = A[index_in];
    __syncthreads();

    // 将共享内存中的元素写入到输出矩阵中
    B[index_out] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int A[N][N], B[N][N];
    int *d_A, *d_B;

    // 初始化输入矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * N + j;
        }
    }

    // 在设备上分配内存并将输入矩阵复制到设备内存
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // 调用核函数
    transpose<<<numBlocks, threadsPerBlock>>>(d_A, d_B);

    // 将输出矩阵从设备内存复制回主机内存
    cudaMemcpy(B, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出矩阵
    printf("Input Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
    printf("Output Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}