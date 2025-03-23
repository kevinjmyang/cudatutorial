#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100  // Matrix size

__global__ void matrixMul(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void generateMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 10;  // Random numbers between 0 and 9
    }
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    int* A = (int*)malloc(N * N * sizeof(int));
    int* B = (int*)malloc(N * N * sizeof(int));
    int* C = (int*)malloc(N * N * sizeof(int));

    generateMatrix(A, N * N);
    generateMatrix(B, N * N);

    int* d_A, * d_B, * d_C;
    size_t size = N * N * sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    matrixMul << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
