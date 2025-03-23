#include <stdio.h>
#include <cuda_runtime.h>

// Define CHECK macro to catch CUDA errors
#define CHECK(call)                                                          \
{                                                                            \
    const cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                                \
    {                                                                        \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                             \
    }                                                                        \
}

__global__ void nestedHelloWorld(int const iSize, int iDepth, int maxDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1 || iDepth >= maxDepth) return;

    int nthreads = iSize >> 1;

    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld << <1, nthreads >> > (nthreads, ++iDepth, maxDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char** argv)
{
    int size = 8;
    int blocksize = 8;
    int igrid = 1;

    if (argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x, block.x);

    nestedHelloWorld << <grid, block >> > (block.x, 0, 2);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}
