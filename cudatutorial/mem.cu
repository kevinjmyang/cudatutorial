#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <windows.h>

#define MEGABYTE    (1024 * 1024)
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
double seconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char** argv)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    if (argc != 2) {
        printf("usage: %s <size-in-mbs>\n", argv[0]);
        return 1;
    }

    int n_mbs = atoi(argv[1]);
    unsigned int nbytes = n_mbs * MEGABYTE;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting at ", argv[0]);
    printf("device %d: %s nbyte %5.2fMB\n", dev,
        deviceProp.name, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory
    double start = seconds();
    float* h_a = (float*)malloc(nbytes);
    double elapsed = seconds() - start;
    printf("Host memory allocation took %2.10f us\n", elapsed * 1000000.0);

    // allocate the device memory
    float* d_a;
    CHECK(cudaMalloc((float**)&d_a, nbytes));

    // initialize the host memory
    for (unsigned int i = 0; i < nbytes / sizeof(float); i++) h_a[i] = 0.5f;

    // transfer data from the host to the device
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_a));

    LARGE_INTEGER frequency;
    LARGE_INTEGER StartingTime, EndingTime;
    double ElapsedTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&StartingTime);

    free(h_a);
    QueryPerformanceCounter(&EndingTime);
    ElapsedTime = (double)(EndingTime.QuadPart - StartingTime.QuadPart) / frequency.QuadPart;
    printf("Elapsed Time: %f seconds", ElapsedTime);
    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
