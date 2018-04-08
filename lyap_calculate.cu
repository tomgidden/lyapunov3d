#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, cuda
#include <driver_functions.h>
#include <cuda_runtime.h>

// CUDA utilities and system includes from the CUDA SDK samples
#include <helper_cuda.h>
#include <helper_functions.h>

#include "kernel.hpp"
#include "scene.hpp"
#include "params.hpp"

// Image and grid parameters
const unsigned int volumeWidth = 512;
const unsigned int volumeHeight = volumeWidth;
const unsigned int volumeDepth = volumeWidth;
const unsigned int blockSize = 8;
const dim3 blocks(volumeWidth / blockSize, volumeHeight / blockSize, volumeDepth / blockSize);
const dim3 threads(blockSize, blockSize, blockSize);

LyapParams *curP = &prm;
LyapCam *curC = &cam;
unsigned int curL = 0;

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif

// Device array of lyapunov exponents
float *cudaExps = 0;

// Device sequence array
Int *cudaSeq;


void cuda_load_sequence(unsigned char *seqStr)
{
    size_t actual_length;
    Int *seq;

    actual_length = scene_convert_sequence(&seq, seqStr);

    checkCudaErrors(cudaMalloc(&cudaSeq, actual_length * sizeof(Int)));
    checkCudaErrors(cudaMemcpy(cudaSeq, seq, actual_length * sizeof(Int), cudaMemcpyHostToDevice));

    free(seq);
}

void render()
{
    params_init();

    cuda_load_sequence(sequence);

    size_t expsSize = sizeof(float) * volumeWidth * volumeHeight * volumeDepth;

    // Allocate points memory
    checkCudaErrors(cudaMalloc(&cudaExps, expsSize));

    // call CUDA kernel, writing results to PBO
    //    for(int i = 0; i < passes; ++i) {
    //        void *dummy;
    kernel_calc_volume<<<blocks, threads>>>(cudaExps, prm, cudaSeq);
    //        cudaMemcpyAsync(dummy, dummy, 1, cudaMemcpyDeviceToDevice);
    //    }
    getLastCudaError("kernel failed");

    printf("Points size = %ld\n", expsSize);

    float *myExps = (float *)malloc(expsSize);
    printf("malloc'ed %p.\n", myExps);

    checkCudaErrors(cudaMemcpy( myExps, cudaExps, expsSize, cudaMemcpyDeviceToHost ));

    FILE *fp = fopen("exps.raw", "wb");
    fwrite(myExps, 1, expsSize, fp);
    fclose(fp);

    free(myExps);

    getLastCudaError("dump failed");
}

void cleanup()
{
    checkCudaErrors(cudaFree(cudaExps));
    checkCudaErrors(cudaFree(cudaSeq));
}

int choose_cuda_device(int argc, char **argv, bool use_gl)
{
    int result = 0;

    result = findCudaDevice(argc, (const char **)argv);

    return result;
}

int main(int argc, char **argv)
{
    // Use command-line specified CUDA device, otherwise use device with
    // highest Gflops/s
    choose_cuda_device(argc, argv, true);

    render();

    cleanup();

    exit(EXIT_SUCCESS);
}
