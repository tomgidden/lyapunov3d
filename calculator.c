#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "lyaputils.h"
#include <time.h>

#define FIX_BLOCKSIZE 1024*100

//#define TWIDTH 768
//#define THEIGHT 512
#define TWIDTH 1024
#define THEIGHT 1024
//#define SETTLE 240
//#define ACCUM 240
//#define DEPTH 512
//#define NEAR_MUL 16
//#define STEP_METHOD 2

#define LYAP_CALC_EVENT 0

static cl_int calcKernel;

static clSetup cl;
static cl_mem clSeq = NULL;
static cl_mem clCam = NULL;
static cl_mem clPrm = NULL;

static cl_mem clBuffer = NULL;

static LyapPoint *outBuffer = NULL;
static size_t outBufferSize = 0;

// Unused
unsigned int windowWidth = 1024;
unsigned int windowHeight = 1024;
unsigned int renderDenominator = 4;


static int setupCL()
{
  memset(&cl, 0, sizeof(cl));
  cl_setup(&cl, 0, 0);
  cl_load(&cl, "common.h", "lyap.cl");

  calcKernel = cl_get_kernel_index(&cl, "lyapcalc");

  cl_load_sequence(&cl, &clSeq, sequence);

#if FIX_BLOCKSIZE
  outBufferSize = FIX_BLOCKSIZE*sizeof(LyapPoint);
#else
  outBufferSize = curC->renderWidth*curC->renderHeight*sizeof(LyapPoint);
#endif

  outBuffer = (LyapPoint *)calloc(outBufferSize,1);
  if(!outBuffer) {
    fprintf(stderr, "Could not allocate output memory\n");
    return 1;
  }

#if FIX_BLOCKSIZE
  clBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, FIX_BLOCKSIZE*sizeof(LyapPoint), 0, &cl.err);
#else
  clBuffer = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, outBufferSize, 0, &cl.err);
#endif
  CL_ASSERT_RET(&cl, "Failed to get create buffer for CL");

  clCam = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(LyapCamLight), curC, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for cam");

  clPrm = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(LyapParams), curP, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for prm");

  return cl.err;
}

static int calcCLSetup()
{
  cl.err = clEnqueueWriteBuffer(cl.queue, clCam, CL_FALSE, 0, sizeof(LyapCamLight), curC, 0, NULL, NULL);
  CL_ASSERT_RET(&cl, "Failed to update cam buffer");

  cl.err = clEnqueueWriteBuffer(cl.queue, clPrm, CL_FALSE, 0, sizeof(LyapParams), curP, 0, NULL, NULL);
  CL_ASSERT_RET(&cl, "Failed to update prm buffer");

  int arg = 3; // Skip buffer and offsets

  cl.err = clSetKernelArg(cl.kernels[calcKernel], arg++, sizeof(cl_mem), (void *)&clCam);
  CL_ASSERT_RET(&cl, "Could not set parameter cam");

  cl.err = clSetKernelArg(cl.kernels[calcKernel], arg++, sizeof(cl_mem), (void *)&clPrm);
  CL_ASSERT_RET(&cl, "Could not set parameter prm");

  cl.err = clSetKernelArg(cl.kernels[calcKernel], arg++, sizeof(cl_mem), (void *)&clSeq);
  CL_ASSERT_RET(&cl, "Could not set parameter sequence");

  return cl.err;
}

static int calcCL(char *ofn)
{
  int err = CL_SUCCESS;

  cl.err = clSetKernelArg(cl.kernels[calcKernel], 0, sizeof(cl_mem), (void *)&clBuffer);
  CL_ASSERT_RET(&cl, "Could not set parameter clBuffer");

  off_t file_off = 0;
  cl_uint done = 0;
  cl_uint base = 0;
  size_t numResults = curC->renderWidth * curC->renderHeight;
  size_t blockSize = floorPow2(numResults);
  size_t localSize = cl.workGroupSize[calcKernel];
  int needToSubdivide = localSize > 1;

  time_t otm = time(NULL);
  time_t tm = otm;

  fprintf(stderr, "Planning %ld\n", numResults);

  if(blockSize < localSize) localSize = blockSize;
#if FIX_BLOCKSIZE
  size_t maxBlockSize = FIX_BLOCKSIZE;
#endif

  do {
#if FIX_BLOCKSIZE
      if(blockSize > maxBlockSize) blockSize = maxBlockSize;
#endif
    base = done;

    cl.err = clSetKernelArg(cl.kernels[calcKernel], 1, sizeof(done), (void *)&done);
    CL_ASSERT_RET(&cl, "Failed to set kernel parameter 1 (start)");

    cl.err = clSetKernelArg(cl.kernels[calcKernel], 2, sizeof(base), (void *)&base);
    CL_ASSERT_RET(&cl, "Failed to set kernel parameter 2 (base)");

    cl.err = clEnqueueNDRangeKernel(cl.queue, cl.kernels[calcKernel], 1, NULL, &blockSize, &localSize, 0, NULL, NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue kernel");

#if FIX_BLOCKSIZE
    cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_CALC_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to enqueue marker");

    cl.err = clEnqueueReadBuffer(cl.queue, clBuffer, CL_TRUE, 0, sizeof(LyapPoint)*blockSize, outBuffer, 1, &cl.events[LYAP_CALC_EVENT], NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue buffer read");

    file_off = saveLyapBlock(ofn, 0, outBuffer, sizeof(LyapPoint)*blockSize);
#endif

    done += blockSize;

    tm = time(NULL);
    time_t rtm = (numResults-done) * (tm-otm)/done;
    fprintf(stderr, "Done %d\t%ld%%\t%lds elapsed\t%lds remaining\n", done, 100*done/numResults, tm-otm, rtm);

    if(done < numResults) {
      size_t remainder = numResults - done;
      if(remainder > cl.workGroupSize[calcKernel]) {
        localSize = cl.workGroupSize[calcKernel];
        blockSize = floorPow2(remainder);
      }
      else {
        localSize = floorPow2(remainder);
        blockSize = localSize;
      }
    }
  } while(needToSubdivide && done < numResults);

#if !FIX_BLOCKSIZE
    cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_CALC_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to enqueue marker");

    cl.err = clEnqueueReadBuffer(cl.queue, clBuffer, CL_TRUE, 0, outBufferSize, outBuffer, 1, &cl.events[LYAP_CALC_EVENT], NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue buffer read");
#endif


  return err;
}

int main(int argc, char** argv)
{
#include "params.h"

#if defined(TWIDTH) && defined(THEIGHT)
  camCalculate(curC, TWIDTH, THEIGHT, 1);
#else
  camCalculate(curC, 0, 0, 1);
#endif

#ifdef SETTLE
  curP->settle = SETTLE;
#endif
#ifdef ACCUM
  curP->accum = ACCUM;
#endif
#ifdef DEPTH
  curP->depth = DEPTH;
#endif
#ifdef NEAR_MUL
  curP->nearMultiplier = NEAR_MUL;
#endif
#ifdef STEP_METHOD
  curP->stepMethod = STEP_METHOD;
#endif

  if(argc != 2) {
    fprintf(stderr, "Syntax: %s <fn>\n",argv[0]);
    return 1;
  }

  char *ofn = argv[--argc];

  int fd;

  if(!(fd = open(ofn, O_CREAT|O_TRUNC|O_WRONLY, 0644))) {
    fprintf(stderr, "Could not open output file\n");
    return 1;
  }

  if(setupCL()) return 1;
  if(calcCLSetup()) return 1;

  if(saveLyapHeader(fd)) return 1;
  close(fd);

  if(calcCL(ofn)) return 1;

#if !FIX_BLOCKSIZE
  if(saveLyapBlock(ofn, 0, outBuffer, outBufferSize)) return 1;
#endif

  return 0;
}
