#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "lyaputils.h"

#define USE_PNG 1

#if USE_PNG
#include <png.h>
png_structp png_ptr;
png_infop info_ptr;
#endif

#define LYAP_SETUP_EVENT 0
#define LYAP_REND_EVENT 1
#define LYAP_LOAD_EVENT 2
#define LYAP_DUMP_EVENT 3

static cl_int rendKernel;

static clSetup cl;
static cl_mem clLights = NULL;
static cl_mem clCam = NULL;
static cl_mem clBuffer = NULL;
static cl_mem clImage = NULL;

static LyapPoint *inBuffer;
static size_t inBufferSize;

static size_t bufferCount = 1024*1024;

static unsigned char *tmpBuffer;
static unsigned char *outBuffer;
static size_t outBufferSize;

// Unused
unsigned int windowWidth = 1024;
unsigned int windowHeight = 1024;
unsigned int renderDenominator = 4;


static int setupCL()
{
  memset(&cl, 0, sizeof(cl));
  cl_setup(&cl, 0, 0);
  cl_load(&cl, "common.h", "lyap.cl");

  rendKernel = cl_get_kernel_index(&cl, "lyaprenderrgba");

  size_t outBufferCount = curC->renderWidth*curC->renderHeight;
  outBufferSize = outBufferCount*4;
  outBuffer = (unsigned char *)calloc(outBufferSize, 1);
  if(!outBuffer) {
    fprintf(stderr, "Could not allocate output memory\n");
    return 1;
  }

  if(bufferCount > outBufferCount)
    bufferCount = outBufferCount;

  tmpBuffer = (unsigned char *)calloc(bufferCount, 4);
  if(!tmpBuffer) {
    fprintf(stderr, "Could not allocate temp memory\n");
    return 1;
  }


  clBuffer = clCreateBuffer(cl.context, CL_MEM_READ_ONLY, bufferCount*sizeof(LyapPoint), 0, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for CL");

  clImage = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bufferCount*4, tmpBuffer, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get create image for CL");

  clLights = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MAX_LIGHTS*sizeof(LyapCamLight), Ls, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for lights");

  clCam = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(LyapCamLight), curC, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for cam");

  return cl.err;
}

static int renderCLSetup()
{
  int arg = 4; // Skip buffer(s) and offsets

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(cl_mem), (void *)&clCam);
  CL_ASSERT_RET(&cl, "Could not set parameter cam");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(cl_mem), (void *)&clLights);
  CL_ASSERT_RET(&cl, "Could not set parameter Ls");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(UINT), (void *)&numLights);
  CL_ASSERT_RET(&cl, "Could not set parameter numLights");

  return cl.err;
}

static int renderCL()
{
  int err = CL_SUCCESS;

  cl_uint done = 0;
  cl_uint base = 0;
  size_t numResults = curC->renderWidth * curC->renderHeight;
  size_t blockSize = floorPow2(numResults);
  size_t localSize = cl.workGroupSize[rendKernel];
  int needToSubdivide = localSize > 1;

  fprintf(stderr, "Planning %ld\n", numResults);

  if(blockSize < localSize) localSize = blockSize;
  do {
    base = 0;
    if(blockSize > bufferCount) blockSize = bufferCount;
    if(localSize > blockSize) localSize = blockSize;

    cl.err = clEnqueueWriteBuffer(cl.queue, clBuffer, CL_TRUE, 0, blockSize*sizeof(LyapPoint), inBuffer+done, 0, NULL, &cl.events[LYAP_LOAD_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to enqueue buffer write");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 0, sizeof(cl_mem), (void *)&clBuffer);
    CL_ASSERT_RET(&cl, "Could not set parameter clBuffer");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 1, sizeof(cl_mem), (void *)&clImage);
    CL_ASSERT_RET(&cl, "Could not set parameter clImage");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 2, sizeof(done), (void *)&done);
    CL_ASSERT_RET(&cl, "Failed to set parameter 2 (start)");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 3, sizeof(base), (void *)&base);
    CL_ASSERT_RET(&cl, "Failed to set parameter 3 (base)");

    cl.err = clEnqueueNDRangeKernel(cl.queue, cl.kernels[rendKernel], 1, NULL, &blockSize, &localSize, 1, &cl.events[LYAP_LOAD_EVENT], NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue kernel");
    clReleaseEvent(cl.events[LYAP_LOAD_EVENT]);

    cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_REND_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to enqueue marker");

    cl.err = clWaitForEvents(1, &cl.events[LYAP_REND_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to wait for rend event");
    clReleaseEvent(cl.events[LYAP_REND_EVENT]);

    cl.err = clEnqueueReadBuffer(cl.queue, clImage, CL_TRUE, 0, blockSize*4, outBuffer+done*4, 0, NULL, &cl.events[LYAP_DUMP_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to enqueue buffer read");

    cl.err = clWaitForEvents(1, &cl.events[LYAP_DUMP_EVENT]);
    CL_ASSERT_RET(&cl, "Failed to wait for dump event");
    clReleaseEvent(cl.events[LYAP_DUMP_EVENT]);

    done += blockSize;
    fprintf(stderr, "Done %d\n", done);

    if(done < numResults) {
      size_t remainder = numResults - done;
      if(remainder > cl.workGroupSize[rendKernel]) {
        localSize = cl.workGroupSize[rendKernel];
        blockSize = floorPow2(remainder);
      }
      else {
        localSize = floorPow2(remainder);
        blockSize = localSize;
      }
    }
  } while(needToSubdivide && done < numResults);

  return err;
}

int savePNG(int fd)
{
  FILE *fh;
  if(!(fh = fdopen(fd, "wb"))) {
    perror("Cannot open stdout");
    return 1;
  }

  if(!(png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL))) {
    fprintf(stderr, "Cannot create write structure\n");
    return 1;
  }

  if(!(info_ptr = png_create_info_struct(png_ptr))) {
    fprintf(stderr, "Cannot create info structure\n");
    return 1;
  }

  if(setjmp(png_jmpbuf(png_ptr))) {
    fprintf(stderr, "Error during init_io\n");
    return 1;
  }
  png_init_io(png_ptr, fh);

  if (setjmp(png_jmpbuf(png_ptr))) {
    fprintf(stderr, "Error writing header\n");
    return 1;
  }

  png_set_IHDR(png_ptr, info_ptr, curC->renderWidth, curC->renderHeight,
               8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);

  if (setjmp(png_jmpbuf(png_ptr))) {
    fprintf(stderr, "Error writing bytes\n");
    return 1;
  }

  size_t y=0;
  int i=0;

  for(y=0; y<curC->renderHeight; y++) {
    png_write_row(png_ptr, &(outBuffer[i]));
    i += curC->renderWidth*4;
  }

  // Setup PNG if requested
  if (setjmp(png_jmpbuf(png_ptr))) {
    fprintf(stderr, "Error ending write\n");
    return 1;
  }

  png_write_end(png_ptr, NULL);
  fclose(fh);

  return 0;
}

int main(int argc, char** argv)
{
  char *ofn = argv[--argc];
  char *ifn = argv[--argc];

  int ifd, ofd;
  if(!(ifd = open(ifn, O_RDONLY))) {
    fprintf(stderr, "Could not open input file\n");
    return 1;
  }

  if(loadLyapHeader(ifd)) {
    fprintf(stderr, "Could not read input file\n");
    return 1;
  }

  if(loadLyapData(ifd, &inBuffer, &inBufferSize)) {
    fprintf(stderr, "Could not read input file\n");
    return 1;
  }

LyapCamLight __L1 = {
  .C = {3.0,7.0,5.0,0.000000},
  .Q = {0.039640,0.840027,-0.538582,-0.052093},
  .M = 1.677200,
  .lightInnerCone = 0.904535,
  .lightOuterCone = 0.516497,
  .lightRange = 0.5,
  .ambient = {0.000000,0.000000,0.000000,0.000000},
  .diffuseColor = {0.300000,0.374694,0.20000,0.000000},
  .diffusePower = 10.000000,
  .specularColor = {1.000000,1.0,1.000000,0.000000},
  .specularPower = 25.000000,
  .specularHardness = 10.000000,
  .chaosColor = {0.000000,0.000000,0.000000,0.000000}
};
Ls[1] = __L1;

  curC->renderDenominator = 1;
  camCalculate(curC, curC->renderWidth, curC->renderHeight, 1);
  size_t i;
  for(i=0; i<numLights; i++)
    camCalculate(&Ls[i], curC->renderWidth, curC->renderHeight, 1);

  setupCL();
  renderCLSetup();
  renderCL();

  if(!(ofd = open(ofn, O_CREAT | O_TRUNC | O_WRONLY, 0644))) {
    fprintf(stderr, "Could not open output file\n");
    return 1;
  }

  savePNG(ofd);
  close(ofd);

  return 0;
}
