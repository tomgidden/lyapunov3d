#include "cglutils.h"
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

size_t floorPow2(size_t n)
{
  if(n<1) return 0;
  size_t fp2 = 1;
  while((fp2<<1) <= n)
    fp2 <<= 1;
  return fp2;
}

int readfile(const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    off_t file_len;
    struct stat file_status;
    ssize_t ret;

    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret) {
        fprintf(stderr, "Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = file_status.st_size;

    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = read(fd, *result_string, file_len);
    if (!ret) {
        fprintf(stderr, "Error reading from file %s\n", file_name);
        return -1;
    }

    close(fd);

    *string_len = file_len;
    return 0;
}

cl_int cl_cleanup(clSetup *cl)
{
  cl_int err = cl->err;
  size_t i;

  for(i=0; i<cl->numKernels; i++) {
    if(cl->kernels[i]) clReleaseKernel(cl->kernels[i]);
  }

  if(cl->program) clReleaseProgram(cl->program);

  if(cl->queues)
    for(i=0; i<cl->numDevices; ++i)
      if(cl->queues[i]) clReleaseCommandQueue(cl->queues[i]);
  // XXX: Free cl->devices?

  if(cl->queue)
    clReleaseCommandQueue(cl->queue);

  if(cl->context) clReleaseContext(cl->context);

  memset(cl, 0, sizeof(clSetup));
  return err;
}

cl_int cl_setup(clSetup *cl, int multiDevice, int glInterop)
{
  size_t i;

  cl->useMultiDevice = multiDevice;

  if(!(cl->platform)) {
    cl_uint numPlatforms;
    cl_platform_id *platformIDs;
    char chBuffer[1024];

    cl->err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CL_ASSERT_RET(cl, "Failed to get platforms");

    cl->err = (numPlatforms==0);
    CL_ASSERT_RET(cl, "No CL platforms found");

    cl->err = (NULL==(platformIDs = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id))));
    CL_ASSERT_RET(cl, "Failed to allocate memory for clGetPlatformIDs");

    cl->err = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    CL_ASSERT_RET(cl, "Failed to clGetPlatformIDs");

    for(i=0; i<numPlatforms; ++i) {
      cl->err = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
      CL_ASSERT_RET(cl, "Failed to clGetPlatformInfo");

//      if(strstr(chBuffer, "NVIDIA") != NULL) {
        cl->platform = platformIDs[i];
        break;
//      }
    }

    free(platformIDs);

    cl->err = (!cl->platform);
    CL_ASSERT_RET(cl, "NVIDIA platform not found");
  }

  if(!(cl->device || cl->devices)) {
    cl_uint numDevices;
    cl->err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    CL_ASSERT_RET(cl, "Failed to get devices");

    cl->err = (numDevices==0);
    CL_ASSERT_RET(cl, "No CL devices found");

    cl->numDevices = numDevices;

    cl->err = (NULL==(cl->devices = (cl_device_id*)malloc(cl->numDevices*sizeof(cl_device_id))));
    CL_ASSERT_RET(cl, "Failed to allocate memory for clGetDeviceIDs");

    cl->err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, cl->numDevices, cl->devices, NULL);
    CL_ASSERT_RET(cl, "Failed to clGetDeviceIDs");

    cl->bestDevice = 0;
    cl_uint bestMetric = 0;

    i = 0;
    do {
      cl_uint computeUnits;
      cl_uint computeFreq;

      clGetDeviceInfo(cl->devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
      clGetDeviceInfo(cl->devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(computeFreq), &computeFreq, NULL);
      cl_uint metric = computeUnits * computeFreq;

      if(metric > bestMetric) {
        bestMetric = metric;
        cl->bestDevice = i;
      }
      ++i;
    } while(i<numDevices);


    //    if(cl->useMultiDevice) {
    //      cl->context = clCreateContext(0, cl->numDevices, cl->devices, NULL, NULL, &cl->err);
    //      CL_ASSERT_RET(cl, "Failed to create contexts");
    //
    //      cl->err = (NULL==(cl->queues = (cl_command_queue*)malloc(cl->numDevices*sizeof(cl_command_queue))));
    //      CL_ASSERT_RET(cl, "Failed to allocate memory for command queues");
    //
    //      for(i=0; i<numDevices; i++) {
    //        cl->queues[i] = clCreateCommandQueue(cl->context, cl->devices[i], 0, &cl->err);
    //        CL_ASSERT_RET(cl, "Failed to create command queue");
    //  }
    //    }
    //    else {
      cl->device = cl->devices[cl->bestDevice];

      free(cl->devices);
      cl->devices = &cl->device;

      if(glInterop) {

        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

        /// XXX: Need to add interop checking here. See oclSimpleGL
        cl_context_properties props[] = {
          CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
          CL_CONTEXT_PLATFORM, (cl_context_properties)cl->platform,
          0
        };
        cl->context = clCreateContext(props, 1, &cl->device, NULL, NULL, &cl->err);
      }
      else {
        cl_context_properties props[] = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)cl->platform,
          0
        };
        cl->context = clCreateContext(props, 1, &cl->device, NULL, NULL, &cl->err);
      }

      CL_ASSERT_RET(cl, "Failed to create context");

      cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &cl->err);// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl->err);
      CL_ASSERT_RET(cl, "Failed to create command queue");
      cl->queues = &cl->queue;

      cl->numDevices = 1;
      //    }
  }

  return CL_SUCCESS;
}

cl_int cl_get_kernel_index(clSetup *cl, char *kernelName)
{
  char buf[1024];
  size_t k;
  for(k=0; k<cl->numKernels; ++k) {
    cl->err = clGetKernelInfo(cl->kernels[k], CL_KERNEL_FUNCTION_NAME, 1024, (void *)buf, NULL);
    CL_ASSERT_RET(cl, "Failed to get kernel name");
    if(0==strcmp(kernelName, buf))
      return k;
  }
  return -1;
}

cl_int cl_load(clSetup *cl, char *filename1, char *filename2)
{
  char *srcs[2];
  size_t lens[2];

  cl->err = readfile(filename1, srcs, lens);
  CL_ASSERT_RET(cl, "Failed to load kernel source");

  if(filename2) {
    cl->err = readfile(filename2, srcs+1, lens+1);
    CL_ASSERT_RET(cl, "Failed to load kernel source #2");
  }

  cl->program = clCreateProgramWithSource(cl->context, filename2?2:1, (const char **)srcs, NULL, &cl->err);
  CL_ASSERT_RET(cl, "Failed to create program");

  char *params;
  if(cl->useFastMath)
    if(cl->useNativeMath)
      params = "-DUSE_NATIVE_MATH=1 -DUSE_FAST_MATH=1 -cl-fast-relaxed-math";
    else
      params = "-DUSE_FAST_MATH=1 -cl-fast-relaxed-math";
  else
    if(cl->useNativeMath)
      params = "-DUSE_NATIVE_MATH=1";
    else
      params = "";

  size_t i;
  for(i=0; i<cl->numDevices; ++i) {
    cl->err = clBuildProgram(cl->program, 1, &cl->devices[i], params, NULL, NULL);
    if(cl->err != CL_SUCCESS) {
      size_t blen;
      char *blog;
      clGetProgramBuildInfo(cl->program, cl->devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &blen);
      if(!(blog = (char *)malloc(blen))) {
        cl->err = 1;
        CL_ASSERT_RET(cl, "Failed to build program");
      }

      clGetProgramBuildInfo(cl->program, cl->devices[i], CL_PROGRAM_BUILD_LOG, blen, blog, NULL);
      cl->err = 1;
      CL_ASSERT_RET(cl, blog);
    }
  }

  cl->err = clCreateKernelsInProgram(cl->program, CL_MAX_KERNELS, cl->kernels, &cl->numKernels);
  CL_ASSERT_RET(cl, "Failed to create compute kernels");

  size_t j;
  for(j=0; j<cl->numKernels; ++j) {
    if(!(cl->workGroupSizes[j] = (size_t *)malloc(sizeof(size_t)*cl->numDevices))) {
      cl->err = 1;
      CL_ASSERT_RET(cl, "Failed to allocate space for work group sizes");
    }

    for(i=0; i<cl->numDevices; ++i) {
      cl->err = clGetKernelWorkGroupInfo(cl->kernels[j], cl->devices[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), cl->workGroupSizes[j], NULL);

      //      fprintf(stderr, "Got size %ld on kernel %ld on device %ld\n", cl->workGroupSizes[j][i], j, i);

      CL_ASSERT_RET(cl, "Failed to get kernel work group info");
    }

    if(!cl->useMultiDevice) {
      cl->workGroupSize[j] = floorPow2(cl->workGroupSizes[j][0]);
    }
  }

  return CL_SUCCESS;
}

int cl_load_sequence(clSetup *cl, cl_mem *clMemP, char *seqStr)
{
  size_t seqLength = 10 * strlen(seqStr) + 1;
  int *seq = (int*)malloc(seqLength * sizeof(cl_int));
  if(!seq) {
    fprintf(stderr, "Cannot allocate sequence memory (%ld)\n", seqLength);
    exit(1);
  }

  int last = 1;
  char *seqLetter = seqStr;
  int *seqp = seq;
  do {
    switch (*seqLetter) {
    case '9': *seqp++ = last;
    case '8': *seqp++ = last;
    case '7': *seqp++ = last;
    case '6': *seqp++ = last;
    case '5': *seqp++ = last;
    case '4': *seqp++ = last;
    case '3': *seqp++ = last;
    case '2': *seqp++ = last;
    case '1': *seqp++ = last; break;

    case 'a': case 'A': *seqp++ = last = 0; break;
    case 'b': case 'B': *seqp++ = last = 1; break;
    case 'c': case 'C': *seqp++ = last = 2; break;
    case 'd': case 'D': *seqp++ = last = 3; break;

    default:
      fprintf(stderr, "Bad sequence letter '%c'\n", *seqLetter);
      exit(1);
    }
  }
  while (*(++seqLetter));
  *seqp = -1;

  cl_mem clSeq = clCreateBuffer(cl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, seqLength*sizeof(*seq), seq, &cl->err);
  CL_ASSERT_RET(cl, "Cannot load sequence");

  free(seq);

  *clMemP = clSeq;

  return cl->err;
}

char* cl_error(int errno)
{
  switch (errno) {
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  default: return "UNKNOWN";
  }
}
