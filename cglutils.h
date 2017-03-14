#include <GL/glew.h>

#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#include <OpenCL/opencl.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#define CL_LOG_FN clLogMessagesToStderrAPPLE

#define CL_MAX_KERNELS 16
#define CL_MAX_EVENTS 16

typedef struct {
  int useFastMath;
  int useNativeMath;

  cl_platform_id platform;

  cl_context context;

  int useMultiDevice;
  cl_uint numDevices;

  // Single-device:
  cl_command_queue queue;
  cl_device_id device;
  size_t workGroupSize[CL_MAX_KERNELS];

  // Multi-device:
  cl_command_queue *queues;
  cl_device_id *devices;
  size_t *workGroupSizes[CL_MAX_KERNELS];
  size_t bestDevice;

  cl_program program;
  cl_uint numKernels;
  cl_kernel kernels[CL_MAX_KERNELS];

  cl_event events[CL_MAX_EVENTS];

  cl_int err;
} clSetup;

#define CL_ASSERT_RET(cl, msg) {                          \
    if((cl)->err != CL_SUCCESS) {                         \
      fprintf(stderr, "%s:%d (%s):\t%d (%s) %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, (cl)->err, cl_error((cl)->err), msg); \
      return cl_cleanup(cl);                              \
    }                                                     \
  }

extern cl_int cl_cleanup(clSetup *cl);
extern cl_int cl_setup(clSetup *cl, int multiDevice, int glInterop);
extern cl_int cl_load(clSetup *cl, char *filename1, char *filename2);
extern size_t floorPow2(size_t n);
extern cl_int cl_get_kernel_index(clSetup *cl, char *kernelName);
extern int cl_load_sequence(clSetup *cl, cl_mem *clMemP, char *seqStr);
extern char *cl_error(int errno);
