#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include "lyaputils.h"

#define LYAP_CALC_EVENT 0
#define LYAP_ACQGL_EVENT 1
#define LYAP_REND_EVENT 2
#define LYAP_RELGL_EVENT 3

static cl_int calcrendKernel;
static cl_int rendKernel;

static clSetup cl;
static cl_mem clSeq = NULL;
static cl_mem clLights = NULL;
static cl_mem clCam = NULL;
static cl_mem clPrm = NULL;

static GLuint textureId = 0;

static cl_mem clBuffer = NULL;
static cl_mem clImage = NULL;

static int update_calc = 1;
static int update_render = 0;
static int update_lights = 1;

static int window;

static unsigned int curL = 0;

static LyapCamLight L0;

unsigned int windowWidth = 1024;
unsigned int windowHeight = 1024;
unsigned int renderDenominator = 4;


static void save()
{
  int fd = open("params.h", O_TRUNC | O_CREAT | O_WRONLY);
  if(!fd) perror("save() failed");
  dumpLyapHeader(fd);
  close(fd);
}

static void setupGL()
{
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glEnable(GL_TEXTURE_2D);

  char *data = calloc(windowWidth*windowHeight, 4);
  char *ptr = data;
  size_t i;
  for(i=0; i<windowWidth*windowHeight; i++) {
    *ptr++ = 0;
    *ptr++ = 0;
    *ptr++ = i>>8;
    *ptr++ = 255;
  }

  if(textureId)
    glDeleteTextures(1, &textureId);

  glGenTextures(1, &textureId);

  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
  free(data);

  glBindTexture(GL_TEXTURE_2D, 0);
}

static void setupLight(unsigned int l)
{
  Ls[l] = L0;
  numLights = l+1;
  update_lights = 1;
}




static int setupCL()
{
  memset(&cl, 0, sizeof(cl));
  cl_setup(&cl, 0, 1);
  cl_load(&cl, "common.h", "lyap.cl");

  calcrendKernel = cl_get_kernel_index(&cl, "lyapcalcrendergl");
  rendKernel = cl_get_kernel_index(&cl, "lyaprendergl");

  cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_CALC_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to enqueue marker");

  cl_load_sequence(&cl, &clSeq, sequence);

  clBuffer = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, windowWidth*windowHeight*sizeof(LyapPoint), 0, &cl.err);

  clImage = clCreateFromGLTexture(cl.context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, textureId, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get GL image for CL");

  clLights = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MAX_LIGHTS*sizeof(LyapCamLight), Ls, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for lights");

  clCam = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(LyapCamLight), curC, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for cam");

  clPrm = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(LyapParams), curP, &cl.err);
  CL_ASSERT_RET(&cl, "Failed to get buffer for prm");

  return cl.err;
}

static int calcrendCLSetup()
{
  int arg = 4; // Skip buffers and offset

  cl.err = clEnqueueWriteBuffer(cl.queue, clCam, CL_FALSE, 0, sizeof(LyapCamLight), curC, 0, NULL, NULL);
  CL_ASSERT_RET(&cl, "Failed to update cam buffer");

  cl.err = clEnqueueWriteBuffer(cl.queue, clPrm, CL_FALSE, 0, sizeof(LyapParams), curP, 0, NULL, NULL);
  CL_ASSERT_RET(&cl, "Failed to update prm buffer");

  if(update_lights) {
    cl.err = clEnqueueWriteBuffer(cl.queue, clLights, CL_FALSE, 0, numLights*sizeof(LyapCamLight), Ls, 0, NULL, NULL);
    CL_ASSERT_RET(&cl, "Failed to update lights buffer");
    update_lights = 0;
  }

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], arg++, sizeof(cl_mem), (void *)&clCam);
  CL_ASSERT_RET(&cl, "Could not set parameter cam");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], arg++, sizeof(cl_mem), (void *)&clPrm);
  CL_ASSERT_RET(&cl, "Could not set parameter prm");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], arg++, sizeof(cl_mem), (void *)&clSeq);
  CL_ASSERT_RET(&cl, "Could not set parameter sequence");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], arg++, sizeof(cl_mem), (void *)&clLights);
  CL_ASSERT_RET(&cl, "Could not set parameter Ls");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], arg++, sizeof(cl_uint), (void *)&numLights);
  CL_ASSERT_RET(&cl, "Could not set parameter numLights");

  return cl.err;
}

static int calcrendCL()
{
  int err = CL_SUCCESS;

  if(update_calc)
    calcrendCLSetup();

  cl.err = clEnqueueAcquireGLObjects(cl.queue, 1, &clImage, 0, NULL, &cl.events[LYAP_ACQGL_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to acquire GL image");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], 0, sizeof(cl_mem), (void *)&clBuffer);
  CL_ASSERT_RET(&cl, "Could not set parameter clBuffer");

  cl.err = clSetKernelArg(cl.kernels[calcrendKernel], 1, sizeof(cl_mem), (void *)&clImage);
  CL_ASSERT_RET(&cl, "Could not set parameter clImage");

  cl_uint done = 0;
  cl_uint base = 0;
  size_t numResults = curC->renderWidth * curC->renderHeight;
  size_t blockSize = floorPow2(numResults);
  size_t localSize = cl.workGroupSize[calcrendKernel];
  int needToSubdivide = localSize > 1;

  unsigned int maxBlockSize = 4096;
  if(blockSize < localSize) localSize = blockSize;
  if(blockSize > maxBlockSize) blockSize = maxBlockSize;
  do {
    cl.err = clSetKernelArg(cl.kernels[calcrendKernel], 2, sizeof(done), (void *)&done);
    CL_ASSERT_RET(&cl, "Failed to set kernel parameter 2 (start)");

    cl.err = clSetKernelArg(cl.kernels[calcrendKernel], 3, sizeof(base), (void *)&base);
    CL_ASSERT_RET(&cl, "Failed to set kernel parameter 2 (base)");

    cl.err = clEnqueueNDRangeKernel(cl.queue, cl.kernels[calcrendKernel], 1, NULL, &blockSize, &localSize, 1, &cl.events[LYAP_ACQGL_EVENT], NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue kernel");

    done += blockSize;
    if(done < numResults) {
      size_t remainder = numResults - done;
      if(remainder > cl.workGroupSize[calcrendKernel]) {
        localSize = cl.workGroupSize[calcrendKernel];
        blockSize = floorPow2(remainder);
      }
      else {
        localSize = floorPow2(remainder);
        blockSize = localSize;
      }
      if(blockSize > maxBlockSize) blockSize = maxBlockSize;
    }
  } while(needToSubdivide && done < numResults);

  //  cl.err = clReleaseEvent(cl.events[LYAP_CALC_EVENT]);
  //  CL_ASSERT_RET(&cl, "Failed to release event");

  cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_REND_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to enqueue marker");

  cl.err = clEnqueueReleaseGLObjects(cl.queue, 1, &clImage, 1, &cl.events[LYAP_REND_EVENT], &cl.events[LYAP_RELGL_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to release GL image");

  cl.err = clWaitForEvents(1, &cl.events[LYAP_RELGL_EVENT]);

  update_calc = 0;
  update_render = 0;

  return err;
}

static int renderCLSetup()
{
  int arg = 2; // Skip buffer(s)

  cl_uint start = 0;
  cl_uint base = 0;

  if(update_lights) {
    cl.err = clEnqueueWriteBuffer(cl.queue, clLights, CL_FALSE, 0, numLights*sizeof(LyapCamLight), Ls, 0, NULL, NULL);
    CL_ASSERT_RET(&cl, "Failed to update lights buffer");
    update_lights = 0;
  }

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(start), (void *)&start);
  CL_ASSERT_RET(&cl, "Could not set parameter start");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(base), (void *)&base);
  CL_ASSERT_RET(&cl, "Could not set parameter base");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(cl_mem), (void *)&clCam);
  CL_ASSERT_RET(&cl, "Could not set parameter cam");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(cl_mem), (void *)&clLights);
  CL_ASSERT_RET(&cl, "Could not set parameter Ls");

  cl.err = clSetKernelArg(cl.kernels[rendKernel], arg++, sizeof(numLights), (void *)&numLights);
  CL_ASSERT_RET(&cl, "Could not set parameter numLights");

  return cl.err;
}

static int renderCL()
{
  int err = CL_SUCCESS;

  if(update_render)
    renderCLSetup();

  cl.err = clEnqueueAcquireGLObjects(cl.queue, 1, &clImage, 0, &cl.events[LYAP_CALC_EVENT], &cl.events[LYAP_ACQGL_EVENT]);

  CL_ASSERT_RET(&cl, "Failed to acquire GL image");

  cl_uint done = 0;
  cl_uint base = 0;
  size_t numResults = curC->renderWidth * curC->renderHeight;
  size_t blockSize = floorPow2(numResults);
  size_t localSize = cl.workGroupSize[rendKernel];
  int needToSubdivide = localSize > 1;

  if(blockSize < localSize) localSize = blockSize;
  do {
    cl.err = clSetKernelArg(cl.kernels[rendKernel], 0, sizeof(cl_mem), (void *)&clBuffer);
    CL_ASSERT_RET(&cl, "Could not set parameter clBuffer");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 1, sizeof(cl_mem), (void *)&clImage);
    CL_ASSERT_RET(&cl, "Could not set parameter clImage");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 2, sizeof(done), (void *)&done);
    CL_ASSERT_RET(&cl, "Failed to set parameter 2 (offset)");

    cl.err = clSetKernelArg(cl.kernels[rendKernel], 2, sizeof(base), (void *)&base);
    CL_ASSERT_RET(&cl, "Failed to set parameter 3 (base)");

    cl.err = clEnqueueNDRangeKernel(cl.queue, cl.kernels[rendKernel], 1, NULL, &blockSize, &localSize, 1, &cl.events[LYAP_ACQGL_EVENT], NULL);
    CL_ASSERT_RET(&cl, "Failed to enqueue kernel");

    done += blockSize;
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

  cl.err = clEnqueueMarker(cl.queue, &cl.events[LYAP_REND_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to enqueue marker");

  cl.err = clEnqueueReleaseGLObjects(cl.queue, 1, &clImage, 1, &cl.events[LYAP_REND_EVENT], &cl.events[LYAP_RELGL_EVENT]);
  CL_ASSERT_RET(&cl, "Failed to release GL image");

  cl.err = clWaitForEvents(1, &cl.events[LYAP_RELGL_EVENT]);

  clReleaseEvent(cl.events[LYAP_REND_EVENT]);
  clReleaseEvent(cl.events[LYAP_ACQGL_EVENT]);

  cl.events[LYAP_REND_EVENT] = 0;
  cl.events[LYAP_ACQGL_EVENT] = 0;

  update_render = 0;

  return err;
}

static int renderGL()
{
  static const GLfloat verts[] = {-1.0f, 1.0f, 0,
                                  -1.0f, -1.0f, 0,
                                  1.0f, 1.0f, 0,
                                  1.0f, -1.0f, 0};

  static GLfloat uvs[] = {0,0, 0,1, 1,0, 1,1};

  uvs[4] = uvs[6] = curC->renderWidth/(float)windowWidth;
  uvs[3] = uvs[7] = curC->renderHeight/(float)windowHeight;

  glColor4f(1.0, 1.0, 1.0, 1.0);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glVertexPointer(3, GL_FLOAT, sizeof(GLfloat)*3, verts);
  glTexCoordPointer(2, GL_FLOAT, sizeof(GLfloat)*2, uvs);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);

  return CL_SUCCESS;
}

static void glut_display(void)
{
  int err = CL_SUCCESS;

  curC->textureWidth = windowWidth;
  curC->textureHeight = windowHeight;
  curC->renderDenominator = renderDenominator;

  if(update_calc) {
    if((err = calcrendCL())) {
      fprintf(stderr, "calcrenderCL(): Error %d\n", err);
      exit(1);
    }
    update_calc = update_render = 0;
  }
  else if(update_render) {
    if((err = renderCL())) {
      fprintf(stderr, "renderCL(): Error %d\n", err);
      exit(1);
    }
    update_render = 0;
  }

  if((err = renderGL())) {
    fprintf(stderr, "renderGL(): Error %d\n", err);
    exit(1);
  }

  glFinish();
}

static void glut_idle(void)
{
  if(update_render || update_calc)
    glutPostRedisplay();
}

static enum {
  MODE_LOCK,
  MODE_ROTPAN,
  MODE_MAG,
  MODE_NTHRESH,
  MODE_OTHRESH,
  MODE_CTHRESH,
  MODE_D,
  MODE_JITTER,
  MODE_L_PAN,
  MODE_L_RANGE,
  MODE_L_COLOR
} controlMode = MODE_ROTPAN;

static int dominant = 0;

static void glut_spaceball_but(int button, int state)
{
  if(state != GLUT_UP) return;

  switch (button) {
  case 1:
    fprintf(stderr, "DOM%d\n", dominant = !dominant);
    break;

  default:
    fprintf(stderr, "B%d\n", button);
  }
}

static void glut_spaceball_rot(int x, int y, int z)
{
  if(controlMode==MODE_ROTPAN) {
    REAL4 axis = REAL4_spaceball_soften(x, -y, z, 50, 500, 1);

    REAL len = REAL4_mag(axis);

    if(len<QUAT_EPSILON) return;

    axis = REAL4_scale(axis, 1/len); // a.k.a Normalize

    if(dominant) axis = REAL4_dominant(axis);

    QUAT rQ = QUAT_fromAxisAngle(axis.x, axis.y, axis.z, len*0.05, 0);
    curC->Q = QUAT_normalize(QUAT_multiply(curC->Q, rQ));
    camCalculate(curC, windowWidth, windowHeight, renderDenominator);

    update_calc = 1;
  }
  else if(controlMode==MODE_MAG) {
    curC->M += (float)x/2500.0;
    camCalculate(curC, windowWidth, windowHeight, renderDenominator);

    printf("M:%.4f\n",curC->M);
    update_calc = 1;
  }
  else if(controlMode==MODE_NTHRESH) {
    curP->nearThreshold += (float)x/10000.0;
    if(curP->opaqueThreshold > curP->nearThreshold)
      curP->nearThreshold = curP->opaqueThreshold;
    update_calc = 1;
    printf("Thresh:%.4f - %.4f\n",curP->nearThreshold, curP->opaqueThreshold);
  }
  else if(controlMode==MODE_OTHRESH) {
    curP->opaqueThreshold += (float)x/10000.0;
    if(curP->opaqueThreshold > curP->nearThreshold)
      curP->opaqueThreshold = curP->nearThreshold;
    update_calc = 1;
    printf("Thresh:%.4f - %.4f\n",curP->nearThreshold, curP->opaqueThreshold);
  }
  else if(controlMode==MODE_CTHRESH) {
    curP->chaosThreshold += (float)x/10000.0;
    update_calc = 1;
  }
  else if(controlMode==MODE_D) {
    curP->d += (float)x/10000.0;

    if(curP->d<curP->lMin)
      curP->d=curP->lMin;
    else if(curP->d>curP->lMax)
      curP->d = curP->lMax;

    printf("d:%.2f\n", curP->d);
    update_calc = 1;
  }
  else if(controlMode==MODE_JITTER) {

    curP->jitter += (float)x/10000.0;

    if(curP->jitter<0)
      curP->jitter=0;
    else if(curP->jitter>1.0)
      curP->jitter=1.0;

    printf("j:%.2f\n", curP->jitter);
    update_calc = 1;
  }
  else if(controlMode==MODE_L_RANGE) {
    if(curL+1 > numLights)
      setupLight(curL);

    Ls[curL].lightRange += (float)x/2500.0;
    if(Ls[curL].lightRange<0)
      Ls[curL].lightRange=0;

    camCalculate(&Ls[curL], windowWidth, windowHeight, renderDenominator);
    update_lights = 1;
    update_render = 1;
  }
}

static void glut_spaceball_pan(int x, int y, int z)
{
  if(controlMode==MODE_ROTPAN) {
    REAL4 delta = REAL4_spaceball_soften(x, -y, z, 50, 500, 0.05);

    if(dominant) delta = REAL4_dominant(delta);

    delta = QUAT_transformREAL4(delta, curC->Q);

    curC->C = REAL4_add(curC->C, delta);

    printf("C:%.4f,%.4f,%.4f\tQ:%.4f,%.4f,%.4f,%.4f\n",
           curC->C.x, curC->C.y, curC->C.z,
           curC->Q.x, curC->Q.y, curC->Q.z, curC->Q.w);

    update_lights = 1;
    update_calc = 1;
  }
  else if(controlMode==MODE_L_PAN) {
    if(curL+1 > numLights) setupLight(curL);

    REAL4 delta = REAL4_spaceball_soften(x, -y, z, 50, 500, 0.25);
    if(dominant) delta = REAL4_dominant(delta);

    delta = QUAT_transformREAL4(delta, curC->Q);

    Ls[curL].C = REAL4_add(Ls[curL].C, delta);
    printf("L%d:%.4f,%.4f,%.4f\n", curL,
           Ls[curL].C.x,
           Ls[curL].C.y,
           Ls[curL].C.z);

    camCalculate(&Ls[curL], windowWidth, windowHeight, renderDenominator);

    update_lights = 1;
    update_render = 1;
  }
  else if(controlMode==MODE_L_COLOR) {
    if(curL+1 > numLights)
      setupLight(curL);

    REAL4 delta = REAL4_spaceball_soften(x, -y, z, 50, 2500, 0.25);
    if(dominant) delta = REAL4_dominant(delta);

    Ls[curL].diffuseColor.x += delta.x;
    Ls[curL].diffuseColor.y += delta.y;
    Ls[curL].diffuseColor.z += delta.z;

    if(Ls[curL].diffuseColor.x<0) Ls[curL].diffuseColor.x = 0;
    else if(Ls[curL].diffuseColor.x>1) Ls[curL].diffuseColor.x = 1;
    if(Ls[curL].diffuseColor.y<0) Ls[curL].diffuseColor.y = 0;
    else if(Ls[curL].diffuseColor.y>1) Ls[curL].diffuseColor.y = 1;
    if(Ls[curL].diffuseColor.z<0) Ls[curL].diffuseColor.z = 0;
    else if(Ls[curL].diffuseColor.z>1) Ls[curL].diffuseColor.z = 1;
    Ls[curL].specularColor = Ls[curL].diffuseColor;
    update_lights = 1;
    update_render = 1;
  }
}

static void glut_keyboard(unsigned char k,
                          int x __attribute__ ((unused)),
                          int y __attribute__ ((unused)))
{
  switch (k) {
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    curL = k-'1';
    curC = &Ls[curL];
    camCalculate(curC, windowWidth, windowHeight, renderDenominator);
    controlMode = MODE_ROTPAN;
    update_render = 1;
    update_lights = 1;
    break;

  case '0':
    curC = &cam;
    camCalculate(curC, windowWidth, windowHeight, renderDenominator);
    controlMode = MODE_ROTPAN;
    update_calc = 1;
    break;

  case 'l':
  case 'L':
    controlMode = MODE_L_PAN;
    update_lights = 1;
    break;

  case 'r':
  case 'R':
    controlMode = MODE_L_RANGE;
    update_lights = 1;
    break;

  case 'o':
  case 'O':
    controlMode = MODE_L_COLOR;
    update_lights = 1;
    break;

  case 'W': case 'w': save(); break;

  case 'd': case 'D': controlMode = MODE_D; break;
  case 'j': case 'J': controlMode = MODE_JITTER; break;
  case 't': case 'T': controlMode = MODE_OTHRESH; break;
  case 'y': case 'Y': controlMode = MODE_NTHRESH; break;
  case 'c': case 'C': controlMode = MODE_CTHRESH; break;
  case 'm': case 'M': controlMode = MODE_MAG; break;
  case '[': case '{': curP->stepMethod = 1; update_calc = 1; break;
  case ']': case '}': curP->stepMethod = 2; update_calc = 1; break;

  case 'z':
    curP->depth -= 1;
    if(curP->depth<1.0)
      curP->depth = 1.0;
    printf("depth=%.0f\n",(float)(int)curP->depth);
    update_calc = 1;
    break;

  case 'Z':
    curP->depth /= 2;
    if(curP->depth<1.0)
      curP->depth = 1.0;
    printf("depth=%.0f\n",(float)(int)curP->depth);
    update_calc = 1;
    break;

  case 'a':
    curP->depth += 1;
    printf("depth=%.0f\n",(float)(int)curP->depth);
    update_calc = 1;
    break;

  case 'A':
    curP->depth *= 2;
    printf("depth=%.0f\n",(float)(int)curP->depth);
    update_calc = 1;
    break;

  case 'h':
  case 'H':
    Ls[curL] = L0;
    Ls[curL].C = cam.C;
    Ls[curL].Q = cam.Q;
    Ls[curL].M = cam.M;
    camCalculate(&Ls[curL], windowWidth, windowHeight, renderDenominator);
    update_lights = 1;
    update_render = 1;
    break;

  case 'x':
    curP->accum --;
    if(curP->accum<1.0)
      curP->accum = 1.0;
    printf("accum=%d\n",curP->accum);
    update_calc = 1;
    break;

  case 'X':
    curP->accum /= 2;
    if(curP->accum<1.0)
      curP->accum = 1.0;
    printf("accum=%d\n",curP->accum);
    update_calc = 1;
    break;

  case 's':
    curP->accum ++;
    printf("accum=%d\n",curP->accum);
    update_calc = 1;
    break;

  case 'S':
    curP->accum *= 2;
    printf("accum=%d\n",curP->accum);
    update_calc = 1;
    break;

  case 'q':
  case 'Q':
    exit(0);

  case '/':
  case '?':
    dumpLyapHeader(1);
    break;

  case '-':
  case '_':
    renderDenominator ++;
    camCalculate(curC, windowWidth, windowHeight, renderDenominator);
    update_calc = 1;
    break;

  case '+':
  case '=':
    if(renderDenominator>1) {
      renderDenominator --;
      camCalculate(curC, windowWidth, windowHeight, renderDenominator);
      update_calc = 1;
    }
    else
      renderDenominator = 1;
    break;

  case ' ': controlMode = MODE_ROTPAN; break;
  }
}

int main(int argc, char** argv)
{
#include "params.h"
#ifdef SETTLE
  curP->settle = SETTLE;
#endif
#ifdef ACCUM
  curP->accum = ACCUM;
#endif
#ifdef DEPTH
  curP->depth = DEPTH;
#endif

  camCalculate(&cam, windowWidth, windowHeight, renderDenominator);

  size_t i;
  for(i=0; i<numLights; i++)
    camCalculate(&Ls[i], windowWidth, windowHeight, renderDenominator);

  L0 = Ls[0];

  /*
  REAL4 z = {0,0,1,1};
  REAL4P_normalize_inplace(&z);

  REAL4 la = {2,2,2,1};
  REAL4 c = {4,4,4,1};
  REAL4 v = REAL4_sub(la, c);//{-1,-1,-1,1};
  REAL4P_normalize_inplace(&v);

  QUAT q = QUAT_fromVectors(z, v);

  REAL4 u = {0,1,0,1};
  REAL4 n = QUAT_transformREAL4(u, q);

  REAL4 r = REAL4_cross(v, u);
  REAL4P_normalize_inplace(&r);

  QUAT q2 = QUAT_fromVectors(n, REAL4_cross(v, r));
  QUAT q3 = QUAT_multiply(q2, q);
  QUAT q4 = QUAT_fromAxisAngle(0, 0, 1, 60, 1);

  curC->Q = QUAT_multiply(q3, q4);

  save();
  return 0;
*/

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
  glutInitWindowSize(windowWidth, windowHeight);

  window = glutCreateWindow("Lyapunov3D");

  setupGL();
  setupCL();

  //  glut_reshape(windowWidth, windowHeight);

  //#if !(defined (__APPLE__) || defined(MACOSX))
  //  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  //#endif

  glutDisplayFunc(glut_display);
  glutIdleFunc(glut_idle);
  glutSpaceballRotateFunc(glut_spaceball_rot);
  glutSpaceballMotionFunc(glut_spaceball_pan);
  glutSpaceballButtonFunc(glut_spaceball_but);
  glutKeyboardFunc(glut_keyboard);

  //  glutMouseFunc(mouseGL);
  //  glutMotionFunc(motionGL);
  //  glutTimerFunc(REFRESH_DELAY, timerEvent,0);

  glutMainLoop();

  return 0;
}
