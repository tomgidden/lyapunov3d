#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes from the CUDA SDK samples
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_gl.h>

#include "lyap.hpp"
#include "scene.hpp"
#include "params.hpp"

// Image and grid parameters
unsigned int imageWidth = 256, imageHeight = 256;
const unsigned int blockSize = 16;
const dim3 blocks(imageWidth / blockSize, imageHeight / blockSize);
const dim3 threads(blockSize, blockSize);
unsigned int windowWidth = imageWidth, windowHeight = imageHeight;
unsigned int renderDenominator = 1;

// Control
static bool dominant = false;

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

static bool update_calc = true;
static bool update_render = false;
static bool update_lights = true;
static bool update_cam = true;


LyapParams *curP = &prm;
LyapCam *curC = &cam;
unsigned int curL = 0;


// Data transfer of Pixel Buffer Object between CUDA and GL
GLuint glPBO;
struct cudaGraphicsResource *cudaPBO;
// A simple semaphore to indicate whether the PBO has been mapped or not
volatile int cudaPBO_map_count = 0;

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif

// Device array of LyapPoint points
LyapPoint *cudaPoints = 0;

// Device array of lights
LyapLight *cudaLights = 0;

// Device pixel buffer
RGBA *cudaRGBA;

// Device sequence array
Int *cudaSeq;



void quit()
{
#if defined(__APPLE__) || defined(MACOSX)
    glutDestroyWindow(glutGetWindow());
    exit(EXIT_SUCCESS);
    return;
#else
    glutDestroyWindow(glutGetWindow());
    return;
#endif
}

void cuda_load_sequence(unsigned char *seqStr)
{
    size_t actual_length;
    Int *seq;

    actual_length = scene_convert_sequence(&seq, seqStr);

    checkCudaErrors(cudaMalloc(&cudaSeq, actual_length * sizeof(Int)));
    checkCudaErrors(cudaMemcpy(cudaSeq, seq, actual_length * sizeof(Int), cudaMemcpyHostToDevice));

    free(seq);
}

void init_scene()
{
    params_init();

    checkCudaErrors(cudaMalloc(&cudaLights, sizeof(LyapLight) * MAX_LIGHTS));
    checkCudaErrors(cudaMalloc(&cudaPoints, sizeof(LyapPoint) * imageWidth * imageHeight));

    cuda_load_sequence(sequence);

    update_lights = true;
    update_cam = true;
}

void update_scene()
{
    if (update_lights) {
        scene_lights_recalculate(lights, num_lights);

        // Load lights into device memory
        checkCudaErrors(cudaMemcpy(cudaLights, lights, sizeof(LyapLight) * num_lights, cudaMemcpyHostToDevice));
    }

    if (update_cam) {
        scene_cam_recalculate(&cam, windowWidth, windowHeight, renderDenominator);
    }
}


/**
 * Perform render step in CUDA, and write results to PBO
 */
void render()
{
    size_t num_bytes;

    // Map PBO to get CUDA device pointer
    cudaPBO_map_count++;
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBO, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cudaRGBA, &num_bytes, cudaPBO));

    // call CUDA kernel, writing results to PBO
    //    for(int i = 0; i < passes; ++i) {
    //        void *dummy;
    kernel_calc_render<<<blocks, threads>>>(cudaRGBA, cudaPoints, cam, prm, cudaSeq, cudaLights, num_lights);
    //        cudaMemcpyAsync(dummy, dummy, 1, cudaMemcpyDeviceToDevice);
    //    }

    if (false) {
        int points_size = sizeof(LyapPoint) * imageWidth * imageHeight;
        printf("Points size = %d\n", points_size);

        LyapPoint *myPoints = (LyapPoint *)malloc(points_size);
        printf("malloc'ed %p.\n", myPoints);

        checkCudaErrors(cudaMemcpy( myPoints, cudaPoints, points_size, cudaMemcpyDeviceToHost ));

        LyapPoint *ptr = myPoints;
        for (int i=0; i<imageWidth * imageHeight; i++, ptr++)
            printf("%d:\t%f,%f,%f\t%f,%f,%f\n",
                   points_size,
                   ptr->P.x, ptr->P.y, ptr->P.z,
                   ptr->N.x, ptr->N.y, ptr->N.z);

        free(myPoints);
        quit();
    }

    // Handle error
    getLastCudaError("kernel render failed");

    // Unmap cleanly
    if (cudaPBO_map_count) {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPBO, 0));
        cudaPBO_map_count--;
    }
}

/**
 * Display the rendered data using GL.  This function is called automatically by GLUT
 */
void display()
{
    // Perform any necessary updates
    update_scene();

    // Render the data
    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glPBO);
    glDrawPixels(imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Output the image to the screen
    glutSwapBuffers();

    glutReportErrors();
}

/**
 * Idle loop.
 */
void idle()
{
    if(update_render || update_calc || update_lights || update_cam)
        glutPostRedisplay();
}

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
        Vec axis = Vec(x, -y, z);
        axis.spaceball_soften(50, 500, 1);

        Real len = axis.mag();

        if (len < 1e-6) return;

        axis /= len; // a.k.a Normalize

        if(dominant) axis.dominantize();

        Quat rQ = Quat(axis, len*0.05, false);

        curC->Q = (curC->Q * rQ).normalized();

        update_cam = true;
        update_calc = true;
    }
    else if(controlMode==MODE_MAG) {
        curC->M += (float)x/2500.0;
        printf("M:%.4f\n",curC->M);

        update_cam = true;
        update_calc = true;
    }
    else if(controlMode==MODE_NTHRESH) {
        curP->nearThreshold += (float)x/10000.0;
        if(curP->opaqueThreshold > curP->nearThreshold)
            curP->nearThreshold = curP->opaqueThreshold;
        printf("Thresh:%.4f - %.4f\n",curP->nearThreshold, curP->opaqueThreshold);

        update_calc = true;
    }
    else if(controlMode==MODE_OTHRESH) {
        curP->opaqueThreshold += (float)x/10000.0;
        if(curP->opaqueThreshold > curP->nearThreshold)
            curP->opaqueThreshold = curP->nearThreshold;
        printf("Thresh:%.4f - %.4f\n",curP->nearThreshold, curP->opaqueThreshold);

        update_calc = true;
    }
    else if(controlMode==MODE_CTHRESH) {
        curP->chaosThreshold += (float)x/10000.0;

        update_calc = true;
    }
    else if(controlMode==MODE_D) {
        curP->d += (float)x/10000.0;

        if(curP->d<curP->lMin)
            curP->d = curP->lMin;

        else if(curP->d>curP->lMax)
            curP->d = curP->lMax;

        printf("d:%.2f\n", curP->d);

        update_calc = true;
    }
    else if(controlMode==MODE_JITTER) {

        curP->jitter += (float)x/10000.0;

        if(curP->jitter<0)
            curP->jitter=0;

        else if(curP->jitter>1.0)
            curP->jitter=1.0;

        printf("j:%.2f\n", curP->jitter);

        update_calc = true;
    }
    // else if(controlMode==MODE_L_RANGE) {
    //     if(curL+1 > num_lights)
    //         setupLight(curL);

    //     lights[curL].lightRange += (float)x/2500.0;
    //     if(lights[curL].lightRange<0)
    //         lights[curL].lightRange=0;

    //     scene_cam_recalculate(&lights[curL], windowWidth, windowHeight, renderDenominator);
    //     update_lights = true;
    //   update_render = true;
    // }
}

static void glut_spaceball_pan(int x, int y, int z)
{
    if(controlMode==MODE_ROTPAN) {
        Vec delta = Vec(x, -y, z);
        delta.spaceball_soften(50, 500, 0.05);

        if(dominant) delta.dominantize();

        delta = curC->Q.transform(delta);

        curC->C += delta;

        printf("C:%.4f,%.4f,%.4f\tQ:%.4f,%.4f,%.4f,%.4f\n",
               curC->C.x, curC->C.y, curC->C.z,
               curC->Q.x, curC->Q.y, curC->Q.z, curC->Q.w);

        update_cam = true;
        update_lights = true;
        update_calc = true;
    }
    else if(controlMode==MODE_L_PAN) {
        if(curL+1 > num_lights) return;//setupLight(curL);

        Vec delta = Vec(x, -y, z);
        delta.spaceball_soften(50, 500, 0.25);

        if(dominant) delta.dominantize();

        delta = curC->Q.transform(delta);

        lights[curL].C += delta;

        printf("L%d:%.4f,%.4f,%.4f\n", curL,
               lights[curL].C.x,
               lights[curL].C.y,
               lights[curL].C.z);

        update_cam = true;
        update_lights = true;
        update_render = true;
    }
    else if(controlMode==MODE_L_COLOR) {
        if(curL+1 > num_lights) return;//setupLight(curL);

        Vec delta = Vec(x, -y, z);
        delta.spaceball_soften(50, 500, 0.25);

        if(dominant) delta.dominantize();

        lights[curL].diffuseColor.x += delta.x;
        lights[curL].diffuseColor.y += delta.y;
        lights[curL].diffuseColor.z += delta.z;

        lights[curL].diffuseColor.x = Vec::clamp(lights[curL].diffuseColor.x);
        lights[curL].diffuseColor.y = Vec::clamp(lights[curL].diffuseColor.y);
        lights[curL].diffuseColor.z = Vec::clamp(lights[curL].diffuseColor.z);

        lights[curL].specularColor = lights[curL].diffuseColor;

        update_lights = true;
        update_render = true;
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
        curC = &lights[curL];
        controlMode = MODE_ROTPAN;
        update_cam = true;
        update_render = true;
        update_lights = true;
        break;

    case '0':
        curC = &cam;
        controlMode = MODE_ROTPAN;
        update_cam = true;
        update_calc = true;
        break;

    case 'l':
    case 'L':
        controlMode = MODE_L_PAN;
        update_lights = true;
        break;

    case 'r':
    case 'R':
        controlMode = MODE_L_RANGE;
        update_lights = true;
        break;

    case 'o':
    case 'O':
        controlMode = MODE_L_COLOR;
        update_lights = true;
        break;

        //case 'W': case 'w': save(); break;

    case 'd': case 'D': controlMode = MODE_D; break;
    case 'j': case 'J': controlMode = MODE_JITTER; break;
    case 't': case 'T': controlMode = MODE_OTHRESH; break;
    case 'y': case 'Y': controlMode = MODE_NTHRESH; break;
    case 'c': case 'C': controlMode = MODE_CTHRESH; break;
    case 'm': case 'M': controlMode = MODE_MAG; break;
    case '[': case '{': curP->stepMethod = 1; update_calc = true; break;
    case ']': case '}': curP->stepMethod = 2; update_calc = true; break;

    case 'z':
        curP->depth -= 1;
        if(curP->depth<1.0)
            curP->depth = 1.0;
        printf("depth=%.0f\n",(float)(int)curP->depth);
        update_calc = true;
        break;

    case 'Z':
        curP->depth /= 2;
        if(curP->depth<1.0)
            curP->depth = 1.0;
        printf("depth=%.0f\n",(float)(int)curP->depth);
        update_calc = true;
        break;

    case 'a':
        curP->depth += 1;
        printf("depth=%.0f\n",(float)(int)curP->depth);
        update_calc = true;
        break;

    case 'A':
        curP->depth *= 2;
        printf("depth=%.0f\n",(float)(int)curP->depth);
        update_calc = true;
        break;

    case 'h':
    case 'H':
        lights[curL] = L0;
        lights[curL].C = cam.C;
        lights[curL].Q = cam.Q;
        lights[curL].M = cam.M;
        scene_cam_recalculate(&lights[curL], windowWidth, windowHeight, renderDenominator);
        update_lights = true;
        update_render = true;
        break;

    case 'x':
        curP->accum --;
        if(curP->accum<1.0)
            curP->accum = 1.0;
        printf("accum=%d\n",curP->accum);
        update_calc = true;
        break;

    case 'X':
        curP->accum /= 2;
        if(curP->accum<1.0)
            curP->accum = 1.0;
        printf("accum=%d\n",curP->accum);
        update_calc = true;
        break;

    case 's':
        curP->accum ++;
        printf("accum=%d\n",curP->accum);
        update_calc = true;
        break;

    case 'S':
        curP->accum *= 2;
        printf("accum=%d\n",curP->accum);
        update_calc = true;
        break;

    case 'q':
    case 'Q':
        exit(0);

    case '/':
    case '?':
        //        dumpLyapHeader(1);
        break;

    case '-':
    case '_':
        renderDenominator ++;
        update_calc = true;
        break;

    case '+':
    case '=':
        if(renderDenominator>1) {
            renderDenominator --;
            update_calc = true;
        }
        else
            renderDenominator = 1;
        break;

    case ' ': controlMode = MODE_ROTPAN; break;
    }
}

/**
 * Resize the window.  We're not going to do anything clever here for the
 * time being, eg. resize the buffers.
 */
void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    checkCudaErrors(cudaFree(cudaLights));
    checkCudaErrors(cudaFree(cudaPoints));
    checkCudaErrors(cudaFree(cudaSeq));

    // Unmap the PBO if it has been mapped
    if (cudaPBO_map_count) {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPBO, 0));
        cudaPBO_map_count--;
    }

    // Unregister this buffer object from CUDA and from GL
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaPBO));
    glDeleteBuffers(1, &glPBO);
}

void init_gl(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

    glutInitWindowSize(windowWidth, windowHeight);

    glutCreateWindow(argv[0]);

    glutDisplayFunc(display);
    glutSpaceballRotateFunc(glut_spaceball_rot);
    glutSpaceballMotionFunc(glut_spaceball_pan);
    glutSpaceballButtonFunc(glut_spaceball_but);
    glutKeyboardFunc(glut_keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    if (!isGLVersionSupported(2,0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }
}

void init_gl_buffer(unsigned int _imageWidth, unsigned int _imageHeight)
{
    if (glPBO) {
        cudaGraphicsUnregisterResource(cudaPBO);
        glDeleteBuffers(1, &glPBO);
    };

    // Create pixel buffer object
    glGenBuffers(1, &glPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, _imageWidth * _imageHeight * sizeof(RGBA), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Register this buffer object with CUDA.  We only need to do this once.
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPBO, glPBO, cudaGraphicsMapFlagsWriteDiscard));
}

/**
 * Find CUDA device
 */
int choose_cuda_device(int argc, char **argv, bool use_gl)
{
    int result = 0;

    if (use_gl) {
        result = findCudaGLDevice(argc, (const char **)argv);
    }
    else {
        result = findCudaDevice(argc, (const char **)argv);
    }

    return result;
}

int main(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    // Use command-line specified CUDA device, otherwise use device with
    // highest Gflops/s
    choose_cuda_device(argc, argv, true);

    // First initialize OpenGL context, so we can properly set the GL for
    // CUDA.  This is necessary in order to achieve optimal performance
    // with OpenGL/CUDA interop.
    init_gl(&argc, argv);

    // Create GL/CUDA interop buffer
    init_gl_buffer(imageWidth, imageHeight);

    // Initialise scene
    init_scene();

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    glutMainLoop();

    exit(EXIT_SUCCESS);
}
