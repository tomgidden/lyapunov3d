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

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

// Image and grid parameters
const unsigned int imageWidth = 512, imageHeight = 512;
const unsigned int blockSize = 16;
const dim3 blocks(imageWidth / blockSize, imageHeight / blockSize);
const dim3 threads(blockSize, blockSize);
const unsigned int windowWidth = imageWidth, windowHeight = imageHeight;
const unsigned int renderDenominator = 1;

// Scene parameters
const unsigned int MAX_LIGHTS = 16;


LyapParams prm;
LyapCam cam;
LyapLight lights[MAX_LIGHTS];

unsigned char *sequence;

unsigned int num_lights;

void init_params()
{
    prm.d = 2.10;
    prm.settle = 10;
    prm.accum = 20;
    prm.stepMethod = 1;
    prm.nearThreshold = -1.0;
    prm.nearMultiplier = 2.0;
    prm.opaqueThreshold = -1.125;
    prm.chaosThreshold = 100000.0;
    prm.depth = 16.0;
    prm.jitter = 0.5;
    prm.refine = 32.0;
    prm.gradient = 0.01;
    prm.lMin = 0.0;
    prm.lMax = 4.0;

    sequence = (unsigned char *)"BCABA";

    cam.C = Vec(3.51f, 3.5f, 3.5f);
    cam.Q = Quat(0.820473f, -0.339851f, -0.175920f, 0.424708f);
    cam.M = 1;//.500000;

    lights[0].C = Vec(5.0f, 7.0f, 3.0f);
    lights[0].Q = Quat(0.710595f, 0.282082f, -0.512168f, 0.391368f);
    lights[0].M = 0.500000;
    lights[0].lightInnerCone = 0.904535f;
    lights[0].lightOuterCone = 0.816497f;
    lights[0].lightRange = 1.0;
    lights[0].ambient = Color(0, 0, 0, 0);
    lights[0].diffuseColor = Color(0.30, 0.40, 0.50, 1);
    lights[0].diffusePower = 10.0;
    lights[0].specularColor = Color(0.90, 0.90, 0.90, 1);
    lights[0].specularPower = 10.0;
    lights[0].specularHardness = 10.0;
    lights[0].chaosColor = Color(0, 0, 1, 0.1);

    lights[1].C = Vec(3, 7, 5);
    lights[1].Q = Quat(0.039640, 0.840027,-0.538582,-0.052093);
    lights[1].M = 1.6772;
    lights[1].lightInnerCone = 0.534489;
    lights[1].lightOuterCone = 0.388485;
    lights[1].lightRange = 0.5;
    lights[1].ambient = Color(0, 0, 0, 0);
    lights[1].diffuseColor = Color(0.3, 0.374694, 0.2, 1);
    lights[1].diffusePower = 10.0;
    lights[1].specularColor = Color(1, 1, 1, 1);
    lights[1].specularPower = 10.0;
    lights[1].specularHardness = 10.0;
    lights[1].chaosColor = Color(0, 0, 1, 1);

    num_lights = 2;
};

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

/**
 * Convert the lyapunov point into a pixel for rendering
 */
__device__ Color shade(LyapPoint point, LyapCam cam, LyapLight *lights, Uint num_lights)
{
    Color color = Color();

    if (isnan(point.a)) {
        color.w = 1;
        return color;
    }

    Uint l;

    // For each defined light
    for(l=0; l<num_lights; ++l) {
        Vec camV;
        Vec lightV, halfV;
        Color diffuse, specular, phong;
        Real lightD2, i, j;

        Vec P = point.P;
        Vec N = point.N;
        Real a = point.a;
        Real c = point.c;

        camV = cam.C - P;

        // Light vector (from point on surface to light source)
        lightV = lights[l].C - P;

        // Get the length^2 of lightV (for falloff)
        lightD2 = lightV.mag2();

        // but then normalize lightV.
        lightV.normalize();

        // i: light vector dot surface normal
        i = lightV % N;

        // j: light vector dot spotlight cone
        j = lightV % lights[l].V;
        j = -j;

        if (j > lights[l].lightOuterCone) {

            // Diffuse component: k * (L.N) * colour
            i = Vec::clamp(i);
            diffuse = lights[l].diffuseColor * (i*lights[l].diffusePower);

            // Halfway direction between camera and light, from point on surface
            halfV = camV + lightV;
            halfV.normalize();

            // Specular component: k * (R.N)^alpha * colour
            // R is natural reflection, which is at 90 degrees to halfV (?)
            // (or is it?  Hmmm.  https://en.wikipedia.org/wiki/Phong_reflection_model)
            i = Vec::clamp(N % halfV);
            i = powf(i, lights[l].specularHardness);

            specular = lights[l].specularColor * (i*lights[l].specularPower);

            phong = (specular + diffuse) * (lights[l].lightRange/lightD2);

            if ( j < lights[l].lightInnerCone) {
                phong *= ((j-lights[l].lightOuterCone)/(lights[l].lightInnerCone-lights[l].lightOuterCone));
            }
            phong += lights[l].ambient;
        }
        else {
            phong = lights[l].ambient;
        }

        if(c>0.0) {
            Color chaos;
            chaos = lights[l].chaosColor * (0.1125/Vec::logf(c));
            phong += chaos;
        }

        color += phong;
    }

    return color;
}

__device__ static Real lyap4d(Vec P, Real d, Uint settle, Uint accum, const Int *seq)
{
    Real abcd[4];
    abcd[0] = P.x;
    abcd[1] = P.y;
    abcd[2] = P.z;
    abcd[3] = d;

    //Real px=P.x-2.0, py=P.y-2.0, pz=P.z-2.0;
    //  if(SQRTF(px*px + py*py + pz*pz) < 2.0)
    //    return -1;
    //  else
    //    return 0;

    Uint seqi; // Position in the sequence loop
    Uint n;    // Iteration counter
    Real r; // Iteration value
    Real v = 0.5; // Iterating value
    Real l = 0; // Result accumulator

    // Initialise for this pixel
    seqi = 0;

    // Settle by running the iteration without accumulation
    for(n = 0; n < settle; n++) {
        r = abcd[seq[seqi++]];
        if(seq[seqi]==-1) seqi = 0;
        v = r * v * (1.0 - v);
    }

    if((v-0.5 <= -1e-4) || (v-0.5 >= 1e-4)) {
        // Now calculate the value by running the iteration with accumulation
        for(n = 0; n < accum; n++) {
            r = abcd[seq[seqi++]];
            if(seq[seqi]==-1) seqi = 0;
            v = r * v * (1.0 - v);
            r = r - 2.0 * r * v;
            r = fabs(r);
            l += Vec::logf(r);
            if(!isfinite(l)) { return NAN; }
        }
    }

    return l/(Real)accum;
}

__device__ static Int raycast(LyapPoint *point,
                              Uint sx, Uint sy,
                              LyapCam cam,
                              LyapParams prm,
                              Int *seq)
{
    // Work out the direction vector: start at C (camera), and
    // find the point on the screen plane (in 3D)
    Vec V = cam.S0 + cam.SDX * sx + cam.SDY * sy;

    V.normalize();
    V /= cam.M;

    Vec P;  // Point under consideration: will become the final hit point.
    Vec N;  // Surface normal at the hit point.

    Real a; // high-low alpha
    Real c; // chaos alpha

    Real l;

    bool near = false;
    Int i;
    Real t;

    //Real thresholdRange = prm.opaqueThreshold - prm.nearThreshold;

    // Find start and end point of ray within Lyapunov space cube

    Real t0=MAXFLOAT, t1=0;
    // Find ray intersection through entire Lyapunov space cube

    // First, find values for 't' for intersections with the six bounding
    // planes x=0, x=4, y=0, y=4, z=0, and z=4.  Any planes that are
    // parallel to the ray will meet the ray at INFINITY.
    Real ts[6];
    ts[0] = (V.x!=0.0f) ? ((LMIN-cam.C.x) / V.x) : INFINITY;
    ts[1] = (V.x!=0.0f) ? ((LMAX-cam.C.x) / V.x) : INFINITY;
    ts[2] = (V.y!=0.0f) ? ((LMIN-cam.C.y) / V.y) : INFINITY;
    ts[3] = (V.y!=0.0f) ? ((LMAX-cam.C.y) / V.y) : INFINITY;
    ts[4] = (V.z!=0.0f) ? ((LMIN-cam.C.z) / V.z) : INFINITY;
    ts[5] = (V.z!=0.0f) ? ((LMAX-cam.C.z) / V.z) : INFINITY;

    // Zero, one or two of these intersections will occur at a point
    // where all x, y and z are between 0 and 4 (ie. inside Lyapunov
    // space). Actually, all six might occur within Lyapunov space, but
    // only if the points are equal. So, zero, one or two _unique_
    // intersections will occur. This is because zero, one or two
    // points on a line intersect a cube.
    //
    // So, for each one, eliminate it if the intersection point lies
    // outside the bounds of the other two axes.
    if(ts[0] != INFINITY) {
        P = cam.C + V * ts[0];
        // If the x-min axis intersection is outside 0<=y<=4, 0<=z<=4, eliminate it.
        if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
            ts[0] = NAN;
    }

    if(ts[1] != INFINITY) {
        P = cam.C + V * ts[1];
        // If the x-max axis intersection is outside 0<=y<=4, 0<=z<=4, eliminate it.
        if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
            ts[1] = NAN;
    }

    if(ts[2] != INFINITY) {
        P = cam.C + V * ts[2];
        // If the y-min axis intersection is outside 0<=x<=4, 0<=z<=4, eliminate it.
        if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
            ts[2] = NAN;
    }

    if(ts[3] != INFINITY) {
        P = cam.C + V * ts[3];
        // If the y-max axis intersection is outside 0<=x<=4, 0<=z<=4, eliminate it.
        if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
            ts[3] = NAN;
    }

    if(ts[4] != INFINITY) {
        P = cam.C + V * ts[4];
        // If the z-min axis intersection is outside 0<=x<=4, 0<=y<=4, eliminate it.
        if(P.x<LMIN || P.x>LMAX || P.y<LMIN || P.y>LMAX)
            ts[4] = NAN;
    }

    if(ts[5] != INFINITY) {
        P = cam.C + V * ts[5];
        // If the z-max axis intersection is outside 0<=x<=4, 0<=y<=4, eliminate it.
        if(P.x<LMIN || P.x>LMAX || P.y<LMIN || P.y>LMAX)
            ts[5] = NAN;
    }

    // Find the smallest and largest finite 't' values for all the
    // intersections.  This identifies which of the bounding planes the
    // ray hits first and last.  The others can be ignored.
    Int i0=-1, i1=-1;
    for(i=0; i<6; i++) {
        if(isfinite(ts[i])) {
            if(i0==-1 || ts[i] < t0)
                t0 = ts[i0=i];

            if(i1==-1 || ts[i] > t1)
                t1 = ts[i1=i];
        }
    }

    // If both failed, then the ray didn't intersect Lyapunov space at
    // all, so exit: noise.
    if(i0==-1 && i1==-1) {
        return 1;
    }

    // If only one point matched, then the ray must(?) start in
    // Lyapunov space and exit it, so we can start at zero instead.
    else if(i1==-1 || i0==i1) {
        i1 = i0;
        t1 = t0;
        i0 = 0;
        t0 = 0;
    }

    // I'm not sure this is necessary, but just to make sure the
    // ray doesn't start behind the camera...
    if(t0 < 0) {
        t0 = 0;
    }

    // So, we start at t=t0
    t = t0;

    // Find P:  P = C + t.V
    P = cam.C + V * t;

    // Set the alpha accumulators to zero
    a = 0;
    c = 0;

    // dt is the amount to add to 't' for each step in the initial
    // ray progression.  We calculate Fdt for the normal value,
    // and Ndt for the finer value used when close to the threshold
    // (ie. under nearThreshold)
    Real dt, Ndt, Fdt;

    // There are different methods of progressing along the ray.

    switch (prm.stepMethod) {
    case 1:
        // Method 1 divides the distance between the start and the end of
        // the ray equally.
        Fdt = (t1-t0) / prm.depth;
        break;

    case 2:
    default:
        // Method 2 (default) divides the distance from the camera to the
        // virtual screen equally.
        Fdt = V.mag() / prm.depth;
    }

    dt = Fdt;
    Ndt = dt/prm.nearMultiplier;
    near = false;

    // Calculate the exponent at the current point.
    l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

    // Okay, now we do the initial ray progression: we trace until the
    // exponent for the point is below a certain value. This value is
    // effectively the transition between transparent and opaque.

    // While the exponent is above the surface threshold (ie. while the
    // current point is in "transparent" space)...
    while (l > prm.opaqueThreshold) {

        // Step along the ray by 'dt' plus/minus a certain amount of
        // jitter (optional). This reduces moire fringes and herringbones
        // resulting from transitioning through thin sheets. Instead we
        // get what looks like noise, but is in fact stochastic sampling
        // of a diaphanous transition membrane.

        if(prm.jitter != 0.0) {
            // We use the fractional part of the last Lyapunov exponent
            // as a pseudo-random number. This is then added to 'dt', scaled
            // by the amount of jitter requested.
            Real jit = l-trunc(l);
            if(jit<0)
                jit = 1.0 - jit*prm.jitter;
            else
                jit = 1.0 + jit*prm.jitter;

            if(isfinite(jit)) {
                t += dt*jit;
                P += V * (dt*jit);
            }
            else {
                t += dt;
                P += V * dt;
            }
        }
        else {
            // No jitter, so just add 'dt'.
            t += dt;
            P += V * dt;
        }

        // If the ray has exited Lyapunov space, then bugger it.
        if (t>t1 || !P.in_lyap_space()) {
            return 1;
        }

        // Calculate this point's exponent
        l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

        // If the ray is still in transparent space, then we may still
        // want to accumulate alpha for clouding.
        if (l > prm.chaosThreshold)
            c += l;

        if (l <= prm.nearThreshold && !near) {
            near = true;
            dt = Ndt;
        }
        else if (l > prm.nearThreshold && near) {
            near = false;
            dt = Fdt;
        }
    }

    // At this point, the ray has either hit an opaque point, or
    // has exited Lyapunov space.

    // If the ray has exited space, then this point is no longer
    // relevant.
    if (t>t1 || !P.in_lyap_space()) {
        return 1;
    }

    // Ray phase 2: now we've hit the surface, we now need to hone the
    // intersection point by reversing back along the ray at half the speed.

    // If we've gone through then sign is 0. 'sign' is
    // the direction of the progression.
    bool sign = 0;
    bool osign = sign;

    // Half speed
    Real Qdt = dt * -0.5f;
    Vec QdV = V * Qdt;

    // Set the range of the honing to <t-dt, t>.
    Real Qt1 = t;
    Real Qt0 = t-dt;

    // Honing continues reversing back and forth, halving speed
    // each time. Once dt is less than or equal to dt/refine,
    // we stop: it's close enough.
    Real min_Qdt = dt/prm.refine;

    // While 't' is still in the range <t-dt, t> AND dt is still
    // of significant size...
    while(t<=Qt1 && t>=Qt0 && (Qdt<=-min_Qdt || Qdt>=min_Qdt)) {

        // Progress along the ray
        t += Qdt;
        P += QdV;

        // Calculate the exponent
        l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

        // If we've hit the threshold exactly, short-circuit.
        if(l==prm.opaqueThreshold) break;

        // Work out whether we reverse or not:
        osign = sign;
        sign = (l < prm.opaqueThreshold) ? 0 : 1;

        // If we've reversed, then halve the speed
        if(sign != osign) {
            Qdt *= -0.5f;
            QdV *= -0.5f;
        }
    }

    // At this point, we should be practically on the surface, rather
    // than above or through. Anyway, we're close enough.  P is now
    // our hit point.

    // Next, we want to find the surface normal at P. A good approximation
    // is to get the vector gradient by calculating the Lyapunov exponent
    // at the six cardinal points surrounding P +/- a tiny amount, which
    // we assume to be small enough that the Lyapunov exponent approximates
    // to a linear function.
    //
    // Find the difference for each axis, and normalize. The result is
    // pretty close.
    //
    // If anyone can work out how to differentiate the Lyapunov exponent
    // as a vector, please do so: it'd be nice to avoid this approximation!
    // (Hang on... that's not possible, is it?)

    Real mag = dt * prm.gradient;
    Vec Ps[6] = {P,P,P,P,P,P};
    Ps[0].x -= mag;
    Ps[1].x += mag;
    Ps[2].y -= mag;
    Ps[3].y += mag;
    Ps[4].z -= mag;
    Ps[5].z += mag;

    Real ls[6];
    for(i=0; i<6; i++) {
        ls[i] = lyap4d(Ps[i], prm.d, prm.settle, prm.accum, seq);
    }

    N.x = ls[1]-ls[0];
    N.y = ls[3]-ls[2];
    N.z = ls[5]-ls[4];

    N.normalize();

    // Okay, we've done it. Output the hit point, the normal, the exact
    // exponent at P (not Really needed, but it does signal a hit failure
    // by l == NAN), and the accumulated alpha.

    point->P = P;
    point->N = N;
    point->a = a;
    point->c = c;

    return 0;
}

__global__ void kernel_calc_render(RGBA *rgba,
                                   LyapPoint *points,
                                   LyapCam cam,
                                   LyapParams prm,
                                   Int *seq,
                                   LyapLight *lights,
                                   Uint num_lights)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int ind = x + y * gridDim.x*blockDim.x;

    // Perform ray-casting (ie. non-bouncing ray-tracing; it's hard enough
    // as it is) to find the hit point for this pixel, accumulating data
    // into the point structure.
    Int ret = raycast(&(points[ind]), x, y, cam, prm, seq);

    // Convert the abstract point structure -- position, surface normal,
    // chaos, etc. -- into a colour, using the lights provided.
    Color color = shade(points[ind], cam, lights, num_lights);

    // Convert the floating-point colour into a 32-bit RGBA pixel
    color.to_rgba((unsigned char *)&rgba[ind]);
}


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

/**
 * Parse and load the sequence string into device memory
 */
void cuda_load_sequence(unsigned char *seqStr)
{
    Int *seq;
    size_t actual_length = 0;
    size_t estimated_length = 10 * strlen((const char *)seqStr) + 1;

    seq = (Int *)malloc(estimated_length * sizeof(Int));

    int last = 1;
    unsigned char *seqLetter = seqStr;
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

    actual_length = (Int) (seqp - seq) + 1;

    checkCudaErrors(cudaMalloc(&cudaSeq, actual_length * sizeof(Int)));
    checkCudaErrors(cudaMemcpy(cudaSeq, seq, actual_length * sizeof(Int), cudaMemcpyHostToDevice));

    free(seq);
}

void camCalculate (LyapCam *camP, Uint tw, Uint th, Uint td)
{
    if (camP->M < 1e-6)
        camP->M = 1e-6;

    camP->Q.normalize();

    if (td>0)
        camP->renderDenominator = td;

    if (tw>0) {
        camP->textureWidth = tw;
        camP->renderWidth = camP->textureWidth/camP->renderDenominator;
    }

    if (th>0) {
        camP->textureHeight = th;
        camP->renderHeight = camP->textureHeight/camP->renderDenominator;
    }

    camP->V = camP->Q.transform(Vec(0,0,1)).normalized();

    camP->S0 = camP->Q.transform(Vec(-camP->M, -camP->M, 1));

    camP->lightInnerCone = camP->V % (camP->Q.transform(Vec(-camP->M, -camP->M, 1.5)).normalized());
    camP->lightOuterCone = camP->V % (camP->Q.transform(Vec(-camP->M, -camP->M, 1)).normalized());

    Vec SX = Vec(2*camP->M / (Real)camP->renderWidth, 0, 0);
    camP->SDX = camP->Q.transform(SX);

    Vec SY = Vec(0, 2*camP->M / (Real)camP->renderHeight, 0);
    camP->SDY = camP->Q.transform(SY);
}


void init_scene()
{
    init_params();

    LyapLight *L = lights;

    for (int l=0; l<num_lights; l++, L++) {
        L->V = L->Q.transform(Vec(0,0,1)).normalized();
        L->lightInnerCone = L->V % (L->Q.transform(Vec(-L->M, -L->M, 1.5))).normalized();
        L->lightOuterCone = L->V % (L->Q.transform(Vec(-L->M, -L->M, 1))).normalized();
    }

    checkCudaErrors(cudaMalloc(&cudaLights, sizeof(LyapLight) * MAX_LIGHTS));

    checkCudaErrors(cudaMalloc(&cudaPoints, sizeof(LyapPoint) * imageWidth * imageHeight));

    cuda_load_sequence(sequence);

    camCalculate(&cam, windowWidth, windowHeight, renderDenominator);
}


/**
 * Perform render step in CUDA, and write results to PBO
 */
void render()
{
    size_t num_bytes;

    // Load lights into device memory
    checkCudaErrors(cudaMemcpy(cudaLights, lights, sizeof(LyapLight) * num_lights, cudaMemcpyHostToDevice));

    // Map PBO to get CUDA device pointer
    cudaPBO_map_count++;
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBO, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cudaRGBA, &num_bytes, cudaPBO));

    // call CUDA kernel, writing results to PBO
    //    for(int i = 0; i < passes; ++i) {
    //    void *dummy;
    kernel_calc_render<<<blocks, threads>>>(cudaRGBA, cudaPoints, cam, prm, cudaSeq, cudaLights, num_lights);
    //    cudaMemcpyAsync(dummy, dummy, 1, cudaMemcpyDeviceToDevice);
    //}

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
    cam.C.x += 0.001;
    glutPostRedisplay();
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
    //glutKeyboardFunc(keyboard);
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
