#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_gl.h>

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

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

#include "real4.h"
#include "structs.h"

// Image and grid parameters
const uint imageWidth = 800, imageHeight = 500;
const dim3 gridSize(imageWidth, imageHeight);
const uint windowWidth = imageWidth, windowHeight = imageHeight;
const uint renderDenominator = 1;

// Scene parameters
const uint MAX_LIGHTS = 16;


LyapParams prm = {
  .d = 2.10,
  .settle = 5,
  .accum = 10,
  .stepMethod = 3,
  .nearThreshold = -1.000000,
  .nearMultiplier = 2.000000,
  .opaqueThreshold = -1.125000,
  .chaosThreshold = 100000.000000,
  .depth = 16.000000,
  .jitter = 0.500000,
  .refine = 32.000000,
  .gradient = 0.010000,
  .lMin = 0.000000,
  .lMax = 4.000000
};

unsigned char *sequence = (unsigned char *)"AAAAAABBBBBBCCCCCCDDDDDD";


LyapCam cam = {
  .C = {4.010000,4.000000,4.000000,0.000000},
  .Q = {0.820473,-0.339851,-0.175920,0.424708},
  .M = 0.500000
};

LyapLight lights[] = {
    {
        .C = {5.000000,7.000000,3.000000,0.000000},
        .Q = {0.710595,0.282082,-0.512168,0.391368},
        .M = 0.500000,
        .lightInnerCone = 0.904535,
        .lightOuterCone = 0.816497,
        .lightRange = 1.000000,
        .ambient = {0.000000,0.000000,0.000000,0.000000},
        .diffuseColor = {0.300000,0.400000,0.500000,0.000000},
        .diffusePower = 10.000000,
        .specularColor = {0.900000,0.900000,0.900000,0.000000},
        .specularPower = 10.000000,
        .specularHardness = 10.000000,
        .chaosColor = {0.000000,0.000000,0.000000,0.000000}
    },

    {
        .C = {3.000000,7.000000,5.000000,0.000000},
        .Q = {0.039640,0.840027,-0.538582,-0.052093},
        .M = 1.677200,
        .lightInnerCone = 0.534489,
        .lightOuterCone = 0.388485,
        .lightRange = 0.500000,
        .ambient = {0.000000,0.000000,0.000000,0.000000},
        .diffuseColor = {0.300000,0.374694,0.200000,0.000000},
        .diffusePower = 10.000000,
        .specularColor = {1.000000,1.000000,1.000000,0.000000},
        .specularPower = 10.000000,
        .specularHardness = 10.000000,
        .chaosColor = {0.000000,0.000000,0.000000,0.000000}
    }
};
unsigned int num_lights = 2;

// Data transfer of Pixel Buffer Object between CUDA and GL
GLuint pbo;
struct cudaGraphicsResource *cuda_pbo_resource;
// A simple semaphore to indicate whether the PBO has been mapped or not
volatile int pbo_map_flag = 0;

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif

// Device array of LyapPoint points
__device__ LyapPoint *cudaPoints = 0;

// Device array of lights
__device__ LyapLight *cudaLights = 0;

// Device pixel buffer
__device__ RGBA *cudaRGBA;

// Device sequence array
__device__ INT *cudaSeq;

/**
 * Convert the lyapunov point into a pixel for rendering
 */
__device__ REAL4 shade(LyapPoint point, LyapCam cam, LyapLight *lights, UINT num_lights)
{
    REAL4 color = REAL4_init(0,0,0);

    if(!isnan(point.a)) {
        UINT l;

        // For each defined light
        for(l=0; l<num_lights; ++l) {
            REAL4 camV, lightV, diffuse, halfV, specular;
            REAL lightD2, i, j;

            REAL4 P = REAL4_init(point.x, point.y, point.z);
            REAL4 N = REAL4_init(point.nx, point.ny, point.nz);
            REAL a = point.a;
            REAL c = point.c;

            camV = REAL4_sub(cam.C, P);
            lightV = REAL4_sub(lights[l].C, P);
            lightD2 = REAL4_mag2(lightV);

            // Get the length of lightV (for falloff)
            // but then normalize lightV.
            lightV = REAL4_normalize(lightV);

            i = REAL4_dot(lightV, N);
            j = REAL4_dot(lightV, lights[l].V);
            j = -j;

            if (j > lights[l].lightOuterCone) {
                i = REAL_clamp(i);
                diffuse = REAL4_scale(lights[l].diffuseColor, i*lights[l].diffusePower);

                halfV = REAL4_normalize(REAL4_add(lightV, camV));

                i = REAL_clamp(REAL4_dot(N, halfV));

                i = POWF(i, lights[l].specularHardness);

                specular = REAL4_scale(lights[l].specularColor, i*lights[l].specularPower);

                lightV = REAL4_add(specular, diffuse);

                lightV = REAL4_scale(lightV, lights[l].lightRange/lightD2);

                if ( j < lights[l].lightInnerCone) {
                    lightV = REAL4_scale(lightV, (j-lights[l].lightOuterCone)/(lights[l].lightInnerCone-lights[l].lightOuterCone));
                }
                lightV = REAL4_add(lightV, lights[l].ambient);
            }
            else {
                lightV = lights[l].ambient;
            }

            if(c>0.0) {
                REAL4 chaos;
                chaos = REAL4_scale(lights[l].chaosColor, 0.1125/LOGF(c));
                lightV = REAL4_add(lightV, chaos);
            }

            lightV.w = 1;

            color = REAL4_add(color, lightV);
        }
    }

    return color;
}

__device__ static __inline__ INT in_lyap_space(REAL4 P, REAL lmin, REAL lmax)
{
    return ((P.x>=lmin) && (P.x<=lmax) &&
            (P.y>=lmin) && (P.y<=lmax) &&
            (P.z>=lmin) && (P.z<=lmax));
}

__device__ static REAL lyap4d(REAL4 P, REAL d, UINT settle, UINT accum, const INT *seq)
{
    REAL abcd[4];
    abcd[0] = P.x;
    abcd[1] = P.y;
    abcd[2] = P.z;
    abcd[3] = d;

    //REAL px=P.x-2.0, py=P.y-2.0, pz=P.z-2.0;
    //  if(SQRTF(px*px + py*py + pz*pz) < 2.0)
    //    return -1;
    //  else
    //    return 0;

    UINT seqi; // Position in the sequence loop
    UINT n;    // Iteration counter
    REAL r; // Iteration value
    REAL v = 0.5; // Iterating value
    REAL l = 0; // Result accumulator

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
            l += LOGF(r);
            if(!isfinite(l)) { return NAN; }
        }
    }

    return l/(REAL)accum;
}

__device__ static INT raycast(REAL4 *PP,
                              REAL4 *NP,
                              REAL *aP,
                              REAL *cP,
                              UINT sx, UINT sy,
                              LyapCam cam,
                              LyapParams prm,
                              INT *seq)
{
    // Work out the direction vector: start at C (camera), and
    // find the point on the screen plane (in 3D)
    REAL4 V = REAL4_init(cam.S0.x + cam.SDX.x * sx + cam.SDY.x * sy,
                         cam.S0.y + cam.SDX.y * sx + cam.SDY.y * sy,
                         cam.S0.z + cam.SDX.z * sx + cam.SDY.z * sy);

    V = REAL4_normalize(V);
    V = REAL4_scale(V, 1.0/cam.M);

    REAL4 P;
    REAL4 N;

    REAL a; // high-low alpha
    REAL c; // chaos alpha
    REAL l;

    bool near = false;
    INT i;
    REAL t;

//REAL thresholdRange = prm.opaqueThreshold - prm.nearThreshold;

    // Find start and end point of ray within Lyapunov space cube

    REAL t0=MAXFLOAT, t1=0;
    // Find ray intersection through entire Lyapunov space cube

    // First, find values for 't' for intersections with the six infinite
    // planes x=0, x=4, y=0, y=4, z=0, and z=4.
    REAL ts[6];
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
    INT i0=-1, i1=-1;

    if(ts[0] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[0]);
        if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
            ts[0] = NAN;
    }

    if(ts[1] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[1]);
        if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
            ts[1] = NAN;
    }

    if(ts[2] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[2]);
        if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
            ts[2] = NAN;
    }

    if(ts[3] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[3]);
        if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
            ts[3] = NAN;
    }

    if(ts[4] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[4]);
        if(P.x<LMIN || P.x>LMAX || P.y<LMIN || P.y>LMAX)
            ts[4] = NAN;
    }

    if(ts[5] != INFINITY) {
        P = REAL4_extrapolate(cam.C, V, ts[5]);
        if(P.x<LMIN || P.x>LMAX || P.y<LMIN || P.y>LMAX)
            ts[5] = NAN;
    }

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
    P = REAL4_extrapolate(cam.C, V, t);

    // Set the alpha accumulators to zero
    a = 0;
    c = 0;

    // dt is the amount to add to 't' for each step in the initial
    // ray progression.  We calculate Fdt for the normal value,
    // and Ndt for the finer value used when close to the threshold
    // (ie. under nearThreshold)
    REAL dt, Ndt, Fdt;

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
        Fdt = REAL4_mag(V) / prm.depth;
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
            REAL jit = l-trunc(l);
            if(jit<0)
                jit = 1.0 - jit*prm.jitter;
            else
                jit = 1.0 + jit*prm.jitter;

            if(isfinite(jit)) {
                t += dt*jit;
                P = REAL4_extrapolate(P, V, dt*jit);
            }
            else {
                t += dt;
                P = REAL4_extrapolate(P, V, dt);
            }
        }
        else {
            // No jitter, so just add 'dt'.
            t += dt;
            P = REAL4_extrapolate(P, V, dt);
        }

        // If the ray has exited Lyapunov space, then bugger it.
        if(t>t1 || !in_lyap_space(P, LMIN, LMAX)) {
            return 1;
        }

        // Calculate this point's exponent
        l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

        // If the ray is still in transparent space, then we may still
        // want to accumulate alpha for clouding.
        if(l > prm.chaosThreshold)
            c += l;

        if(l <= prm.nearThreshold && !near) {
            near = true;
            dt = Ndt;
        }
        else if(l > prm.nearThreshold && near) {
            near = false;
            dt = Fdt;
        }
    }

    // At this point, the ray has either hit an opaque point, or
    // has exited Lyapunov space.

    // If the ray has exited space, then this point is no longer
    // relevant.
    if(t>t1 || !in_lyap_space(P, LMIN, LMAX)) {
        return 1;
    }

    // Ray phase 2: now we've hit the surface, we now need to hone the
    // intersection point by reversing back along the ray at half the speed.

    // If we've gone through then sign is 0. 'sign' is
    // the direction of the progression.
    BOOL sign = 0;
    BOOL osign = sign;

    // Half speed
    REAL Qdt = dt * -0.5f;
    REAL4 QdV = REAL4_scale(V, Qdt);

    // Set the range of the honing to <t-dt, t>.
    REAL Qt1 = t;
    REAL Qt0 = t-dt;

    // Honing continues reversing back and forth, halving speed
    // each time. Once dt is less than or equal to dt/refine,
    // we stop: it's close enough.
    REAL min_Qdt = dt/prm.refine;

    // While 't' is still in the range <t-dt, t> AND dt is still
    // of significant size...
    while(t<=Qt1 && t>=Qt0 && (Qdt<=-min_Qdt || Qdt>=min_Qdt)) {

        // Progress along the ray
        t += Qdt;
        P = REAL4_add(P, QdV);

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
            QdV = REAL4_scale(QdV, -0.5f);
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

    REAL mag = dt * prm.gradient;
    REAL4 Ps[6] = {P,P,P,P,P,P};
    Ps[0].x -= mag;
    Ps[1].x += mag;
    Ps[2].y -= mag;
    Ps[3].y += mag;
    Ps[4].z -= mag;
    Ps[5].z += mag;

    REAL ls[6];
    for(i=0; i<6; i++) {
        ls[i] = lyap4d(Ps[i], prm.d, prm.settle, prm.accum, seq);
    }

    N.x = ls[1]-ls[0];
    N.y = ls[3]-ls[2];
    N.z = ls[5]-ls[4];
    N.w = 0;

    N = REAL4_normalize(N);

    // Okay, we've done it. Output the hit point, the normal, the exact
    // exponent at P (not REALly needed, but it does signal a hit failure
    // by l == NAN), and the accumulated alpha.

    *PP = P;
    *NP = N;
    *aP = a;
    *cP = c;

    return 0;
}

__global__ void kernel_calc_render(RGBA *rgba,
                                   LyapPoint *points,
                                   LyapCam cam,
                                   LyapParams prm,
                                   INT *seq,
                                   LyapLight *lights,
                                   UINT num_lights)
{
    REAL4 P, N;
    REAL a, c;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int ind = x + y * gridDim.x;

    //uint sx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    //uint sy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    //float su = sx / (float) cam.renderWidth;
    //float sv = sy / (float) cam.renderHeight;
    //uint i = __umul24(sy, cam.textureWidth) + sx;

    INT ret = raycast(&P,
                      &N,
                      &a,
                      &c,
                      x, y,
                      cam,
                      prm,
                      seq);

    points[ind].x = P.x;
    points[ind].y = P.y;
    points[ind].z = P.z;
    points[ind].nx = N.x;
    points[ind].ny = N.y;
    points[ind].nz = N.z;
    points[ind].a = ret ? NAN : a;
    points[ind].c = c;

    REAL4 color = shade(points[ind], cam, lights, num_lights);

    rgba[ind].r = (int)(255.0 * color.x) & 0xff;
    rgba[ind].g = (int)(255.0 * color.y) & 0xff;
    rgba[ind].b = (int)(255.0 * color.z) & 0xff;
    rgba[ind].a = (int)(255.0 * color.w) & 0xff;
}

/**
 * Parse and load the sequence string into device memory
 */
void cuda_load_sequence(INT **ret, unsigned char *seqStr)
{
    INT *seq;
    size_t actual_length = 0;
    size_t estimated_length = 10 * strlen((const char *)seqStr) + 1;

    seq = (INT *)malloc(estimated_length * sizeof(INT));

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

    actual_length = (INT) (seqp - seq);

    checkCudaErrors(cudaMalloc(ret, actual_length * sizeof(INT)));
    checkCudaErrors(cudaMemcpy(*ret, seq, actual_length * sizeof(INT), cudaMemcpyHostToDevice));

    free(seq);
}

void camCalculate (LyapCam *camP, UINT tw, UINT th, UINT td)
{
  if(camP->M<QUAT_EPSILON)
    camP->M = QUAT_EPSILON;

  camP->Q = QUAT_normalize(camP->Q);

  if(td>0)
    camP->renderDenominator = td;

  if(tw>0) {
    camP->textureWidth = tw;
    camP->renderWidth = camP->textureWidth/camP->renderDenominator;
  }

  if(th>0) {
    camP->textureHeight = th;
    camP->renderHeight = camP->textureHeight/camP->renderDenominator;
  }

  camP->V = REAL4_normalize(QUAT_transformREAL4(REAL4_init(0,0,1), camP->Q));

  camP->S0 = QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1), camP->Q);

  camP->lightInnerCone = REAL4_dot(camP->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1.5), camP->Q)));
  camP->lightOuterCone = REAL4_dot(camP->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1), camP->Q)));

  camP->SDX = QUAT_transformREAL4(REAL4_init(2*camP->M/(REAL)camP->renderWidth, 0, 0), camP->Q);
  camP->SDY = QUAT_transformREAL4(REAL4_init(0, 2*camP->M/(REAL)camP->renderHeight, 0), camP->Q);
}


void init_scene()
{
    LyapLight *L = lights;
    for (int l=0; l<num_lights; l++, L++) {
        L->V = REAL4_normalize(QUAT_transformREAL4(REAL4_init(0,0,1), L->Q));
        L->lightInnerCone = REAL4_dot(L->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-L->M, -L->M, 1.5), L->Q)));
        L->lightOuterCone = REAL4_dot(L->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-L->M, -L->M, 1), L->Q)));
    }

    checkCudaErrors(cudaMalloc(&cudaLights, sizeof(LyapLight) * MAX_LIGHTS));

    checkCudaErrors(cudaMalloc(&cudaPoints, sizeof(LyapPoint) * imageWidth * imageHeight));

    cuda_load_sequence(&cudaSeq, sequence);

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

    // Increment the map flag: like a semaphore
    pbo_map_flag++;

    // Map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cudaRGBA, &num_bytes, cuda_pbo_resource));

    // call CUDA kernel, writing results to PBO
    kernel_calc_render<<<gridSize, 1>>>(cudaRGBA, cudaPoints, cam, prm, cudaSeq, cudaLights, num_lights);

    /*
    int points_size = sizeof(LyapPoint) * imageWidth * imageHeight;
    printf("Points size = %d\n", points_size);

    LyapPoint *myPoints = (LyapPoint *)malloc(points_size);
    printf("malloc'ed %p.\n", myPoints);

    checkCudaErrors(cudaMemcpy( myPoints, cudaPoints, points_size, cudaMemcpyDeviceToHost ));

    LyapPoint *ptr = myPoints;
    for (int y=0; y<2; y++)
        for (int x=0; x<2; x++) {
            printf("%d:\t%f,%f,%f\t%f,%f,%f\n",
               points_size,
               ptr->x, ptr->y, ptr->z,
               ptr->nx, ptr->ny, ptr->nz);
        }

    free(myPoints);
    */

    // Handle error
    getLastCudaError("render_kernel failed");

    // Unmap cleanly
    if (pbo_map_flag) {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        pbo_map_flag--;
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
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
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
    if (pbo_map_flag) {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        pbo_map_flag--;
    }

    // Unregister this buffer object from CUDA and from GL
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
}

void initGLBuffers()
{
    // Create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageWidth*imageHeight*sizeof(RGBA), 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Lyapunov2018");
    glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    if (!isGLVersionSupported(2,0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }
}

/**
 * Find CUDA device
 */
int chooseCudaDevice(int argc, char **argv, bool use_gl)
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

    // First initialize OpenGL context, so we can properly set the GL for
    // CUDA.  This is necessary in order to achieve optimal performance
    // with OpenGL/CUDA interop.
    initGL(&argc, argv);

    // Use command-line specified CUDA device, otherwise use device with
    // highest Gflops/s
    chooseCudaDevice(argc, argv, true);

    // OpenGL buffers
    initGLBuffers();

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
