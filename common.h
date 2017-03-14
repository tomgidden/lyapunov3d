#ifdef __OPENCL_VERSION__

typedef struct { uchar r, g, b, a; } RGBA;
typedef int INT;
typedef int2 INT2;
typedef unsigned int UINT;
typedef float4 QUAT;
typedef float REAL;
typedef float4 REAL4;
typedef bool BOOL;

#  define VEC_EXTRAPOLATE(Q,P,V,t) { Q = P + V*t; }
#  define VEC_SCALE(Q,P,s) { Q = P * (float)s; }
#  define VEC_ADD(R,P,Q) { R = P + Q; }
#  define VEC_SUB(R,P,Q) { R = P - Q; }
#  define VEC_MAG2(R,V) { R = V.x*V.x + V.y*V.y + V.z*V.z; }
#  define VEC_CLAMP(R,V) { R = clamp(V, 0.0f, 1.0f); }
#  define NUM_CLAMP(R,N) { R = clamp(N, 0.0f, 1.0f); }
#  define VEC_DOT(R,P,Q) { R = dot(P,Q); }
#  define RETURN(X) return X
#  if USE_FAST_MATH
#    define VEC_MAG(R,V) {R=fast_length(V);}
#    define NORMALIZE(X) {X=fast_normalize(X);}
#  else
#    define VEC_MAG(R,V) {R=length(V);}
#    define NORMALIZE(X) {X=normalize(X);}
#  endif
#  if USE_NATIVE_MATH
#    define POWF(X,Y) pow(X,Y)
#    define FABSF(X) native_fabs(X)
#    define SQRTF(X) native_sqrt(X)
#    define LOGF(X) native_log(X)
#  else
#    define POWF(X,Y) pow(X,Y)
#    define FABSF(X) fabs(X)
#    define SQRTF(X) sqrt(X)
#    define LOGF(X) log(X)
#  endif

#else

#include "cglutils.h"

typedef cl_float REAL;

typedef struct {
  REAL x, y, z, w;
} my_float4 __attribute__((aligned(16)));

typedef my_float4 REAL4;
typedef cl_int INT;
typedef cl_uint UINT;
typedef REAL4 QUAT;
typedef QUAT* QUATP;
typedef unsigned char BOOL;

#include <math.h>
#include <stdio.h>
#include "real4.h"

#endif

typedef struct {
  REAL4 C;
  QUAT Q;
  REAL M;
  REAL4 V;
  REAL4 S0;
  REAL4 SDX;
  REAL4 SDY;
  UINT textureWidth;
  UINT textureHeight;
  UINT renderWidth;
  UINT renderHeight;
  UINT renderDenominator;

  REAL lightInnerCone, lightOuterCone;
  REAL lightRange;
  REAL4 ambient;
  REAL4 diffuseColor;
  REAL diffusePower;
  REAL4 specularColor;
  REAL specularPower;
  REAL specularHardness;
  REAL4 chaosColor;
} LyapCamLight;

typedef struct {
  REAL d;
  UINT settle;
  UINT accum;
  UINT stepMethod;
  REAL nearThreshold;
  REAL nearMultiplier;
  REAL opaqueThreshold;
  REAL chaosThreshold;
  REAL depth;
  REAL jitter;
  REAL refine;
  REAL gradient;
  REAL lMin;
  REAL lMax;
} LyapParams;

typedef struct {
  REAL x, y, z;
  REAL nx, ny, nz;
  REAL a;
  REAL c;
} LyapPoint;

