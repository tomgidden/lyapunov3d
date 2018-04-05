// -*- mode: cuda; -*-

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"

#ifndef __LYAP_STRUCTS_HPP__
#define __LYAP_STRUCTS_HPP__

typedef float Real;
typedef float3 Real3;
typedef float4 Real4;
typedef QUAT<Real, Real3, Real4> Quat;
typedef VEC3<Real, Real3> Vec;
typedef COLOR<Real, Real4> Color;
typedef unsigned int Uint;
typedef signed int Int;

typedef struct _LyapCamLight {
  Vec C;
  Quat Q;
  Real M;
  Vec V;
  Vec S0;
  Vec SDX;
  Vec SDY;
  Uint textureWidth;
  Uint textureHeight;
  Uint renderWidth;
  Uint renderHeight;
  Uint renderDenominator;

  Real lightInnerCone, lightOuterCone;
  Real lightRange;
  Color ambient;
  Color diffuseColor;
  Real diffusePower;
  Color specularColor;
  Real specularPower;
  Real specularHardness;
  Color chaosColor;
} LyapLight;

typedef struct _LyapCamLight LyapCam;

typedef struct {
  Real d;
  Uint settle;
  Uint accum;
  Uint stepMethod;
  Real nearThreshold;
  Real nearMultiplier;
  Real opaqueThreshold;
  Real chaosThreshold;
  Real depth;
  Real jitter;
  Real refine;
  Real gradient;
  Real lMin;
  Real lMax;
} LyapParams;

typedef struct {
  Vec P;
  Vec N;
  Real a;
  Real c;
} LyapPoint;

typedef struct {
    unsigned char r, g, b, a;
} RGBA;

#endif
