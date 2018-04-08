/* -*- mode: cuda; -*- */

#ifndef __LYAP_KERNEL_HPP__
#define __LYAP_KERNEL_HPP__

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

extern __device__ Color shade(LyapPoint point, LyapCam cam, LyapLight *lights, Uint num_lights);

extern __device__ Real lyap4d(Vec P, Real d, Uint settle, Uint accum, const Int *seq);

extern __device__ Int raycast(LyapPoint *point, Uint sx, Uint sy, LyapCam cam, LyapParams prm, Int *seq);

extern __global__ void kernel_calc_render(RGBA *rgba, LyapPoint *points, LyapCam cam, LyapParams prm, Int *seq, LyapLight *lights, Uint num_lights);

extern __global__ void kernel_calc_volume(Real *exps, LyapParams prm, Int *seq);


#endif