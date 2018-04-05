/* -*- mode: cuda; -*- */

#ifndef __LYAP_SCENE_HPP__
#define __LYAP_SCENE_HPP__

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

extern __host__ size_t scene_convert_sequence(Int **seqP, unsigned char *seqStr);

extern __host__ void scene_cam_recalculate (LyapCam *camP, Uint tw, Uint th, Uint td);

extern __host__ void scene_lights_recalculate(LyapLight *lights, size_t num_lights);

#endif