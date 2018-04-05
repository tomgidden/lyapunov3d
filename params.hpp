/* -*- mode: cuda; -*- */

#ifndef __LYAP_PARAMS_HPP__
#define __LYAP_PARAMS_HPP__

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

extern LyapParams prm;
extern LyapCam cam;

extern LyapLight lights[];
extern LyapLight L0;

extern unsigned int num_lights;

extern unsigned char *sequence;

const unsigned int MAX_LIGHTS = 16;


extern __host__ void params_init();

#endif
