/* -*- mode: cuda; -*- */

#include "params.hpp"

LyapParams prm;
LyapCam cam;

LyapLight lights[MAX_LIGHTS];
LyapLight L0;

unsigned int num_lights;

unsigned char *sequence;

void params_init()
{
    prm.d = 1.85;
    prm.settle = 10;
    prm.accum = 200;
    prm.stepMethod = 2;
    prm.nearThreshold = -1.0;
    prm.nearMultiplier = 2.0;
    prm.opaqueThreshold = -1.0;
    prm.chaosThreshold = 0;
    prm.depth = 256.0;
    prm.jitter = 0.5;
    prm.refine = 32.0;
    prm.gradient = 0.01;
    prm.lMin = 0.0;
    prm.lMax = 4.0;

    sequence = (unsigned char *)"BCABA";

    cam.C = Vec(3.51f, 3.5f, 3.5f);
    cam.Q = Quat(-0.039979,0.891346,-0.299458,-0.337976);
    cam.M = 0.5;

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

    L0 = lights[0];
};
