/* -*- mode: cuda; -*- */

#include "params.hpp"

LyapParams prm;
LyapCam cam;

LyapLight lights[MAX_LIGHTS];
LyapLight L0;

unsigned int num_lights;

unsigned char *sequence;

const unsigned int res = 4096;

unsigned int default_imageWidth = 3840;
unsigned int default_imageHeight = 2160;


void params_init()
{
    prm.d = 2.1;
    prm.settle = 18;
    prm.accum = 1008;
    prm.stepMethod = 2;   // 2: better, slightly slower. 1: worse.
    prm.nearThreshold = -1.0;
    prm.nearMultiplier = 2.0;
    prm.opaqueThreshold = -0.75;
    prm.chaosThreshold = -0.5;
    prm.depth = res;
    prm.jitter = 0.5;
    prm.refine = 32;
    prm.gradient = 0.01;
    prm.lMin = 0.0;
    prm.lMax = 4.0;

//    sequence = (unsigned char *)"A6B6C6D6";
    sequence = (unsigned char *)"BCABA";
//    sequence = (unsigned char*)"A6B6C6";

    /*startcalc*/    Vec dir = Vec(4,4,4);

    Vec side = Vec(-4,4,4);
    side.normalize();

    Vec up = side * dir.normalized();
    up.normalize();

    Quat rot0 = Quat(up, -20, 1);
    Quat rot1 = Quat(up, 20, 1);

    Quat nrot = rot0.nlerp(rot1, 1);
    Vec nd = nrot.transform(dir).normalized() * -1.0 ;

    cam.C = Vec(4.0 - 0.9*1, 4.0 - 0.9*1, 4.0 - 0.9*1) - nd;
    cam.Q = Quat(Vec(0,0,1), nd, 1.0);

//    cam.C = Vec(4.125, 4.125, 4.125);
//    cam.Q = Quat(0.820473,-0.339851,-0.175920,0.424708);

/*endcalc*/
    cam.M = 0.45;

    if (false) {
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
    }
    else {
        lights[0].C = Vec(6.0f, 5.0f, 3.0f);
        lights[0].Q = Quat(0.710595f, 0.282082f, -0.512168f, 0.391368f);
        lights[0].M = 0.500000;
        lights[0].lightInnerCone = 0.904535f;
        lights[0].lightOuterCone = 0.816497f;
        lights[0].lightRange = 1.0;
        lights[0].ambient = Color(0.1, 0, 0, 0);
        lights[0].diffuseColor = Color(1.0, 0.25, 0.125, 1);
        lights[0].diffusePower = 10.0;
        lights[0].specularColor = Color(1.0, 1.0, 1.0, 1);
        lights[0].specularPower = 10.0;
        lights[0].specularHardness = 10.0;
        lights[0].chaosColor = Color(0, 0, 0, 0);
        num_lights = 1;
    }


    L0 = lights[0];
};
