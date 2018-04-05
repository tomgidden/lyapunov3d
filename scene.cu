// -*- mode: cuda; -*-

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

#include "scene.hpp"

#include <stdio.h>

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif

void scene_lights_recalculate(LyapLight *lights, size_t num_lights)
{
    LyapLight *L = lights;

    for (int l=0; l<num_lights; l++, L++) {
        L->V = L->Q.transform(Vec(0,0,1)).normalized();
        L->lightInnerCone = L->V % (L->Q.transform(Vec(-L->M, -L->M, 1.5))).normalized();
        L->lightOuterCone = L->V % (L->Q.transform(Vec(-L->M, -L->M, 1))).normalized();
    }
}

void scene_cam_recalculate (LyapCam *camP, Uint tw, Uint th, Uint td)
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


/**
 * Parse and load the sequence string into device memory
 */
__host__ size_t scene_convert_sequence(Int **seqP, unsigned char *seqStr)
{
    Int *seq;
    size_t actual_length = 0;
    size_t estimated_length = 10 * strlen((const char *)seqStr) + 1;

    *seqP = (Int *)malloc(estimated_length * sizeof(Int));
    seq = *seqP;

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
    return actual_length;
}

