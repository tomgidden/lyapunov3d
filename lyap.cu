// -*- mode: cuda; -*-

#include "vec3.hpp"
#include "quat.hpp"
#include "color.hpp"
#include "structs.hpp"

#include "lyap.hpp"

#include <stdio.h>

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif



/**
 * Convert the lyapunov point into a pixel for rendering
 */
__device__ Color shade(LyapPoint point, LyapCam cam, LyapLight *lights, Uint num_lights)
{
    Color color = Color();

    if (isnan(point.a)) {
        color.x = 1;
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

__device__ Real lyap4d(Vec P, Real d, Uint settle, Uint accum, const Int *seq)
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

    if((v-0.5 <= -1e-8) || (v-0.5 >= 1e-8)) {
        // Now calculate the value by running the iteration with accumulation
        for(n = 0; n < accum; n++) {
            r = abcd[seq[seqi++]];
            if(seq[seqi]==-1) seqi = 0;
            v = r * v * (1.0 - v);
            r = r - 2.0 * r * v;
            if (r < 0) r = -r;
            l += Vec::logf(r);
            if (!isfinite(l)) {
                return NAN;
            }
        }
    }

    return l/(Real)accum;
}

__device__ Int raycast(LyapPoint *point, Uint sx, Uint sy, LyapCam cam, LyapParams prm, Int *seq)
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
        if (isfinite(ts[i])) {
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

            if (isfinite(jit)) {
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

        // If the ray has passed the first exit plane, then bugger it.
        //if (t>t1 || !P.in_lyap_space()) { // Overkill: passing t1 should be the exit of L-space anyway
        if (t>t1) {
            return 1;
        }

        // Calculate this point's exponent
        l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

        // If the ray is still in transparent space, then we may still
        // want to accumulate alpha for clouding.
        if (l > prm.chaosThreshold) {
            c += l;
        }
        else if (l > prm.opaqueThreshold) {
            a += l;
        }

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
    //if (t>t1 || !P.in_lyap_space()) { // Overkill: passing t1 should be the exit of L-space anyway
    if (t>t1) {
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
    while (t<=Qt1 && t>=Qt0 && (Qdt<=-min_Qdt || Qdt>=min_Qdt)) {

        // Progress along the ray
        t += Qdt;
        P += QdV;

        // Calculate the exponent
        l = lyap4d(P, prm.d, prm.settle, prm.accum, seq);

        // If we've hit the threshold exactly, short-circuit.
        if (l==prm.opaqueThreshold) break;

        // Work out whether we reverse or not:
        osign = sign;
        sign = (l < prm.opaqueThreshold) ? 0 : 1;

        // If we've reversed, then halve the speed
        if (sign != osign) {
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

__global__ void kernel_calc_render(RGBA *rgba, LyapPoint *points, LyapCam cam, LyapParams prm, Int *seq, LyapLight *lights, Uint num_lights)
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

