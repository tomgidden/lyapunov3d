extern REAL4 shade(__global LyapPoint *buf, UINT ind, __constant LyapCamLight *cam, __constant LyapCamLight *Ls, UINT numLights);

#if USE_LMINMAX
#define LMIN prm->lMin
#define LMAX prm->lMax
#else
#define LMIN 0.0
#define LMAX 4.0
#endif

REAL4 shade(__global LyapPoint *buf, UINT ind, __constant LyapCamLight *cam, __constant LyapCamLight *Ls, UINT numLights)
{
  REAL4 color = {0,0,0,0};

  if(!isnan(buf[ind].a)) {
    UINT l;
    for(l=0; l<numLights; ++l) {
      REAL4 camV, lightV, diffuse, halfV, specular;
      REAL lightD2, i, j;
      REAL4 P = {buf[ind].x, buf[ind].y, buf[ind].z, 0};
      REAL4 N = {buf[ind].nx, buf[ind].ny, buf[ind].nz, 0};
      //REAL a = buf[ind].a;
      REAL c = buf[ind].c;

      VEC_SUB(camV, cam->C, P);
      VEC_SUB(lightV, Ls[l].C, P);
      VEC_MAG2(lightD2, lightV);

      // Get the length of lightV (for falloff)
      REAL lightD = SQRTF(lightD2);
      // but then normalize lightV.
      lightV /= lightD;

      VEC_DOT(i, lightV, N);

      VEC_DOT(j, -lightV, Ls[l].V);
      if(j>Ls[l].lightOuterCone) {
        NUM_CLAMP(i, i);
        VEC_SCALE(diffuse, Ls[l].diffuseColor, i*Ls[l].diffusePower);

        VEC_ADD(halfV, lightV, camV);
        NORMALIZE(halfV);

        VEC_DOT(i, N, halfV);
        NUM_CLAMP(i, i);

        i = POWF(i, Ls[l].specularHardness);
        VEC_SCALE(specular, Ls[l].specularColor, i*Ls[l].specularPower);

        VEC_ADD(lightV, specular, diffuse);

        VEC_SCALE(lightV, lightV, Ls[l].lightRange/lightD2);

        if(j<Ls[l].lightInnerCone) {
          VEC_SCALE(lightV, lightV, (j-Ls[l].lightOuterCone)/(Ls[l].lightInnerCone-Ls[l].lightOuterCone));
        }
        VEC_ADD(lightV, lightV, Ls[l].ambient);
      }
      else {
        lightV = Ls[l].ambient;
      }
      if(c>0.0) {
        REAL4 chaos;
        VEC_SCALE(chaos, Ls[l].chaosColor, 0.1125/LOGF(c));
        lightV += chaos;
      }

      lightV.w = 1;

      color += lightV;
    }
  }

  VEC_CLAMP(color, color);
  return color;
}

static inline INT in_lyap_space(REAL4 P, REAL lmin, REAL lmax)
{
  return ((P.x>=lmin) && (P.x<=lmax) &&
          (P.y>=lmin) && (P.y<=lmax) &&
          (P.z>=lmin) && (P.z<=lmax));
}

REAL lyap4d(REAL4 P, REAL d, UINT settle, UINT accum, __constant INT *seq)
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

static inline INT raycast(REAL4 *PP,
                          REAL4 *NP,
                          REAL *aP,
                          REAL *cP,
                          INT2 coord,
                          __constant LyapCamLight *cam,
                          __constant LyapParams *prm,
                          __constant INT *seq)
{
  // Work out the direction vector: start at C (camera), and
  // find the point on the screen plane (in 3D)
  REAL4 V = cam->S0 + (cam->SDX*(REAL)coord.x) + (cam->SDY*(REAL)coord.y);
  NORMALIZE(V);
  VEC_SCALE(V,V,1.0/cam->M);

  REAL4 P;
  REAL4 N;

  REAL a; // high-low alpha
  REAL c; // chaos alpha
  REAL l;

  bool near = false;
  INT i;
  REAL t;

  REAL thresholdRange = prm->opaqueThreshold - prm->nearThreshold;

  // Find start and end point of ray within Lyapunov space cube

  REAL t0=MAXFLOAT, t1=0;
  // Find ray intersection through entire Lyapunov space cube

  // First, find values for 't' for intersections with the six infinite
  // planes x=0, x=4, y=0, y=4, z=0, and z=4.
  REAL ts[6];
  ts[0] = (V.x!=0.0f) ? ((LMIN-cam->C.x) / V.x) : INFINITY;
  ts[1] = (V.x!=0.0f) ? ((LMAX-cam->C.x) / V.x) : INFINITY;
  ts[2] = (V.y!=0.0f) ? ((LMIN-cam->C.y) / V.y) : INFINITY;
  ts[3] = (V.y!=0.0f) ? ((LMAX-cam->C.y) / V.y) : INFINITY;
  ts[4] = (V.z!=0.0f) ? ((LMIN-cam->C.z) / V.z) : INFINITY;
  ts[5] = (V.z!=0.0f) ? ((LMAX-cam->C.z) / V.z) : INFINITY;

  // Zero, one or two of these intersections will occur at a point
  // where all x, y and z are between 0 and 4 (ie. inside Lyapunov
  // space). Actually, all six might occur within Lyapunov space, but
  // only if the points are equal. So, zero, one or two _unique_
  // intersections will occur. This is because zero, one or two
  // points on a line intersect a cube.
  INT i0=-1, i1=-1;

  if(ts[0] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[0]);
    if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
      ts[0] = NAN;
  }

  if(ts[1] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[1]);
    if(P.y<LMIN || P.y>LMAX || P.z<LMIN || P.z>LMAX)
      ts[1] = NAN;
  }

  if(ts[2] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[2]);
    if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
      ts[2] = NAN;
  }

  if(ts[3] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[3]);
    if(P.x<LMIN || P.x>LMAX || P.z<LMIN || P.z>LMAX)
      ts[3] = NAN;
  }

  if(ts[4] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[4]);
    if(P.x<LMIN || P.x>LMAX || P.y<LMIN || P.y>LMAX)
      ts[4] = NAN;
  }

  if(ts[5] != INFINITY) {
    VEC_EXTRAPOLATE(P, cam->C, V, ts[5]);
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
    RETURN(1);
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
  VEC_EXTRAPOLATE(P, cam->C, V, t);

  // Set the alpha accumulators to zero
  a = 0;
  c = 0;

  // dt is the amount to add to 't' for each step in the initial
  // ray progression.  We calculate Fdt for the normal value,
  // and Ndt for the finer value used when close to the threshold
  // (ie. under nearThreshold)
  REAL dt, Ndt, Fdt;

  // There are different methods of progressing along the ray.

  switch (prm->stepMethod) {
  case 1:
    // Method 1 divides the distance between the start and the end of
    // the ray equally.
    Fdt = (t1-t0) / prm->depth;
    break;

  case 2:
  default:
    // Method 2 (default) divides the distance from the camera to the
    // virtual screen equally.
    Fdt = length(V) / prm->depth;
  }

  Ndt = dt/prm->nearMultiplier;

  dt = Fdt;
  near = false;

  // Calculate the exponent at the current point.
  l = lyap4d(P, prm->d, prm->settle, prm->accum, seq);

  // Okay, now we do the initial ray progression: we trace until the
  // exponent for the point is below a certain value. This value is
  // effectively the transition between transparent and opaque.

  // While the exponent is above the surface threshold (ie. while the
  // current point is in "transparent" space)...
  while (l > prm->opaqueThreshold) {

    // Step along the ray by 'dt' plus/minus a certain amount of
    // jitter (optional). This reduces moire fringes and herringbones
    // resulting from transitioning through thin sheets. Instead we
    // get what looks like noise, but is in fact stochastic sampling
    // of a diaphanous transition membrane.

    if(prm->jitter != 0.0) {
      // We use the fractional part of the last Lyapunov exponent
      // as a pseudo-random number. This is then added to 'dt', scaled
      // by the amount of jitter requested.
      REAL jit = l-trunc(l);
      if(jit<0)
        jit = 1.0 - jit*prm->jitter;
      else
        jit = 1.0 + jit*prm->jitter;

      if(isfinite(jit)) {
        t += dt*jit;
        VEC_EXTRAPOLATE(P, P, V, dt*jit);
      }
      else {
        t += dt;
        VEC_EXTRAPOLATE(P, P, V, dt);
      }
    }
    else {
      // No jitter, so just add 'dt'.
      t += dt;
      VEC_EXTRAPOLATE(P, P, V, dt);
    }

    // If the ray has exited Lyapunov space, then bugger it.
    if(t>t1 || !in_lyap_space(P, LMIN, LMAX)) {
      RETURN(1);
    }

    // Calculate this point's exponent
    l = lyap4d(P, prm->d, prm->settle, prm->accum, seq);

    // If the ray is still in transparent space, then we may still
    // want to accumulate alpha for clouding.
    if(l > prm->chaosThreshold)
      c += l;

    if(l <= prm->nearThreshold && !near) {
      near = true;
      dt = Ndt;
    }
    else if(l > prm->nearThreshold && near) {
      near = false;
      dt = Fdt;
    }
  }

  // At this point, the ray has either hit an opaque point, or
  // has exited Lyapunov space.

  // If the ray has exited space, then this point is no longer
  // relevant.
  if(t>t1 || !in_lyap_space(P, LMIN, LMAX)) {
    RETURN(1);
  }

  // Ray phase 2: now we've hit the surface, we now need to hone the
  // intersection point by reversing back along the ray at half the speed.

  // If we've gone through then sign is 0. 'sign' is
  // the direction of the progression.
  BOOL sign = 0;
  BOOL osign = sign;

  // Half speed
  REAL Qdt = dt * -0.5f;
  REAL4 QdV;
  VEC_SCALE(QdV, V, Qdt);

  // Set the range of the honing to <t-dt, t>.
  REAL Qt1 = t;
  REAL Qt0 = t-dt;

  // Honing continues reversing back and forth, halving speed
  // each time. Once dt is less than or equal to dt/refine,
  // we stop: it's close enough.
  REAL min_Qdt = dt/prm->refine;

  // While 't' is still in the range <t-dt, t> AND dt is still
  // of significant size...
  while(t<=Qt1 && t>=Qt0 && (Qdt<=-min_Qdt || Qdt>=min_Qdt)) {

    // Progress along the ray
    t += Qdt;
    VEC_ADD(P, P, QdV);

    // Calculate the exponent
    l = lyap4d(P, prm->d, prm->settle, prm->accum, seq);

    // If we've hit the threshold exactly, short-circuit.
    if(l==prm->opaqueThreshold) break;

    // Work out whether we reverse or not:
    osign = sign;
    sign = (l < prm->opaqueThreshold) ? 0 : 1;

    // If we've reversed, then halve the speed
    if(sign != osign) {
      Qdt *= -0.5f;
      VEC_SCALE(QdV, QdV, -0.5f);
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

  REAL mag = dt * prm->gradient;
  REAL4 Ps[6] = {P,P,P,P,P,P};
  Ps[0].x -= mag;
  Ps[1].x += mag;
  Ps[2].y -= mag;
  Ps[3].y += mag;
  Ps[4].z -= mag;
  Ps[5].z += mag;

  REAL ls[6];
  for(i=0; i<6; i++) {
    ls[i] = lyap4d(Ps[i], prm->d, prm->settle, prm->accum, seq);
  }

  N.x = ls[1]-ls[0];
  N.y = ls[3]-ls[2];
  N.z = ls[5]-ls[4];
  N.w = 0;

  NORMALIZE(N);

  // Okay, we've done it. Output the hit point, the normal, the exact
  // exponent at P (not REALly needed, but it does signal a hit failure
  // by l == NAN), and the accumulated alpha.

  *PP = P;
  *NP = N;
  *aP = a;
  *cP = c;

  RETURN(0);
}

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

static inline void locate(int2 *coordP, UINT *indP, UINT start, UINT base, __constant LyapCamLight *cam)
{
  UINT o = (UINT)get_global_id(0);
  UINT p = o + start;

  coordP->x = p % cam->renderWidth;
  coordP->y = (UINT)(((REAL)(p-coordP->x)) / (REAL)cam->renderWidth);

  *indP = o;
}


__kernel void lyapcalc(__global LyapPoint *buf,
                       UINT start,
                       UINT base,
                       __constant LyapCamLight *cam,
                       __constant LyapParams *prm,
                       __constant INT *seq)
{
  INT2 coord;
  UINT ind;
  locate(&coord, &ind, start, base, cam);

  REAL4 P, N;
  REAL a, c;

  INT ret = raycast(&P,
                    &N,
                    &a,
                    &c,
                    coord,
                    cam,
                    prm,
                    seq);

  buf[ind].x = P.x;
  buf[ind].y = P.y;
  buf[ind].z = P.z;
  buf[ind].nx = N.x;
  buf[ind].ny = N.y;
  buf[ind].nz = N.z;
  buf[ind].a = ret ? NAN : a;
  buf[ind].c = c;
}

__kernel void lyapcalcrendergl(__global LyapPoint *buf,
                               __write_only image2d_t img,
                               UINT start,
                               UINT base,
                               __constant LyapCamLight *cam,
                               __constant LyapParams *prm,
                               __constant INT *seq,
                               __constant LyapCamLight *Ls,
                               UINT numLights)
{
  INT2 coord;
  UINT ind;
  locate(&coord, &ind, start, base, cam);

  REAL4 P, N;
  REAL a, c;

  INT ret = raycast(&P,
                    &N,
                    &a,
                    &c,
                    coord,
                    cam,
                    prm,
                    seq);

  buf[ind].x = P.x;
  buf[ind].y = P.y;
  buf[ind].z = P.z;
  buf[ind].nx = N.x;
  buf[ind].ny = N.y;
  buf[ind].nz = N.z;
  buf[ind].a = ret ? NAN : a;
  buf[ind].c = c;

  REAL4 color = shade(buf, ind, cam, Ls, numLights);

  write_imagef(img, coord, color);
}

__kernel void lyaprendergl(__global LyapPoint *buf,
                           __write_only image2d_t img,
                           UINT start,
                           UINT base,
                           __constant LyapCamLight *cam,
                           __constant LyapCamLight *Ls,
                           UINT numLights)
{
  int2 coord;
  UINT ind;
  locate(&coord, &ind, start, base, cam);

  REAL4 color = shade(buf, ind, cam, Ls, numLights);
  write_imagef(img, coord, color);
}


__kernel void lyaprenderrgba(__global LyapPoint *buf,
                             __global RGBA *img,
                             UINT start,
                             UINT base,
                             __constant LyapCamLight *cam,
                             __constant LyapCamLight *Ls,
                             UINT numLights)
{
  int2 coord;
  UINT ind;
  locate(&coord, &ind, start, base, cam);

  REAL4 color = shade(buf, ind, cam, Ls, numLights);
  img[ind].r = (uchar)(255.0 * color.x);
  img[ind].g = (uchar)(255.0 * color.y);
  img[ind].b = (uchar)(255.0 * color.z);
  img[ind].a = (uchar)(255.0 * color.w);
}
