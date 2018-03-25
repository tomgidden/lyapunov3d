
#include <vector_types.h>
#include <vector_functions.h>

#define static /*static*/

#define LOG(x) logf(x)
#define FABS(x) fabs(x)
#define POW(x,y) powf(x,y)
#define SQRT(x) sqrtf(x)
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define ACOS(x) acosf(x)

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795029L
#endif

typedef int INT;
typedef int2 INT2;
typedef unsigned int UINT;
typedef float4 QUAT;
typedef float REAL;
typedef float4 REAL4;
typedef bool BOOL;

#define QUAT_EPSILON 0.00001f

#define _M44(row,col)  mat[col*4+row]




#  define VEC_EXTRAPOLATE(Q,P,V,t) { Q = REAL4_extrapolate(P, V, t); }
#  define VEC_SCALE(Q,P,s) { Q = REAL4_scale(P, (float)s); }
#  define VEC_DIV(Q,P,s) { Q = REAL4_div(P, (float)s); }
#  define VEC_ADD(R,P,Q) { R = REAL4_add(P, Q); }
#  define VEC_SUB(R,P,Q) { R = REAL4_sub(P, Q); }
#  define VEC_MAG2(R,V) { R = REAL4_mag2(V); }
#  define VEC_CLAMP(R,V) { R = REAL4_clamp(V); }
#  define NUM_CLAMP(R,N) { R = REAL_clamp(N); }
#  define VEC_DOT(R,P,Q) { R = REAL4_dot(P, Q); }
#  define RETURN(X) return X
#  define VEC_MAG(R,V) { R = REAL4_mag(V); }
#  define NORMALIZE(X) { X = REAL4_normalize(X); }

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

__device__ __host__ static __inline__ REAL4 REAL4_init(REAL _x, REAL _y, REAL _z)
{
    return make_float4(_x, _y, _z, 0);
}

__device__ __host__ static __inline__ REAL4 REAL4_init0()
{
    return REAL4_init(0,0,0);
}

__device__ __host__ static REAL4 REAL4_spaceball_soften(REAL x, REAL y, REAL z, REAL lim, REAL range, REAL scale)
{
    REAL4 r;

    if(x<=-lim) r.x = scale * (x+lim)/(range-lim);
    else if(x>=lim) r.x = scale * (x-lim)/(range-lim);
    else r.x = 0;

    if(y<=-lim) r.y = scale * (y+lim)/(range-lim);
    else if(y>=lim) r.y = scale * (y-lim)/(range-lim);
    else r.y = 0;

    if(z<=-lim) r.z = scale * (z+lim)/(range-lim);
    else if(z>=lim) r.z = scale * (z-lim)/(range-lim);
    else r.z = 0;

    r.w = 0;

    return r;
}

__device__ __host__ static __inline__ REAL4 REAL4_add(REAL4 p, REAL4 q)
{
    return REAL4_init(p.x+q.x, p.y+q.y, p.z+q.z);
}

__device__ __host__ static __inline__ REAL4 REAL4_sub(REAL4 p, REAL4 q)
{
    return REAL4_init(p.x-q.x, p.y-q.y, p.z-q.z);
}

__device__ __host__ static __inline__ REAL REAL4_dot(REAL4 p, REAL4 q)
{
    return p.x*q.x + p.y*q.y + p.z*q.z;
}

__device__ __host__ static __inline__ float REAL4P_dot(REAL4 *p, REAL4 *q)
{
    return p->x*q->x + p->y*q->y + p->z*q->z;
}

__device__ __host__ static __inline__ REAL4 REAL4_scale(REAL4 v, REAL s)
{
    return REAL4_init(v.x*s, v.y*s, v.z*s);
}

__device__ __host__ static __inline__ REAL4 REAL4_div(REAL4 v, REAL s)
{
    return REAL4_init(v.x/s, v.y/s, v.z/s);
}

__device__ __host__ static __inline__ REAL4 REAL4_cross(REAL4 p, REAL4 q)
{
    return REAL4_init(p.y*q.z-p.z*q.y, p.z*q.x-p.x*q.z, p.x*q.y-p.y*q.x);
}

__device__ __host__ static __inline__ void REAL4P_cross(REAL4 *r, REAL4 *p, REAL4 *q)
{
    r->x = p->y*q->z - p->z*q->y;
    r->y = p->z*q->x - p->x*q->z;
    r->z = p->x*q->y - p->y*q->x;
}

__device__ __host__ static __inline__ REAL REAL4_mag2(REAL4 v)
{
    return (v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ __host__ static __inline__ REAL REAL4_mag(REAL4 v)
{
    return SQRT(REAL4_mag2(v));
}

__device__ __host__ static REAL4 REAL4_normalize(REAL4 p)
{
    REAL mod = p.x*p.x + p.y*p.y + p.z*p.z;
    if(mod < QUAT_EPSILON) {
        p.x = p.y = p.z = 0;
    }
    else if (mod == 1.0f) {
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
    }
    else {
        mod = 1.0f/SQRT(mod);
        p.x *= mod;
        p.y *= mod;
        p.z *= mod;
    }
    return p;
}

__device__ __host__ static __inline__ void REAL4P_normalize(REAL4 *r, REAL4 *p)
{
    REAL mod = p->x*p->x + p->y*p->y + p->z*p->z;
    if(mod < QUAT_EPSILON) {
        r->x = r->y = r->z = 0;
    }
    else if (mod == 1.0f) {
        if(r!=p) {
            r->x = p->x;
            r->y = p->y;
            r->z = p->z;
        }
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
        r->x = p->x;
        r->y = p->y;
        r->z = p->z;
    }
    else {
        mod = 1.0f/SQRT(mod);
        r->x = p->x * mod;
        r->y = p->y * mod;
        r->z = p->z * mod;
    }
}

__device__ __host__ static __inline__ void REAL4P_normalize_inplace(REAL4 *r)
{
    REAL mod = r->x*r->x + r->y*r->y + r->z*r->z;
    if(mod < QUAT_EPSILON) {
        r->x = r->y = r->z = 0;
    }
    else if (mod == 1.0f) {
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
    }
    else {
        mod = 1.0f/SQRT(mod);
        r->x *= mod;
        r->y *= mod;
        r->z *= mod;
    }
}

__device__ __host__ static __inline__ REAL4 REAL4_dominant(REAL4 p)
{
    REAL ax = p.x<0 ? -p.x : p.x;
    REAL ay = p.y<0 ? -p.y : p.y;
    REAL az = p.z<0 ? -p.z : p.z;
    if(ax > ay)
        if(ax > az)
            p.y = p.z = 0;
        else
            p.x = p.y = 0;
    else
        if(ay > az)
            p.x = p.z = 0;
        else
            p.x = p.y = 0;
    return p;
}

__device__ __host__ static __inline__ REAL4 REAL4_clamp(REAL4 v)
{
    return REAL4_init(v.x<0.0 ? 0.0 : v.x>1.0 ? 1.0 : v.x,
                      v.y<0.0 ? 0.0 : v.y>1.0 ? 1.0 : v.y,
                      v.z<0.0 ? 0.0 : v.z>1.0 ? 1.0 : v.z);
}

__device__ __host__ static __inline__ REAL REAL_clamp(REAL n)
{
    return n<0 ? 0 : n>1.0 ? 1.0 : n;
}

__device__ __host__ static __inline__ REAL REAL_clamp_ab(REAL n, REAL a, REAL b)
{
    return n<a ? a : n>b ? b : n;
}

__device__ __host__ static __inline__ REAL4 REAL4_extrapolate(REAL4 v, REAL4 d, REAL f)
{
    return REAL4_init(v.x+d.x*f, v.y+d.y*f, v.z+d.z*f);
}

__device__ __host__ static __inline__ int REAL4_in_lyap_space(REAL4 v)
{
    return
        (v.x>=0) && (v.x<=4.0) &&
        (v.y>=0) && (v.y<=4.0) &&
        (v.z>=0) && (v.z<=4.0);
}

__device__ __host__ static REAL REAL4_sphere3d(REAL4 P)
{
    REAL4 C = REAL4_init(2,2,2);
    REAL4 H = REAL4_sub(P, C);
    REAL m = REAL4_mag(H)-1.5;
    return m;
}

__device__ __host__ static REAL REAL4_lyap3d(REAL4 P, int *seq, int settle, int accum)
{
    REAL abc[3];
    abc[0] = P.x;
    abc[1] = P.y;
    abc[2] = P.z;

    int *seqp; // Position in the sequence loop
    int n; // Iteration counter
    REAL r; // Iteration value
    REAL v = 0.5;
    REAL l = 0;

    // Initialise for this pixel
    seqp = seq;

    // Settle by running the iteration without accumulation
    for(n = 0; n < settle; n++) {
        r = abc[*seqp++];
        if(*seqp==-1) seqp = seq;
        v = r * v * (1.0 - v);
    }

    if(FABS(v-0.5) >= 1e-10 ) {
        // Now calculate the value by running the iteration with accumulation
        for(n = 0; n < accum; n++) {
            r = abc[*seqp++];
            if(*seqp==-1) seqp = seq;
            v = r * v * (1.0 - v);
            r = r - 2.0 * r * v;
            l += LOG(r<0 ? -r : r);
        }
    }

    return l;
}


////////////////////////////////////////////////////////////////////////////



__device__ __host__ static QUAT QUAT_multiply(QUAT p, QUAT q)
{
    QUAT r;
    r.x = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y;
    r.y = p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x;
    r.z = p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w;
    r.w = p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z;
    return r;
}

__device__ __host__ static void QUATP_multiply(QUAT *r, QUAT *p, QUAT *q)
{
    QUAT t;
    if (p == r) {
        t = *p;
        p = &t;
    }
    else if (q == r) {
        t = *q;
        q = &t;
    }

    r->x = p->w*q->x + p->x*q->w + p->y*q->z - p->z*q->y;
    r->y = p->w*q->y - p->x*q->z + p->y*q->w + p->z*q->x;
    r->z = p->w*q->z + p->x*q->y - p->y*q->x + p->z*q->w;
    r->w = p->w*q->w - p->x*q->x - p->y*q->y - p->z*q->z;
}

__device__ __host__ static QUAT QUAT_add(QUAT p, QUAT q)
{
    QUAT r;
    r.x = p.x + q.x;
    r.y = p.y + q.y;
    r.z = p.z + q.z;
    r.w = p.w + q.w;
    return r;
}

__device__ __host__ static void QUATP_add(QUAT *r, QUAT *p, QUAT *q)
{
    QUAT t;
    if (p == r) {
        t = *p;
        p = &t;
    }
    else if (q == r) {
        t = *q;
        q = &t;
    }

    r->x = p->x + q->x;
    r->y = p->y + q->y;
    r->z = p->z + q->z;
    r->w = p->w + q->w;
}

__device__ __host__ static QUAT QUAT_sub(QUAT p, QUAT q)
{
    QUAT r;
    r.x = p.x - q.x;
    r.y = p.y - q.y;
    r.z = p.z - q.z;
    r.w = p.w - q.w;
    return r;
}

__device__ __host__ static void QUATP_sub(QUAT *r, QUAT *p, QUAT *q)
{
    QUAT t;
    if (p == r) {
        t = *p;
        p = &t;
    }
    else if (q == r) {
        t = *q;
        q = &t;
    }

    r->x = p->x - q->x;
    r->y = p->y - q->y;
    r->z = p->z - q->z;
    r->w = p->w - q->w;
}

__device__ __host__ static QUAT QUAT_scale(QUAT p, REAL fac)
{
    QUAT r;
    r.x = fac * p.x;
    r.y = fac * p.y;
    r.z = fac * p.z;
    r.w = fac * p.w;
    return r;
}

__device__ __host__ static void QUATP_scale(QUAT *r, QUAT *p, REAL fac)
{
    r->x = fac * p->x;
    r->y = fac * p->y;
    r->z = fac * p->z;
    r->w = fac * p->w;
}

__device__ __host__ static QUAT QUAT_conjugate(QUAT p)
{
    QUAT r;
    r.x = -p.x;
    r.y = -p.y;
    r.z = -p.z;
    r.w =  p.w;
    return r;
}

__device__ __host__ static void QUATP_conjugate(QUAT *r, QUAT *p)
{
    r->x = -p->x;
    r->y = -p->y;
    r->z = -p->z;
    r->w =  p->w;
}

__device__ __host__ static REAL QUAT_modulus2(QUAT p)
{
    return p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w;
}

__device__ __host__ static REAL QUATP_modulus2(QUAT *p)
{
    return p->x*p->x + p->y*p->y + p->z*p->z + p->w*p->w;
}

__device__ __host__ static REAL QUAT_modulus(QUAT p)
{
    return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w);
}

__device__ __host__ static REAL QUATP_modulus(QUAT *p)
{
    return sqrtf(p->x*p->x + p->y*p->y + p->z*p->z + p->w*p->w);
}

__device__ __host__ static QUAT QUAT_normalize(QUAT p)
{
    REAL mod = p.x*p.x + p.y*p.y + p.z*p.z + p.w*p.w;
    if(mod < QUAT_EPSILON) {
        p.x = p.y = p.z = 0;
        p.w = 1.0f;
    }
    else if (mod == 1.0f) {
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
    }
    else {
        mod = 1.0f/sqrtf(mod);
        p.x *= mod;
        p.y *= mod;
        p.z *= mod;
        p.w *= mod;
    }

    return p;
}

__device__ __host__ static void QUATP_normalize(QUAT *r, QUAT *p)
{
    REAL mod = p->x*p->x + p->y*p->y + p->z*p->z + p->w*p->w;
    if(mod < QUAT_EPSILON) {
        r->x = r->y = r->z = 0;
        r->w = 1.0f;
    }
    else if (mod == 1.0f) {
        if(r!=p) {
            r->x = p->x;
            r->y = p->y;
            r->z = p->z;
            r->w = p->w;
        }
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
        r->x = p->x;
        r->y = p->y;
        r->z = p->z;
        r->w = p->w;
    }
    else {
        mod = 1.0f/SQRT(mod);
        r->x = p->x * mod;
        r->y = p->y * mod;
        r->z = p->z * mod;
        r->w = p->w * mod;
    }
}

__device__ __host__ static void QUATP_normalize_inplace(QUAT *r)
{
    REAL mod = r->x*r->x + r->y*r->y + r->z*r->z + r->w*r->w;
    if(mod < QUAT_EPSILON) {
        r->x = r->y = r->z = 0;
        r->w = 1.0f;
    }
    else if (mod == 1.0f) {
    }
    else if (mod > (1.0f-QUAT_EPSILON) && mod < (1.0f+QUAT_EPSILON)) {
    }
    else {
        mod = 1.0f/SQRT(mod);
        r->x *= mod;
        r->y *= mod;
        r->z *= mod;
        r->w *= mod;
    }
}

__device__ __host__ static REAL QUAT_dot(QUAT p, QUAT q)
{
    return p.x*q.x + p.y*q.y + p.z*q.z + p.w*q.w;
}

__device__ __host__ static REAL QUATP_dot(QUAT *p, QUAT *q)
{
    return p->x*q->x + p->y*q->y + p->z*q->z + p->w*q->w;
}

__device__ __host__ static QUAT QUAT_nlerp(QUAT p, QUAT q, REAL t)
{
    QUAT r;
    if (t==0.0 || (t<QUAT_EPSILON)) {
        r = p;
    }
    else if (t==1.0f || (t>1.0f-QUAT_EPSILON)) {
        r = q;
    }
    else {
        REAL dot = QUAT_dot(p, q);
        REAL tA = dot>=0 ? t : -t;
        REAL tI = 1.0f-t;
        r.x = p.x*tI + q.x*tA;
        r.y = p.y*tI + q.y*tA;
        r.z = p.z*tI + q.z*tA;
        r.w = p.w*tI + q.w*tA;
        QUATP_normalize_inplace(&r);
    }
    return r;
}

__device__ __host__ static void QUATP_nlerp(QUAT *r, QUAT *p, QUAT *q, REAL t)
{
    if (t==0.0 || (t<QUAT_EPSILON)) {
        if(r!=p)
            *r = *p;
    }
    else if (t==1.0f || (t>1.0f-QUAT_EPSILON)) {
        if(r!=q)
            *r = *q;
    }
    else {
        REAL dot = QUATP_dot(p, q);
        REAL tA = dot>=0 ? t : -t;
        REAL tI = 1.0f-t;
        r->x = p->x*tI + q->x*tA;
        r->y = p->y*tI + q->y*tA;
        r->z = p->z*tI + q->z*tA;
        r->w = p->w*tI + q->w*tA;
        QUATP_normalize_inplace(r);
    }
}

__device__ __host__ static QUAT QUAT_fromVectors_scaleangle(REAL4 p, REAL4 q, REAL scale)
{
    REAL cosa = REAL4_dot(p, q);

    if (cosa < -1.0)
        cosa = -1.0;
    else if (cosa > 1.0)
        cosa = 1.0;


    if (cosa == 0 || (cosa >= -QUAT_EPSILON && cosa <= QUAT_EPSILON) ) {
        QUAT nullQ = {0,0,0,1};
        return nullQ;
    }

    REAL ang = ACOS(cosa);

    REAL4 axis;
    axis = REAL4_cross(p, q);

    QUAT r;
    REAL sinang1 = SIN(ang*0.5*scale)/SIN(ang);
    r.x = axis.x*sinang1;
    r.y = axis.y*sinang1;
    r.z = axis.z*sinang1;
    r.w = COS(ang*0.5*scale);

    return r;
}

__device__ __host__ static QUAT QUAT_fromVectors(REAL4 p, REAL4 q)
{
    return QUAT_fromVectors_scaleangle(p, q, 1.0);
}

__device__ __host__ static void QUATP_fromVectors_scaleangle(QUAT *r, REAL4 *_p, REAL4 *_q, REAL scale)
{
    REAL cosa = REAL4_dot(*_p, *_q);

    if(cosa<-1.0) cosa=-1.0;
    else if(cosa>1.0) cosa=1.0;

    if (cosa == 0 || (cosa >= -QUAT_EPSILON && cosa <= QUAT_EPSILON) ) {
        r->x = r->y = r->z = 0;
        r->w = 1;
        return;
    }

    REAL ang = ACOS(cosa);

    REAL4 axis;
    axis = REAL4_cross(*_p, *_q);

    REAL sinang1 = SIN(ang*0.5*scale)/SIN(ang);
    r->x = axis.x*sinang1;
    r->y = axis.y*sinang1;
    r->z = axis.z*sinang1;
    r->w = COS(ang*0.5*scale);
}

__device__ __host__ static void QUATP_fromVectors(QUAT *r, REAL4 *_p, REAL4 *_q)
{
    QUATP_fromVectors_scaleangle(r, _p, _q, 1.0);
}

__device__ __host__ static QUAT QUAT_fromAxisAngleV(REAL axis[], REAL ang, BOOL inDegrees)
{
    QUAT r;
    ang = inDegrees ? (ang*M_PI/360.0f) : (ang*0.5f);
    REAL s = SIN(ang);
    r.x = axis[0]*s;
    r.y = axis[1]*s;
    r.z = axis[2]*s;
    r.w = COS(ang);
    QUATP_normalize_inplace(&r);
    return r;
}

__device__ __host__ static void QUATP_fromAxisAngleV(QUAT *r, REAL axis[], REAL ang, BOOL inDegrees)
{
    ang = inDegrees ? (ang*M_PI/360.0f) : (ang*0.5f);
    REAL s = SIN(ang);
    r->x = axis[0]*s;
    r->y = axis[1]*s;
    r->z = axis[2]*s;
    r->w = COS(ang);
    QUATP_normalize_inplace(r);
}

__device__ __host__ static QUAT QUAT_fromAxisAngle(REAL ax, REAL ay, REAL az, REAL ang, BOOL inDegrees)
{
    QUAT r;
    ang = inDegrees ? (ang*M_PI/360.0f) : (ang*0.5f);
    REAL s = SIN(ang);
    r.x = ax*s;
    r.y = ay*s;
    r.z = az*s;
    r.w = COS(ang);
    QUATP_normalize_inplace(&r);
    return r;
}

__device__ __host__ static void QUATP_fromAxisAngle(QUAT *r, REAL ax, REAL ay, REAL az, REAL ang, BOOL inDegrees)
{
    ang = inDegrees ? (ang*M_PI/360.0f) : (ang*0.5f);
    REAL s = SIN(ang);
    r->x = ax*s;
    r->y = ay*s;
    r->z = az*s;
    r->w = COS(ang);
    QUATP_normalize_inplace(r);
}

__device__ __host__ static BOOL QUAT_equals(QUAT p, QUAT q)
{
    if(p.x != q.x) return 0;
    if(p.y != q.y) return 0;
    if(p.z != q.z) return 0;
    if(p.w != q.w) return 0;
    return 1;
}

__device__ __host__ static BOOL QUATP_equals(QUAT *p, QUAT *q)
{
    if(p==q) return 1;
    if(p->x != q->x) return 0;
    if(p->y != q->y) return 0;
    if(p->z != q->z) return 0;
    if(p->w != q->w) return 0;
    return 1;
}

__device__ __host__ static BOOL QUAT_equalsNearly(QUAT p, QUAT q)
{
    if(p.x == q.x &&
       p.y == q.y &&
       p.z == q.z &&
       p.w == q.w) return 1;

    REAL diff;
    diff = p.x - q.x;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p.y - q.y;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p.z - q.z;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p.w - q.w;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    return 1;
}

__device__ __host__ static BOOL QUATP_equalsNearly(QUAT *p, QUAT *q)
{
    if(p==q) return 1;

    REAL diff;
    diff = p->x - q->x;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p->y - q->y;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p->z - q->z;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    diff = p->w - q->w;
    if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

    return 1;
}

__device__ __host__ static void QUAT_zvec(REAL vec[], QUAT p)
{
    vec[0] = 2.0f*(p.y*p.w + p.z*p.x);
    vec[1] = 2.0f*(p.z*p.y - p.x*p.w);
    vec[2] = p.z*p.z - p.y*p.y - p.x*p.x + p.w*p.w;
}

__device__ __host__ static void QUATP_zvec(REAL vec[], QUAT *p)
{
    vec[0] = 2.0f*(p->y*p->w + p->z*p->x);
    vec[1] = 2.0f*(p->z*p->y - p->x*p->w);
    vec[2] = p->z*p->z - p->y*p->y - p->x*p->x + p->w*p->w;
}

__device__ __host__ static REAL4 QUAT_transformREAL4(REAL4 p, QUAT q)
{
    REAL4 r;
    r.x = q.w*q.w*p.x + 2*q.y*q.w*p.z - 2*q.z*q.w*p.y + q.x*q.x*p.x + 2*q.y*q.x*p.y + 2*q.z*q.x*p.z - q.z*q.z*p.x - q.y*q.y*p.x;
    r.y = 2*q.x*q.y*p.x + q.y*q.y*p.y + 2*q.z*q.y*p.z + 2*q.w*q.z*p.x - q.z*q.z*p.y + q.w*q.w*p.y - 2*q.x*q.w*p.z - q.x*q.x*p.y;
    r.z = 2*q.x*q.z*p.x + 2*q.y*q.z*p.y + q.z*q.z*p.z - 2*q.w*q.y*p.x - q.y*q.y*p.z + 2*q.w*q.x*p.y - q.x*q.x*p.z + q.w*q.w*p.z;
    r.w = 1;
    return r;
}

__device__ __host__ static REAL4 QUAT_transformREAL4P(REAL4 *p, QUAT q)
{
    REAL4 r;
    r.x = q.w*q.w*p->x + 2*q.y*q.w*p->z - 2*q.z*q.w*p->y + q.x*q.x*p->x + 2*q.y*q.x*p->y + 2*q.z*q.x*p->z - q.z*q.z*p->x - q.y*q.y*p->x;
    r.y = 2*q.x*q.y*p->x + q.y*q.y*p->y + 2*q.z*q.y*p->z + 2*q.w*q.z*p->x - q.z*q.z*p->y + q.w*q.w*p->y - 2*q.x*q.w*p->z - q.x*q.x*p->y;
    r.z = 2*q.x*q.z*p->x + 2*q.y*q.z*p->y + q.z*q.z*p->z - 2*q.w*q.y*p->x - q.y*q.y*p->z + 2*q.w*q.x*p->y - q.x*q.x*p->z + q.w*q.w*p->z;
    r.w = 1;
    return r;
}

__device__ __host__ static void QUAT_transformREAL4P_P(REAL4 *r, REAL4 *p, QUAT q)
{
    r->x = q.w*q.w*p->x + 2*q.y*q.w*p->z - 2*q.z*q.w*p->y + q.x*q.x*p->x + 2*q.y*q.x*p->y + 2*q.z*q.x*p->z - q.z*q.z*p->x - q.y*q.y*p->x;
    r->y = 2*q.x*q.y*p->x + q.y*q.y*p->y + 2*q.z*q.y*p->z + 2*q.w*q.z*p->x - q.z*q.z*p->y + q.w*q.w*p->y - 2*q.x*q.w*p->z - q.x*q.x*p->y;
    r->z = 2*q.x*q.z*p->x + 2*q.y*q.z*p->y + q.z*q.z*p->z - 2*q.w*q.y*p->x - q.y*q.y*p->z + 2*q.w*q.x*p->y - q.x*q.x*p->z + q.w*q.w*p->z;
    r->w = 1;
}

__device__ __host__ static void QUAT_transformCoords(REAL *rx, REAL *ry, REAL *rz, QUAT q)
{
    REAL vx = *rx;
    REAL vy = *ry;
    REAL vz = *rz;
    *rx = q.w*q.w*vx + 2*q.y*q.w*vz - 2*q.z*q.w*vy + q.x*q.x*vx + 2*q.y*q.x*vy + 2*q.z*q.x*vz - q.z*q.z*vx - q.y*q.y*vx;
    *ry = 2*q.x*q.y*vx + q.y*q.y*vy + 2*q.z*q.y*vz + 2*q.w*q.z*vx - q.z*q.z*vy + q.w*q.w*vy - 2*q.x*q.w*vz - q.x*q.x*vy;
    *rz = 2*q.x*q.z*vx + 2*q.y*q.z*vy + q.z*q.z*vz - 2*q.w*q.y*vx - q.y*q.y*vz + 2*q.w*q.x*vy - q.x*q.x*vz + q.w*q.w*vz;
}

__device__ __host__ static void QUATP_transformCoords(REAL *rx, REAL *ry, REAL *rz, QUAT *q)
{
    REAL vx = *rx;
    REAL vy = *ry;
    REAL vz = *rz;
    *rx = q->w*q->w*vx + 2*q->y*q->w*vz - 2*q->z*q->w*vy + q->x*q->x*vx + 2*q->y*q->x*vy + 2*q->z*q->x*vz - q->z*q->z*vx - q->y*q->y*vx;
    *ry = 2*q->x*q->y*vx + q->y*q->y*vy + 2*q->z*q->y*vz + 2*q->w*q->z*vx - q->z*q->z*vy + q->w*q->w*vy - 2*q->x*q->w*vz - q->x*q->x*vy;
    *rz = 2*q->x*q->z*vx + 2*q->y*q->z*vy + q->z*q->z*vz - 2*q->w*q->y*vx - q->y*q->y*vz + 2*q->w*q->x*vy - q->x*q->x*vz + q->w*q->w*vz;
}

__device__ __host__ static void QUATP_inv_to_matrix4x4(REAL mat[], QUAT *p, REAL cx, REAL cy, REAL cz)
{
    QUAT q;
    QUATP_normalize(&q, p);
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;

    REAL xx = q.x * q.x;
    REAL yy = q.y * q.y;
    REAL zz = q.z * q.z;
    REAL ww = q.w * q.w;

    _M44(0,0) = xx - yy - zz + ww;
    _M44(1,1) = -xx + yy - zz + ww;
    _M44(2,2) = -xx - yy + zz + ww;

    REAL t1 = q.x*q.y;
    REAL t2 = q.z*q.w;

    _M44(0,1) = 2.0f * (t1 + t2);
    _M44(1,0) = 2.0f * (t1 - t2);

    t1 = q.x*q.z;
    t2 = q.y*q.w;
    _M44(0,2) = 2.0f * (t1 - t2);
    _M44(2,0) = 2.0f * (t1 + t2);

    t1 = q.y*q.z;
    t2 = q.x*q.w;
    _M44(1,2) = 2.0f * (t1 + t2);
    _M44(2,1) = 2.0f * (t1 - t2);

    _M44(0,3) = cx - cx * _M44(0,0) - cy * _M44(0,1) - cz * _M44(0,2);
    _M44(1,3) = cy - cx * _M44(1,0) - cy * _M44(1,1) - cz * _M44(1,2);
    _M44(2,3) = cz - cx * _M44(2,0) - cy * _M44(2,1) - cz * _M44(2,2);
    _M44(3,0) = _M44(3,1) = _M44(3,2) = 0;
    _M44(3,3) = 1.0f;
}

__device__ __host__ static void QUAT_inv_to_matrix4x4(REAL mat[], QUAT p, REAL cx, REAL cy, REAL cz)
{
    QUATP_inv_to_matrix4x4(mat, &p, cx, cy, cz);
}

__device__ __host__ static void QUATP_to_matrix4x4(REAL mat[], QUAT *p, REAL cx, REAL cy, REAL cz)
{
    QUAT q;
    QUATP_normalize(&q, p);

    REAL xx = q.x * q.x;
    REAL yy = q.y * q.y;
    REAL zz = q.z * q.z;
    REAL ww = q.w * q.w;

    _M44(0,0) = xx - yy - zz + ww;
    _M44(1,1) = -xx + yy - zz + ww;
    _M44(2,2) = -xx - yy + zz + ww;

    REAL t1 = q.x*q.y;
    REAL t2 = q.z*q.w;

    _M44(0,1) = 2.0f * (t1 + t2);
    _M44(1,0) = 2.0f * (t1 - t2);

    t1 = q.x*q.z;
    t2 = q.y*q.w;
    _M44(0,2) = 2.0f * (t1 - t2);
    _M44(2,0) = 2.0f * (t1 + t2);

    t1 = q.y*q.z;
    t2 = q.x*q.w;
    _M44(1,2) = 2.0f * (t1 + t2);
    _M44(2,1) = 2.0f * (t1 - t2);

    _M44(0,3) = cx - cx * _M44(0,0) - cy * _M44(0,1) - cz * _M44(0,2);
    _M44(1,3) = cy - cx * _M44(1,0) - cy * _M44(1,1) - cz * _M44(1,2);
    _M44(2,3) = cz - cx * _M44(2,0) - cy * _M44(2,1) - cz * _M44(2,2);
    _M44(3,0) = _M44(3,1) = _M44(3,2) = 0;
    _M44(3,3) = 1.0f;
}

__device__ __host__ static void QUAT_to_matrix4x4(REAL mat[], QUAT p, REAL cx, REAL cy, REAL cz)
{
    QUATP_to_matrix4x4(mat, &p, cx, cy, cz);
}

__device__ __host__ static void matrix4x4_print(REAL mat[])
{
#if defined(NSLog)
    NSLog(@"%.2f\t%.2f\t%.2f\t%.2f\n", mat[0], mat[4], mat[8], mat[12]);
    NSLog(@"%.2f\t%.2f\t%.2f\t%.2f\n", mat[1], mat[5], mat[9], mat[13]);
    NSLog(@"%.2f\t%.2f\t%.2f\t%.2f\n", mat[2], mat[6], mat[10], mat[14]);
    NSLog(@"%.2f\t%.2f\t%.2f\t%.2f\n", mat[3], mat[7], mat[11], mat[15]);
#else
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", mat[0], mat[4], mat[8], mat[12]);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", mat[1], mat[5], mat[9], mat[13]);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", mat[2], mat[6], mat[10], mat[14]);
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", mat[3], mat[7], mat[11], mat[15]);
#endif
}

__device__ __host__ static void QUAT_print(QUAT p)
{
#if defined(NSLog)
    NSLog(@"X{%.2f, %.2f, %.2f, %.2f}\n", p.x, p.y, p.z, p.w);
#else
    printf("{%.2f, %.2f, %.2f, %.2f}\n", p.x, p.y, p.z, p.w);
#endif
}

__device__ __host__ static void QUATP_print(QUAT *p)
{
#if defined(NSLog)
    NSLog(@"X{%.2f, %.2f, %.2f, %.2f}\n", p->x, p->y, p->z, p->w);
#else
    printf("{%.2f, %.2f, %.2f, %.2f}\n", p->x, p->y, p->z, p->w);
#endif
}
