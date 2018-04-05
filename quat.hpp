// -*- mode: cuda; -*-

#include <cmath>

#ifndef MAXFLOAT
#  define MAXFLOAT	3.40282347e+38F
#endif

#ifndef __LYAP_QUAT_HPP__
#define __LYAP_QUAT_HPP__

static const double QUAT_EPSILON = 1e-6;
static const double QUAT_PI = 3.14159265358979323846264338327950288;

#define BOTH __device__ __host__

template<class T,class T3,class T4> class QUAT : public T4 {

public:

    BOTH static T fabs(T x)
    {
        return x < 0 ? -x : x;
    }

    BOTH static T logf(T x)
    {
        return std::log(static_cast<T>(x));
    }

    BOTH static T sqrt(T x)
    {
        return std::sqrt(static_cast<T>(x));
    }

    BOTH static T acos(T x)
    {
        return std::acos(static_cast<T>(x));
    }

    BOTH static T sin(T x)
    {
        return std::sin(static_cast<T>(x));
    }

    BOTH static T cos(T x)
    {
        return std::cos(static_cast<T>(x));
    }


    BOTH QUAT<T,T3,T4>(const QUAT<T,T3,T4> &that)
    {
        this->x = that.x;
        this->y = that.y;
        this->z = that.z;
        this->w = that.w;
    }

    BOTH QUAT<T,T3,T4>(T _x, T _y, T _z, T _w)
    {
        this->x = _x;
        this->y = _y;
        this->z = _z;
        this->w = _w;
    }

    BOTH QUAT<T,T3,T4>()
    {
        this->x = this->y = this->z = 0;
        this->w = 1.0;
    }

    BOTH QUAT<T,T3,T4> operator*(const QUAT<T,T3,T4> &that)
    {
        return QUAT<T,T3,T4>(this->w*that.x + this->x*that.w + this->y*that.z - this->z*that.y,
                             this->w*that.y - this->x*that.z + this->y*that.w + this->z*that.x,
                             this->w*that.z + this->x*that.y - this->y*that.x + this->z*that.w,
                             this->w*that.w - this->x*that.x - this->y*that.y - this->z*that.z);
    }


    BOTH QUAT<T,T3,T4> operator-(const QUAT<T,T3,T4> &that)
    {
        return QUAT<T,T3,T4>(this->x - that.x,
                             this->y - that.y,
                             this->z - that.z,
                             this->w - that.w);
    }

    BOTH QUAT<T,T3,T4> operator*(T fac)
    {
        return QUAT<T,T3,T4>(fac * this->x,
                             fac * this->y,
                             fac * this->z,
                             fac * this->w);
    }

    BOTH QUAT<T,T3,T4> operator/(T divisor)
    {
        return QUAT<T,T3,T4>(this->x / divisor,
                             this->y / divisor,
                             this->z / divisor,
                             this->w / divisor);
    }

    BOTH QUAT<T,T3,T4> conjugated()
    {
        return QUAT<T,T3,T4>(-this->x, -this->y, -this->z, this->w);
    }

    BOTH void conjugate()
    {
        this->x = -this->x;
        this->y = -this->y;
        this->z = -this->z;
    }

    BOTH T mag2()
    {
        return this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w;
    }

    BOTH T mag()
    {
        return this->sqrt(this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
    }

    BOTH QUAT<T,T3,T4> normalized()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
        if ( mag2 < QUAT_EPSILON ) {
            return QUAT<T,T3,T4>();
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - QUAT_EPSILON) && mag2 < (1.0f + QUAT_EPSILON))) {
            return QUAT<T,T3,T4>(*this);
        }
        else {
            mag2 = this->sqrt(mag2);
            return QUAT<T,T3,T4>(this->x * mag2,
                                 this->y * mag2,
                                 this->z * mag2,
                                 this->w * mag2);
        }
    }

    BOTH void normalize()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);

        if ( mag2 < QUAT_EPSILON ) {
            this->x = this->y = this->z = 0;
            this->w = 1.0;
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - QUAT_EPSILON) && mag2 < (1.0f + QUAT_EPSILON))) {
            // nop
        }
        else {
            mag2 = QUAT<T,T3,T4>::sqrt(mag2);
            this->x *= mag2;
            this->y *= mag2;
            this->z *= mag2;
            this->w *= mag2;
        }
    }

    BOTH T operator%(const QUAT<T,T3,T4> &that)
    {
        return this->x*that.x + this->y*that.y + this->z*that.z + this->w*that.w;
    }

    BOTH QUAT<T,T3,T4> nlerp(const QUAT<T,T3,T4> &that, T t)
    {
        QUAT<T,T3,T4> r;
        if (t==0.0 || (t<QUAT_EPSILON)) {
            r = *this;
        }
        else if (t==1.0f || (t>1.0f-QUAT_EPSILON)) {
            r = that;
        }
        else {
            T dot = *this % that;
            T tA = dot>=0 ? t : -t;
            T tI = 1.0f-t;
            r.x = this->x*tI + that.x*tA;
            r.y = this->y*tI + that.y*tA;
            r.z = this->z*tI + that.z*tA;
            r.w = this->w*tI + that.w*tA;
            r.normalize();
        }
        return r;
    }

    BOTH QUAT<T,T3,T4>(VEC3<T,T3> p, VEC3<T,T3> q, T scale)
    {
        T cosa = p % q;

        if (cosa < -1.0)
            cosa = -1.0;

        else if (cosa > 1.0)
            cosa = 1.0;

        if (cosa == 0 || (cosa >= -QUAT_EPSILON && cosa <= QUAT_EPSILON) ) {
            this->x = this->y = this->z = 0;
            this->w = 1.0;
            return;
        }

        T ang = this->acos(cosa);

        VEC3<T,T3> axis;
        axis = p * q;

        T sinang1 = this->sin(ang*0.5*scale)/this->sin(ang);
        this->x = axis.x*sinang1;
        this->y = axis.y*sinang1;
        this->z = axis.z*sinang1;
        this->w = this->cos(ang*0.5*scale);
    }

    BOTH QUAT<T,T3,T4>(VEC3<T,T3> p, VEC3<T,T3> q)
    {
        QUAT<T,T3,T4>(p, q, 1.0);
    }

    BOTH QUAT<T,T3,T4>(VEC3<T,T3> axis, T ang, bool inDegrees)
    {
        ang = inDegrees ? (ang*QUAT_PI/360.0f) : (ang*0.5f);

        T s = QUAT<T,T3,T4>::sin(ang);

        this->x = axis.x*s;
        this->y = axis.y*s;
        this->z = axis.z*s;
        this->w = QUAT<T,T3,T4>::cos(ang);
        this->normalize();
    }

    BOTH bool operator==(const QUAT<T,T3,T4> &that)
    {
        if(this->x != that.x) return 0;
        if(this->y != that.y) return 0;
        if(this->z != that.z) return 0;
        if(this->w != that.w) return 0;
        return 1;
    }

    BOTH bool nearlyEquals(const QUAT<T,T3,T4> &that)
    {
        if(this->x == that.x &&
           this->y == that.y &&
           this->z == that.z &&
           this->w == that.w) return 1;

        T diff;
        diff = this->x - that.x;
        if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

        diff = this->y - that.y;
        if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

        diff = this->z - that.z;
        if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

        diff = this->w - that.w;
        if(diff>QUAT_EPSILON || diff<-QUAT_EPSILON) return 0;

        return 1;
    }

    BOTH VEC3<T,T3> zvec(QUAT<T,T3,T4> p)
    {
        return VEC3<T,T3>(2.0f*(this->y*this->w + this->z*this->x),
                          2.0f*(this->z*this->y - this->x*this->w),
                          this->z*this->z - this->y*this->y - this->x*this->x + this->w*this->w);
    }

    BOTH VEC3<T,T3> transform(const VEC3<T,T3> &v)
    {
        return VEC3<T,T3>(this->w*this->w*v.x + 2*this->y*this->w*v.z - 2*this->z*this->w*v.y + this->x*this->x*v.x + 2*this->y*this->x*v.y + 2*this->z*this->x*v.z - this->z*this->z*v.x - this->y*this->y*v.x,
                          2*this->x*this->y*v.x + this->y*this->y*v.y + 2*this->z*this->y*v.z + 2*this->w*this->z*v.x - this->z*this->z*v.y + this->w*this->w*v.y - 2*this->x*this->w*v.z - this->x*this->x*v.y,
                          2*this->x*this->z*v.x + 2*this->y*this->z*v.y + this->z*this->z*v.z - 2*this->w*this->y*v.x - this->y*this->y*v.z + 2*this->w*this->x*v.y - this->x*this->x*v.z + this->w*this->w*v.z);
    }

#define _M44(row,col)  mat[col*4+row]

    BOTH void inv_to_matrix4x4(T mat[], QUAT<T,T3,T4> &_that, T cx, T cy, T cz)
    {
        QUAT<T,T3,T4> that = _that.normalized();
        that.x = -that.x;
        that.y = -that.y;
        that.z = -that.z;

        T xx = that.x * that.x;
        T yy = that.y * that.y;
        T zz = that.z * that.z;
        T ww = that.w * that.w;

        _M44(0,0) = xx - yy - zz + ww;
        _M44(1,1) = -xx + yy - zz + ww;
        _M44(2,2) = -xx - yy + zz + ww;

        T t1 = that.x*that.y;
        T t2 = that.z*that.w;

        _M44(0,1) = 2.0f * (t1 + t2);
        _M44(1,0) = 2.0f * (t1 - t2);

        t1 = that.x*that.z;
        t2 = that.y*that.w;
        _M44(0,2) = 2.0f * (t1 - t2);
        _M44(2,0) = 2.0f * (t1 + t2);

        t1 = that.y*that.z;
        t2 = that.x*that.w;
        _M44(1,2) = 2.0f * (t1 + t2);
        _M44(2,1) = 2.0f * (t1 - t2);

        _M44(0,3) = cx - cx * _M44(0,0) - cy * _M44(0,1) - cz * _M44(0,2);
        _M44(1,3) = cy - cx * _M44(1,0) - cy * _M44(1,1) - cz * _M44(1,2);
        _M44(2,3) = cz - cx * _M44(2,0) - cy * _M44(2,1) - cz * _M44(2,2);
        _M44(3,0) = _M44(3,1) = _M44(3,2) = 0;
        _M44(3,3) = 1.0f;
    }

    BOTH void to_matrix4x4(T mat[], QUAT<T,T3,T4> &_that, T cx, T cy, T cz)
    {
        QUAT<T,T3,T4> that = _that.normalized();

        T xx = that.x * that.x;
        T yy = that.y * that.y;
        T zz = that.z * that.z;
        T ww = that.w * that.w;

        _M44(0,0) = xx - yy - zz + ww;
        _M44(1,1) = -xx + yy - zz + ww;
        _M44(2,2) = -xx - yy + zz + ww;

        T t1 = that.x*that.y;
        T t2 = that.z*that.w;

        _M44(0,1) = 2.0f * (t1 + t2);
        _M44(1,0) = 2.0f * (t1 - t2);

        t1 = that.x*that.z;
        t2 = that.y*that.w;
        _M44(0,2) = 2.0f * (t1 - t2);
        _M44(2,0) = 2.0f * (t1 + t2);

        t1 = that.y*that.z;
        t2 = that.x*that.w;
        _M44(1,2) = 2.0f * (t1 + t2);
        _M44(2,1) = 2.0f * (t1 - t2);

        _M44(0,3) = cx - cx * _M44(0,0) - cy * _M44(0,1) - cz * _M44(0,2);
        _M44(1,3) = cy - cx * _M44(1,0) - cy * _M44(1,1) - cz * _M44(1,2);
        _M44(2,3) = cz - cx * _M44(2,0) - cy * _M44(2,1) - cz * _M44(2,2);
        _M44(3,0) = _M44(3,1) = _M44(3,2) = 0;
        _M44(3,3) = 1.0f;
    }

    BOTH void print()
    {
#if defined(NSLog)
        NSLog(@"X{%.2f, %.2f, %.2f, %.2f}\n", this->x, this->y, this->z, this->w);
#else
        printf("{%.2f, %.2f, %.2f, %.2f}\n", this->x, this->y, this->z, this->w);
#endif
    }

};

#endif
