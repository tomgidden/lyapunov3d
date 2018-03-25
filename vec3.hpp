#include <cmath>

#ifndef __LYAP_VEC3_HPP__
#define __LYAP_VEC3_HPP__

static const double VEC_EPSILON = 1e-6;

#define BOTH __device__ __host__

template<class T, class T3> class VEC3 : public T3 {

public:

    BOTH static T fabs(T x)
    {
        return x < 0 ? -x : x;
    }

    BOTH static T logf(T x)
    {
        return std::logf(static_cast<T>(x));
    }

    BOTH static T sqrt(T x)
    {
        return std::sqrt(static_cast<T>(x));
    }

    BOTH VEC3<T,T3>(T _x, T _y, T _z)
    {
        this->x = _x;
        this->y = _y;
        this->z = _z;
    }

    BOTH VEC3<T,T3>()
    {
        this->x = this->y = this->z = 0;
    }

    BOTH VEC3<T,T3> &operator+=(const VEC3<T,T3> &that)
    {
        this->x += that.x;
        this->y += that.y;
        this->z += that.z;
        return *this;
    }

    BOTH VEC3<T,T3> &operator-=(const VEC3<T,T3> &that)
    {
        this->x -= that.x;
        this->y -= that.y;
        this->z -= that.z;
        return *this;
    }

    BOTH VEC3<T,T3> &operator*=(const T scale)
    {
        this->x *= scale;
        this->y *= scale;
        this->z *= scale;
        return *this;
    }

    BOTH VEC3<T,T3> &operator/=(const T divisor)
    {
        this->x /= divisor;
        this->y /= divisor;
        this->z /= divisor;
        return *this;
    }

    BOTH inline void set(T _x, T _y, T _z)
    {
        this->x = _x;
        this->y = _y;
        this->z = _z;
    }

    BOTH VEC3<T,T3> &operator*=(const VEC3<T,T3> &that)
    {
        this->set(this->y*that.z-this->z*that.y, this->z*that.x-this->x*that.z, this->x*that.y-this->y*that.x);
        return *this;
    }

    BOTH VEC3<T,T3> operator+(const VEC3<T,T3> &that)
    {
        return VEC3<T,T3>(this->x + that.x, this->y + that.y, this->z + that.z);
    }

    BOTH VEC3<T,T3> operator-(const VEC3<T,T3> &that)
    {
        return VEC3<T,T3>(this->x - that.x, this->y - that.y, this->z - that.z);
    }

    BOTH T operator%(const VEC3<T,T3> &that)
    {
        return this->x*that.x + this->y*that.y + this->z*that.z;
    }

    BOTH VEC3<T,T3> operator*(const T scale)
    {
        return VEC3<T,T3>(this->x * scale, this->y * scale, this->z * scale);
    }

    BOTH VEC3<T,T3> operator/(const T divisor)
    {
        return VEC3<T,T3>(this->x / divisor, this->y / divisor, this->z / divisor);
    }

    BOTH VEC3<T,T3> operator*(const VEC3<T,T3> &that)
    {
        return VEC3<T,T3>(this->y*that.z-this->z*that.y, this->z*that.x-this->x*that.z, this->x*that.y-this->y*that.x);
    }

    BOTH T mag2()
    {
        return (this->x*this->x + this->y*this->y + this->z*this->z);
    }

    BOTH T mag()
    {
        return VEC3<T,T3>::sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
    }

    BOTH VEC3<T,T3> normalized()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z);
        if ( mag2 < VEC_EPSILON ) {
            return VEC3<T,T3>();
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - VEC_EPSILON) && mag2 < (1.0f + VEC_EPSILON))) {
            return *this;
        }
        else {
            return (*this) / VEC3<T,T3>::sqrt(mag2);
        }
    }

    BOTH void normalize()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z);
        if ( mag2 < VEC_EPSILON ) {
            this->x = this->y = this->z = 0;
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - VEC_EPSILON) && mag2 < (1.0f + VEC_EPSILON))) {
            // nop
        }
        else {
            mag2 = VEC3<T,T3>::sqrt(mag2);
            *this /= mag2;
        }
    }

    BOTH VEC3<T,T3> dominant()
    {
        T ax = this->x<0 ? -this->x : this->x;
        T ay = this->y<0 ? -this->y : this->y;
        T az = this->z<0 ? -this->z : this->z;
        if(ax > ay)
            if(ax > az)
                return VEC3<T,T3>(this->x, 0, 0);
            else
                return VEC3<T,T3>(0, 0, this->z);
        else
            if(ay > az)
                return VEC3<T,T3>(0, this->y, 0);
            else
                return VEC3<T,T3>(0, 0, this->z);
    }

    BOTH void dominantize()
    {
        T ax = this->x<0 ? -this->x : this->x;
        T ay = this->y<0 ? -this->y : this->y;
        T az = this->z<0 ? -this->z : this->z;
        if(ax > ay)
            if(ax > az)
                this->y = this->z = 0;
            else
                this->x = this->y = 0;
        else
            if(ay > az)
                this->x = this->z = 0;
            else
                this->x = this->y = 0;
    }

    BOTH static T clamp(T that)
    {
        return that<0.0 ? 0.0 : that>1.0 ? 1.0 : that;
    }

    BOTH void clamp()
    {
        if (this->x < 0.0) this->x = 0.0;
        else if (this->x > 1.0) this->x = 1.0;

        if (this->y < 0.0) this->y = 0.0;
        else if (this->y > 1.0) this->y = 1.0;

        if (this->z < 0.0) this->z = 0.0;
        else if (this->z > 1.0) this->z = 1.0;
    }

    BOTH VEC3<T,T3> clamped()
    {
        return VEC3<T,T3>(this->x<0.0 ? 0.0 : this->x>1.0 ? 1.0 : this->x,
                    this->y<0.0 ? 0.0 : this->y>1.0 ? 1.0 : this->y,
                    this->z<0.0 ? 0.0 : this->z>1.0 ? 1.0 : this->z);
    }

    BOTH VEC3<T,T3> extrapolate(VEC3<T,T3> d, T f)
    {
        return (d*f) + *this;
    }

    BOTH bool in_lyap_space()
    {
        return
            (this->x>=0) && (this->x<=4.0) &&
            (this->y>=0) && (this->y<=4.0) &&
            (this->z>=0) && (this->z<=4.0);
    }

    BOTH T sphere3d()
    {
        VEC3<T,T3> C = VEC3<T,T3>(2,2,2);
        VEC3<T,T3> H = *this - C;
        T m = H.mag() - 1.5;
        return m;
    }

    BOTH T lyap3d(signed int *seq, unsigned int settle, unsigned int accum)
    {
        T abc[3];
        abc[0] = this->x;
        abc[1] = this->y;
        abc[2] = this->z;

        signed int *seqp; // Position in the sequence loop
        unsigned int n; // Iteration counter
        T r; // Iteration value
        T v = 0.5;
        T l = 0;

        // Initialise for this pixel
        seqp = seq;

        // Settle by running the iteration without accumulation
        for(n = 0; n < settle; n++) {
            r = abc[*seqp++];
            if(*seqp==-1) seqp = seq;
            v = r * v * (1.0 - v);
        }

        if(this->fabs(v-0.5) >= 1e-10 ) {
            // Now calculate the value by running the iteration with accumulation
            for(n = 0; n < accum; n++) {
                r = abc[*seqp++];
                if(*seqp==-1) seqp = seq;
                v = r * v * (1.0 - v);
                r = r - 2.0 * r * v;
                l += this->logf(r<0 ? -r : r);
            }
        }

        return l;
    }
};


#endif
