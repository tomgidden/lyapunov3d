#include <cmath>

#ifndef __LYAP_COLOR_HPP__
#define __LYAP_COLOR_HPP__

static const double COLOR_EPSILON = 1e-6;

#define BOTH __device__ __host__

template<class T,class T4> class COLOR : public T4 {

public:

    BOTH COLOR<T,T4>(T _r, T _g, T _b, T _a)
    {
        this->x = _r;
        this->y = _g;
        this->z = _b;
        this->w = _a;
    }

    BOTH COLOR<T,T4>()
    {
        this->x = this->y = this->z = this->w = 0;
    }

    BOTH COLOR<T,T4> &operator+=(const COLOR<T,T4> &that)
    {
        this->x += that.x;
        this->y += that.y;
        this->z += that.z;
        this->w += that.w;
        return *this;
    }

    BOTH COLOR<T,T4> &operator-=(const COLOR<T,T4> &that)
    {
        this->x -= that.x;
        this->y -= that.y;
        this->z -= that.z;
        this->w -= that.w;
        return *this;
    }

    BOTH COLOR<T,T4> &operator*=(const T scale)
    {
        this->x *= scale;
        this->y *= scale;
        this->z *= scale;
        this->w *= scale;
        return *this;
    }

    BOTH COLOR<T,T4> &operator/=(const T divisor)
    {
        this->x /= divisor;
        this->y /= divisor;
        this->z /= divisor;
        this->w /= divisor;
        return *this;
    }

    BOTH inline void set(T _r, T _g, T _b, T _a)
    {
        this->x = _r;
        this->y = _g;
        this->z = _b;
        this->w = _a;
    }

    BOTH COLOR<T,T4> operator+(const COLOR<T,T4> &that)
    {
        return COLOR<T,T4>(this->x + that.x, this->y + that.y, this->z + that.z, this->w + that.w);
    }

    BOTH COLOR<T,T4> operator-(const COLOR<T,T4> &that)
    {
        return COLOR<T,T4>(this->x - that.x, this->y - that.y, this->z - that.z, this->w - that.w);
    }

    BOTH COLOR<T,T4> operator*(const T scale)
    {
        return COLOR<T,T4>(this->x * scale, this->y * scale, this->z * scale, this->w * scale);
    }

    BOTH COLOR<T,T4> operator/(const T divisor)
    {
        return COLOR<T,T4>(this->x / divisor, this->y / divisor, this->z / divisor, this->w / divisor);
    }

    BOTH T mag2()
    {
        return (this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
    }

    BOTH T mag()
    {
        return this->sqrt(this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
    }

    BOTH COLOR<T,T4> normalized()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
        if ( mag2 < COLOR_EPSILON ) {
            return COLOR<T,T4>();
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - COLOR_EPSILON) && mag2 < (1.0f + COLOR_EPSILON))) {
            return *this;
        }
        else {
            return *this / this->sqrt(mag2);
        }
    }

    BOTH void normalize()
    {
        T mag2 = (this->x*this->x + this->y*this->y + this->z*this->z + this->w*this->w);
        if ( mag2 < COLOR_EPSILON ) {
            this->x = this->y = this->z = this->w = 0;
        }
        else if (mag2 == 1.0f || (mag2 > (1.0f - COLOR_EPSILON) && mag2 < (1.0f + COLOR_EPSILON))) {
            // nop
        }
        else {
            mag2 = this->sqrt(mag2);
            this->x /= mag2;
            this->y /= mag2;
            this->z /= mag2;
            this->w /= mag2;
        }
    }

    BOTH static void clamp(T &that)
    {
        if (that < 0.0) that = 0.0;
        else if (that > 1.0) that = 1.0;
    }

    BOTH static T clamped(T that)
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

        if (this->w < 0.0) this->w = 0.0;
        else if (this->w > 1.0) this->w = 1.0;
    }

    BOTH COLOR<T,T4> clamped()
    {
        return COLOR<T,T4>(this->x<0.0 ? 0.0 : this->x>1.0 ? 1.0 : this->x,
                           this->y<0.0 ? 0.0 : this->y>1.0 ? 1.0 : this->y,
                           this->z<0.0 ? 0.0 : this->z>1.0 ? 1.0 : this->z,
                           this->w<0.0 ? 0.0 : this->w>1.0 ? 1.0 : this->w);
    }

    BOTH void to_rgba(unsigned char *rgba)
    {
        *rgba++ = (unsigned char)(255.0 * this->x);
        *rgba++ = (unsigned char)(255.0 * this->y);
        *rgba++ = (unsigned char)(255.0 * this->z);
        *rgba++ = (unsigned char)(255.0 * this->w);
    }
};


#endif
