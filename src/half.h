// filepath: /home/user/ixy/src/half.h
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 主机侧 half：存 IEEE754 binary16 的 bit pattern（16-bit）
typedef uint16_t half;

// float -> half bits (round to nearest even)
static inline half float_to_half(float f);

// half bits -> float
static inline float half_to_float(half h);

#ifdef __cplusplus
}
#endif

// ---- Implementation ----
// 基于常见的 IEEE754 binary32 <-> binary16 位级转换（不依赖 _Float16）
static inline half float_to_half(float f) {
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t x = v.u;

    uint32_t sign = (x >> 16) & 0x8000u;          // sign in bit 15
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu); // float exponent
    uint32_t mant = x & 0x7FFFFFu;                // float mantissa

    // NaN/Inf
    if (exp == 255) {
        if (mant != 0) {
            // NaN -> qNaN
            return (half)(sign | 0x7E00u);
        }
        return (half)(sign | 0x7C00u); // Inf
    }

    // Convert exponent from bias 127 to bias 15
    int32_t half_exp = exp - 127 + 15;

    if (half_exp >= 31) {
        // overflow -> Inf
        return (half)(sign | 0x7C00u);
    }

    if (half_exp <= 0) {
        // subnormal or underflow to zero
        if (half_exp < -10) {
            // too small -> signed zero
            return (half)sign;
        }
        // subnormal half: mantissa with implicit leading 1
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(1 - half_exp); // 1..10
        // we need to shift mant from 23 bits to 10 bits, plus subnormal shift
        // total shift = 13 + shift
        uint32_t rshift = 13u + shift;

        // round to nearest even
        uint32_t remainder = mant & ((1u << rshift) - 1u);
        uint32_t halfway   = 1u << (rshift - 1u);
        uint32_t half_mant = mant >> rshift;

        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
            half_mant++;
        }

        return (half)(sign | (half_mant & 0x03FFu));
    }

    // normal half
    // round mantissa from 23 bits to 10 bits
    uint32_t half_mant = mant >> 13;
    uint32_t remainder = mant & 0x1FFFu; // low 13 bits
    // round to nearest even
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x0400u) { // mantissa overflow
            half_mant = 0;
            half_exp++;
            if (half_exp >= 31) {
                return (half)(sign | 0x7C00u);
            }
        }
    }

    return (half)(sign | ((uint32_t)half_exp << 10) | (half_mant & 0x03FFu));
}

static inline float half_to_float(half h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x03FFu;

    union { uint32_t u; float f; } v;

    if (exp == 0) {
        if (mant == 0) {
            // zero
            v.u = sign;
            return v.f;
        }
        // subnormal -> normalize
        // value = mant * 2^(−24) (for half), convert to float
        exp = 1;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x03FFu;
        uint32_t float_exp = (uint32_t)(exp - 1 + 127 - 15);
        uint32_t float_mant = mant << 13;
        v.u = sign | (float_exp << 23) | float_mant;
        return v.f;
    }

    if (exp == 31) {
        // Inf/NaN
        v.u = sign | 0x7F800000u | (mant << 13);
        return v.f;
    }

    // normal
    uint32_t float_exp  = (exp - 15 + 127);
    uint32_t float_mant = mant << 13;
    v.u = sign | (float_exp << 23) | float_mant;
    return v.f;
}