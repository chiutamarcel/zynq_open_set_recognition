#include "common.hpp"
#include "hls_math.h"

int clamp(int x, int min, int max) {
    if (x < min) x = min;
    if (x > max) x = max;

    return x;
}

I8 quantize(float x, float scale, int bitwidth, bool isSigned) {
    int32_t rez_32 = round(x / scale);
    I8 rez = (I8)0;
    int32_t rez_min = 0;
    int32_t rez_max = 0;

    if (isSigned) {
        rez_max = (1 << (bitwidth-1)) - 1;
        rez_min = -(rez_max + 1);
    } else {
        rez_max = (1 << (bitwidth)) - 1;
        rez_min = 0;
    }

    rez = (I8)clamp(rez_32, rez_min, rez_max);

    return rez;
}

//float dequantize(I32 qx, float scale, bool isSigned) {
//    return qx * scale;
//}

axis_float make_axis_float(float value, bool last) {
    axis_float temp;
    temp.data = value;
    temp.keep = -1;     // All bytes valid
    temp.strb = -1;     // All bytes valid
    temp.last = last ? 1 : 0;
    return temp;
}
