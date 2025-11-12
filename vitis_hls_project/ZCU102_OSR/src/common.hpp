#pragma once

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// 3x3 conv kernel
#define FILTER_V_SIZE		3
#define FILTER_H_SIZE		3

#ifndef MIN
#define MIN(a,b) ((a<b)?a:b)
#endif
#ifndef MAX
#define MAX(a,b) ((a<b)?b:a)
#endif


typedef ap_int<8>       		I8;
typedef ap_int<32>         		I32;

typedef ap_uint<8>              U8;

typedef ap_fixed<9, 1>          FP_T;

typedef hls::axis<float, 0, 0, 0> axis_float;

typedef ap_axiu<32, 0, 0, 0> axis32_t;

int clamp(int x, int min, int max);

I8 quantize(float x, float scale, int bitwidth=8, bool isSigned=true);
//float dequantize(I32 qx, float scale, bool isSigned=true);
axis_float make_axis_float(float value, bool last = false);
