#pragma once

//#include <ap_float.h>
#include <hls_stream.h>
#include "common.hpp"

#define MAX_SIZE_IN 300
#define MAX_SIZE_OUT 10

void gemm(hls::stream<float>& in, hls::stream<I32>& weights, hls::stream<axis32_t>& out, float in_scale);
