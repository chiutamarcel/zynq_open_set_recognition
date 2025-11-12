#include "gemm.hpp"
#include "common.hpp"

void gemm(hls::stream<float>& in, hls::stream<I32>& weights, hls::stream<axis32_t>& out, float in_scale) {
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    #pragma HLS INTERFACE mode=s_axilite port=in_scale bundle=control

    #pragma HLS INTERFACE axis port=in name=in
    #pragma HLS INTERFACE axis port=weights name=weights
    #pragma HLS INTERFACE axis port=out name=out

    I8 localIn[MAX_SIZE_IN];
    #pragma HLS ARRAY_PARTITION variable=localIn dim=0 complete

    I8 localWeights[MAX_SIZE_IN][MAX_SIZE_OUT];
    #pragma HLS ARRAY_PARTITION variable=localWeights dim=2 complete

    I32 localOut[MAX_SIZE_OUT] = {0};
    #pragma HLS ARRAY_PARTITION variable=localOut dim=0 complete

    LOAD_INPUT:
        for (int i = 0; i < MAX_SIZE_IN; i++) {
            float tmp = in.read();
            localIn[i] = quantize(tmp, in_scale);
        }


    LOAD_WEIGHTS:
        for (int i = 0; i < MAX_SIZE_IN; i++) {
            for (int j = 0; j < MAX_SIZE_OUT; j++) {
                localWeights[i][j] = (I8) weights.read();
            }
        }


    systolic1:
    for (int i = 0; i < MAX_SIZE_IN; i++) {

        systolic2:
        for (int k = 0; k < MAX_SIZE_OUT; k++) {
            #pragma HLS UNROLL

            I32 last = (i == 0) ? (I32)0 : localOut[k];
            I8 a_val = localIn[i];
            I8 b_val = localWeights[i][k];
            I32 result = last + (I32)a_val * (I32)b_val;

            localOut[k] = result;
        }
    }

    for (int i = 0; i < MAX_SIZE_OUT; i++) {
        axis32_t tmp_out;

        tmp_out.data = localOut[i];
        tmp_out.keep = -1;
        tmp_out.strb = 0;
        tmp_out.last = (i == MAX_SIZE_OUT - 1);

        out.write(tmp_out);
    }

}
