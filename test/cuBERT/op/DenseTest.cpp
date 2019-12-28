#include "../common_test.h"
#include "cuBERT/op/Dense.h"
using namespace cuBERT;

TEST_F(CommonTest, dense) {
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};

    Dense<float> dense(handle, 2, 3, kernel, bias, 16);

    float input[6] = {1, 2, 3, 4, 5, 6};
    float *input_gpu = (float*) cuBERT::malloc(6 * sizeof(float));
    cuBERT::memcpy(input_gpu, input, 6 * sizeof(float), 1);

    float *output_gpu = (float*) cuBERT::malloc(9 * sizeof(float));

    dense.compute(3, input_gpu, output_gpu);

    float output[9];
    cuBERT::memcpy(output, output_gpu, sizeof(float) * 9, 2);

    cuBERT::free(output_gpu);
    cuBERT::free(input_gpu);

    EXPECT_FLOAT_EQ(output[0], 9 - 3);
    EXPECT_FLOAT_EQ(output[1], 12 - 2);
    EXPECT_FLOAT_EQ(output[2], 15 - 1);
    EXPECT_FLOAT_EQ(output[3], 19 - 3);
    EXPECT_FLOAT_EQ(output[4], 26 - 2);
    EXPECT_FLOAT_EQ(output[5], 33 - 1);
    EXPECT_FLOAT_EQ(output[6], 29 - 3);
    EXPECT_FLOAT_EQ(output[7], 40 - 2);
    EXPECT_FLOAT_EQ(output[8], 51 - 1);
}

TEST_F(CommonTest, dense_qkv) {
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};

    DenseQKV<float> dense(handle, 2, 3,
                          kernel, kernel, kernel,
                          bias, bias, bias,
                          16);
    
    float input[6] = {1, 2, 3, 4, 5, 6};
    float *input_gpu = (float*) cuBERT::malloc(6 * sizeof(float));
    cuBERT::memcpy(input_gpu, input, 6 * sizeof(float), 1);

    float *output_gpu = (float*) cuBERT::malloc(9 * sizeof(float) * 3);

    dense.compute(3, input_gpu, output_gpu, 9);

    float output[27];
    cuBERT::memcpy(output, output_gpu, sizeof(float) * 9 * 3, 2);

    cuBERT::free(output_gpu);
    cuBERT::free(input_gpu);

    EXPECT_FLOAT_EQ(output[0], 9 - 3);
    EXPECT_FLOAT_EQ(output[1], 12 - 2);
    EXPECT_FLOAT_EQ(output[2], 15 - 1);
    EXPECT_FLOAT_EQ(output[3], 19 - 3);
    EXPECT_FLOAT_EQ(output[4], 26 - 2);
    EXPECT_FLOAT_EQ(output[5], 33 - 1);
    EXPECT_FLOAT_EQ(output[6], 29 - 3);
    EXPECT_FLOAT_EQ(output[7], 40 - 2);
    EXPECT_FLOAT_EQ(output[8], 51 - 1);

    EXPECT_FLOAT_EQ(output[9], 9 - 3);
    EXPECT_FLOAT_EQ(output[10], 12 - 2);
    EXPECT_FLOAT_EQ(output[11], 15 - 1);
    EXPECT_FLOAT_EQ(output[12], 19 - 3);
    EXPECT_FLOAT_EQ(output[13], 26 - 2);
    EXPECT_FLOAT_EQ(output[14], 33 - 1);
    EXPECT_FLOAT_EQ(output[15], 29 - 3);
    EXPECT_FLOAT_EQ(output[16], 40 - 2);
    EXPECT_FLOAT_EQ(output[17], 51 - 1);

    EXPECT_FLOAT_EQ(output[18], 9 - 3);
    EXPECT_FLOAT_EQ(output[19], 12 - 2);
    EXPECT_FLOAT_EQ(output[20], 15 - 1);
    EXPECT_FLOAT_EQ(output[21], 19 - 3);
    EXPECT_FLOAT_EQ(output[22], 26 - 2);
    EXPECT_FLOAT_EQ(output[23], 33 - 1);
    EXPECT_FLOAT_EQ(output[24], 29 - 3);
    EXPECT_FLOAT_EQ(output[25], 40 - 2);
    EXPECT_FLOAT_EQ(output[26], 51 - 1);
}
