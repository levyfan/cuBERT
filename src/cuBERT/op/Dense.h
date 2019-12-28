#ifndef CUBERT_DENSE_H
#define CUBERT_DENSE_H


#include <cstddef>

namespace cuBERT {
/**
 * Input: batch_size * inputs
 * Kernel: inputs * units
 * Bias: units
 * Output: batch_size * units
 *
 * Output = Input @ Kernel + Bias
 */
    template<typename T>
    class Dense {
    public:
        explicit Dense(void* handle,
                       size_t inputs,
                       size_t units,
                       T *kernel,
                       T *bias,
                       size_t max_batch_size,
                       int algo = -1);

        virtual ~Dense();

        void _pre_compute(size_t batch_size, T *output);

        void _in_compute(size_t batch_size, T *input, T *output);

        void compute(size_t batch_size, T *input, T *output);

    private:
        void* handle;

        size_t inputs;
        size_t units;
        int algo;

        // gpu/cpu buffer
        T *kernel;
        T *bias;
    };

    template<typename T>
    class DenseQKV {
    public:
        explicit DenseQKV(void* handle,
                          size_t inputs, size_t units,
                          T *kernel_q, T *kernel_k, T *kernel_v,
                          T *bias_q, T *bias_k, T *bias_v,
                          size_t max_batch_size,
                          int algo = -1);
        
        virtual ~DenseQKV();

        void _pre_compute(size_t batch_size, T *output, size_t output_stride);

        void _in_compute(size_t batch_size, T *input, T *output, size_t output_stride);
        
        void compute(size_t batch_size, T *input, T *output, size_t output_stride);

    private:
        void* handle;

        size_t inputs;
        size_t units;
        int algo;

        T *kernel; T *kernel_q; T *kernel_k; T *kernel_v;
        T *bias; T *bias_q; T *bias_k; T *bias_v;
    };
}

#endif //CUBERT_DENSE_H
