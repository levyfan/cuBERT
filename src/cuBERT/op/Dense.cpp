#include "cuBERT/common.h"
#include "Dense.h"

namespace cuBERT {

    template<typename T>
    Dense<T>::Dense(void* handle,
                 size_t inputs,
                 size_t units,
                 T *kernel,
                 T *bias,
                 size_t max_batch_size,
                 int algo) {
        this->handle = handle;
        this->inputs = inputs;
        this->units = units;
        this->algo = algo;

        this->kernel = static_cast<T *>(cuBERT::malloc(sizeof(T) * inputs * units));
        cuBERT::memcpy(this->kernel, kernel, inputs * units * sizeof(T), 1);

        this->bias = static_cast<T *>(cuBERT::malloc(sizeof(T) * units * max_batch_size));
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias + units * i, bias, units * sizeof(T), 1);
        }
    }

    template<typename T>
    Dense<T>::~Dense() {
        cuBERT::free(bias);
        cuBERT::free(kernel);
    }

    template<typename T>
    void Dense<T>::compute(size_t batch_size, T *input, T *output) {
        _pre_compute(batch_size, output);
        _in_compute(batch_size, input, output);
    }

    template<typename T>
    void Dense<T>::_pre_compute(size_t batch_size, T *output) {
        void* streamId = blas_get_stream(handle);
        cuBERT::memcpyAsync(output, bias, units * batch_size * sizeof(T), 3, streamId);
    }

    template<typename T>
    void Dense<T>::_in_compute(size_t batch_size, T *input, T *output) {
        cuBERT::blas_gemm(handle,
                           false, false,
                           units, batch_size, inputs,
                           1.f,
                           kernel, units,
                           input, inputs,
                           1.f,
                           output, units,
                           algo);
    }

    template class Dense<float>;
#ifdef HAVE_CUDA
    template class Dense<half>;
#endif

    template<typename T>
    DenseQKV<T>::DenseQKV(void* handle,
                          size_t inputs, size_t units,
                          T *kernel_q, T *kernel_k, T *kernel_v,
                          T *bias_q, T *bias_k, T *bias_v,
                          size_t max_batch_size,
                          int algo) {
        this->handle = handle;
        this->inputs = inputs;
        this->units = units;
        this->algo = algo;

        this->kernel = static_cast<T *>(cuBERT::malloc(sizeof(T) * inputs * units * 3));
        this->kernel_q = this->kernel;
        this->kernel_k = this->kernel + inputs * units;
        this->kernel_v = this->kernel + inputs * units * 2;
        cuBERT::memcpy(this->kernel_q, kernel_q, inputs * units * sizeof(T), 1);
        cuBERT::memcpy(this->kernel_k, kernel_k, inputs * units * sizeof(T), 1);
        cuBERT::memcpy(this->kernel_v, kernel_v, inputs * units * sizeof(T), 1);

        this->bias = static_cast<T *>(cuBERT::malloc(sizeof(T) * units * max_batch_size * 3));
        this->bias_q = this->bias;
        this->bias_k = this->bias + units * max_batch_size;
        this->bias_v = this->bias + units * max_batch_size * 2;
        for (int i = 0; i < max_batch_size; ++i) {
            cuBERT::memcpy(this->bias_q + units * i, bias_q, units * sizeof(T), 1);
            cuBERT::memcpy(this->bias_k + units * i, bias_k, units * sizeof(T), 1);
            cuBERT::memcpy(this->bias_v + units * i, bias_v, units * sizeof(T), 1);
        }
    }

    template<typename T>
    DenseQKV<T>::~DenseQKV() {
        cuBERT::free(this->bias);
        cuBERT::free(this->kernel);
    }

    template<typename T>
    void DenseQKV<T>::_pre_compute(size_t batch_size, T *output, size_t output_stride) {
        void* streamId = blas_get_stream(handle);
        cuBERT::memcpyAsync(output, bias_q, units * batch_size * sizeof(T), 3, streamId);
        cuBERT::memcpyAsync(output + output_stride, bias_k, units * batch_size * sizeof(T), 3, streamId);
        cuBERT::memcpyAsync(output + output_stride * 2, bias_v, units * batch_size * sizeof(T), 3, streamId);
    }

    template<typename T>
    void DenseQKV<T>::_in_compute(size_t batch_size, T *input, T *output, size_t output_stride) {
        cuBERT::blas_gemm_strided_batch(handle,
                                        false, false,
                                        units, batch_size, inputs,
                                        1.f,
                                        kernel, units, inputs * units,
                                        input, inputs, 0,
                                        1.f,
                                        output, units, output_stride,
                                        3);
    }

    template<typename T>
    void DenseQKV<T>::compute(size_t batch_size, T *input, T *output, size_t output_stride) {
        _pre_compute(batch_size, output, output_stride);
        _in_compute(batch_size, input, output, output_stride);
    }

    template class DenseQKV<float>;
#ifdef HAVE_CUDA
    template class DenseQKV<half>;
#endif
}
