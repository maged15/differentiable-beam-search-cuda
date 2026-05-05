// SPDX-License-Identifier: MIT
#include "dbs_cuda.h"

int dbs_cuda_available(void) { return 0; }

int dbs_cuda_set_synchronization(int synchronize) {
    return (synchronize == 0 || synchronize == 1) ? DBS_CUDA_STATUS_OK : DBS_CUDA_STATUS_INVALID_ARGUMENT;
}

int dbs_cuda_get_synchronization(void) { return 1; }

const char* dbs_cuda_status_string(int status) {
    switch (status) {
        case DBS_CUDA_STATUS_OK: return "ok";
        case DBS_CUDA_STATUS_UNAVAILABLE: return "cuda backend unavailable in this build";
        case DBS_CUDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case DBS_CUDA_STATUS_LAUNCH_FAILED: return "cuda launch failed";
        default: return "unknown cuda status";
    }
}

int dbs_cuda_decode_forward(const float*, int, int, int, int, int, int32_t*, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

int dbs_cuda_decode_forward_variable(const float*, int, int, int, int, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int32_t*, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

int dbs_cuda_sparse_backward_scatter(const int64_t*, const float*, int64_t, float*, int64_t, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

int dbs_cuda_decode_forward_fast(const float*, int, int, int, int, int, int32_t*, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}

int dbs_cuda_decode_forward_fast_ex(const float*, int, int, int, int, int, int, int32_t*, float*, void*) {
    return DBS_CUDA_STATUS_UNAVAILABLE;
}
