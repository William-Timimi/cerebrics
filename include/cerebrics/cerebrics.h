#ifndef CEREBRICS_H
#define CEREBRICS_H

#include "cerebrics/errors.h"
#include <stddef.h>

#ifdef __cplusplus
#extern "C" {
#endif

#define CBR_MAX_DIMS 4

typedef struct CBR_Tensor {
    float *data;
    float *grad;
    int ndim;
    int shape[CBR_MAX_DIMS];
    int strides[CBR_MAX_DIMS];
    int size;
    int requires_grad;
} CBR_Tensor;

// ways to create and destroy a tensor
CBR_Tensor *cbr_tensor_create(int ndim, const int *shape, int requires_grad);
CBR_Tensor *cbr_tensor_from_data(const float *data, int ndim, const int *shape, int requires_grad);
void cbr_tensor_free(CBR_Tensor *t);

// Get the different properties of a tensor
int cbr_num_el(CBR_Tensor *t);
float *cbr_data(CBR_Tensor *t);
float *cbr_grad(CBR_Tensor *t);
int cbr_shape(CBR_Tensor *t, int axis);

// functions to get and set data in a tensor
CBR_Status cbr_tensor_fill(CBR_Tensor *t, float value);
CBR_Status cbr_tensor_get(const CBR_Tensor *t, const int *indices, float *out_value);
CBR_Status cbr_tensor_set(CBR_Tensor *t, const int *indices, float value);
void cbr_tensor_zero_grad(CBR_Tensor *t);

// Mathmatical operations for tensors
CBR_Tensor *cbr_add(CBR_Tensor *a, CBR_Tensor *b);
CBR_Tensor *cbr_sub(CBR_Tensor *a, CBR_Tensor *b);
CBR_Tensor *cbr_mul(CBR_Tensor *a, CBR_Tensor *b);
CBR_Tensor *cbr_scalmul(float f, CBR_Tensor *t);
CBR_Tensor *cbr_matmul(CBR_Tensor *a, CBR_Tensor *b);
CBR_Tensor *cbr_relu(CBR_Tensor *x);
CBR_Tensor *cbr_sum(CBR_Tensor *x);

#ifdef __cplusplus
}
#endif
#endif
