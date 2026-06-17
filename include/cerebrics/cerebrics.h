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

CBR_Tensor *cbr_tensor_create(int ndim, const int *shape, int requires_grad);
CBR_Tensor *cbr_tensor_from_data(const float *data, int ndim, const int *shape, int requires_grad);
void cbr_tensor_free(CBR_Tensor *t);

CBR_Status cbr_tensor_fill(CBR_Tensor *t, float value);
CBR_Status cbr_tensor_get(const CBR_Tensor *t, const int *indices, float *out_value);
CBR_Status cbr_tensor_set(CBR_Tensor *t, const int *indices, float value);

int cbr_num_el(CBR_Tensor *t);
float *cbr_data(CBR_Tensor *t);
float *cbr_grad(CBR_Tensor *t);
int cbr_shape(CBR_Tensor *t, int axis);

#ifdef __cplusplus
}
#endif
#endif
