#include "cerebrics/cerebrics.h"
#include <stdlib.h>
#include <string.h>

static CBR_Status cbr_tensor_init_shape(CBR_Tensor *t, int ndim, const int *shape) {
    if (!t || !shape) {
        return CBR_ERR_INVALID_ARGUMENT;
    }
    if (ndim < 1 || ndim > CBR_MAX_DIMS) {
        return CBR_ERR_SHAPE;
    }

    t->ndim = ndim;
    t->size = 1;

    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return CBR_ERR_SHAPE;
        }
        t->shape[i] = shape[i];
    }

    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = t->size;
        t->size *= t->shape[i];
    }

    return CBR_OK;
}

CBR_Tensor *cbr_tensor_create(int ndim, const int *shape, int requires_grad) {
    CBR_Tensor *t = (CBR_Tensor *)calloc(1, sizeof(CBR_Tensor));
    if (!t) {
        return NULL;
    }

    if (cbr_tensor_init_shape(t, ndim, shape) != CBR_OK) {
        free(t);
        return NULL;
    }

    t->data = (float *)calloc((size_t)t->size, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }

    t->grad = NULL;
    t->requires_grad = requires_grad;
    return t;
}

CBR_Tensor *cbr_tensor_from_data(const float *data, int ndim, const int *shape, int requires_grad) {
    if (!data) {
        return NULL;
    }
    CBR_Tensor *t = cbr_tensor_create(ndim, shape, requires_grad);
    if (!t) {
        return NULL;
    }
    memcpy(t->data, data, sizeof(float) * (size_t)t->size);
    return t;
}

void cbr_tensor_free(CBR_Tensor *t) {
    if (!t) {
        return;
    }
    free(t->data);
    free(t->grad);
    free(t);
}

void cbr_tensor_zero_grad(CBR_Tensor *t) {
    if (!t) {
        return;
    }
    if (t->grad) {
        memset(t->grad, 0, sizeof(float) * (size_t)t->size);
    }
}

CBR_Status cbr_tensor_fill(CBR_Tensor *t, float value) {
    if (!t) {
        return CBR_ERR_INVALID_ARGUMENT;
    }
    for (int i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
    return CBR_OK;
}

CBR_Status cbr_get_flat_index(const CBR_Tensor *t, const int *indices, int *flat_index_out) {
    if (!t || !indices || !flat_index_out) {
        return CBR_ERR_INVALID_ARGUMENT;
    }

    int flat_index = 0;
    for (int i = 0; i < t->ndim; i++) {
        int index = indices[i];
        if (index < 0 || index >= t->shape[i]) {
            return CBR_ERR_OUT_OF_BOUNDS;
        }

        flat_index += index * t->strides[i];
    }

    *flat_index_out = flat_index;
    return CBR_OK;
}

// Given a serries of indices, gets the value at that position in the
// tensor and sets out_value to it.
// returns 0 for a success and 1 for an out of bounds error.
CBR_Status cbr_tensor_get(const CBR_Tensor *t, const int *indices, float *out_value) {
    if (!out_value) {
        return CBR_ERR_INVALID_ARGUMENT;
    }

    int flat_index = 0;
    CBR_Status status = cbr_get_flat_index(t, indices, &flat_index);
    if (status != CBR_OK) {
        return status;
    }

    *out_value = t->data[flat_index];
    return CBR_OK;
}

CBR_Status cbr_tensor_set(CBR_Tensor *t, const int *indices, float value) {

    int flat_index = 0;
    CBR_Status status = cbr_get_flat_index(t, indices, &flat_index);
    if (status != CBR_OK) {
        return status;
    }

    t->data[flat_index] = value;
    return CBR_OK;
}

int cbr_num_el(CBR_Tensor *t) {
    if (!t) {
        return 0;
    }
    return t->size;
}

float *cbr_data(CBR_Tensor *t) {
    if (!t) {
        return NULL;
    }
    return t->data;
}

float *cbr_grad(CBR_Tensor *t) {
    if (!t) {
        return NULL;
    }
    return t->grad;
}

int cbr_shape(CBR_Tensor *t, int axis) {
    if (!t || axis < 0 || axis >= t->ndim) {
        return -1;
    }
    return t->shape[axis];
}
