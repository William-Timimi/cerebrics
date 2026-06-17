#include "cerebrics/cerebrics.h"
#include <stdlib.h>
#include <string.h>

/*------------------------------------------------------------
-   Functions for creating and freeing tensors               -
------------------------------------------------------------*/

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

/*------------------------------------------------------------
-   Get the different properties of a tensor                 -
------------------------------------------------------------*/

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

/*------------------------------------------------------------
-   functions to get and set data in a tensor                -
------------------------------------------------------------*/

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

/*------------------------------------------------------------
-   Mathmatical operations for tensors                       -
------------------------------------------------------------*/

static int are_ts_same_shape(CBR_Tensor *a, CBR_Tensor *b) {
    if (!a || !b) {
        return 0;
    }
    // checks both tensors have same number of dimensions
    // then checks each dimension is of same size
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

CBR_Tensor *cbr_add(CBR_Tensor *a, CBR_Tensor *b) {
    // can only add tensors of same shape
    if (!are_ts_same_shape(a, b)) {
        return NULL;
    }

    // Creates output tensor, returns null if creation unsucsessful
    CBR_Tensor *out = cbr_tensor_create(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    if (!out) {
        return NULL;
    }

    // loops through every element and sets it as elwise addition
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

CBR_Tensor *cbr_sub(CBR_Tensor *a, CBR_Tensor *b) {
    // can only subtract tensors of same shape
    if (!are_ts_same_shape(a, b)) {
        return NULL;
    }

    // Creates output tensor, returns null if creation unsucsessful
    CBR_Tensor *out = cbr_tensor_create(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    if (!out) {
        return NULL;
    }

    // loops through every element and sets it as elwise subtraction
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return out;
}

CBR_Tensor *cbr_mul(CBR_Tensor *a, CBR_Tensor *b) {
    // can only multiply tensors of same shape
    if (!are_ts_same_shape(a, b)) {
        return NULL;
    }

    // Creates output tensor, returns null if creation unsucsessful
    CBR_Tensor *out = cbr_tensor_create(a->ndim, a->shape, a->requires_grad || b->requires_grad);
    if (!out) {
        return NULL;
    }

    // loops through every element and sets it as elwise multiplication
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    return out;
}

CBR_Tensor *cbr_scalmul(float f, CBR_Tensor *t) {

    // Creates output tensor, returns null if creation unsucsessful
    CBR_Tensor *out = cbr_tensor_create(t->ndim, t->shape, t->requires_grad);
    if (!out) {
        return NULL;
    }

    // loops through every element and multiplies it by the scaler
    for (int i = 0; i < out->size; i++) {
        out->data[i] = t->data[i] * f;
    }
    return out;
}

CBR_Tensor *cbr_matmul(CBR_Tensor *a, CBR_Tensor *b) {
    // can only do matrix multiplaction with matrices
    if (!a || !b || a->ndim != 2 || b->ndim != 2) {
        return NULL;
    }

    // checks matrices have dimensions that can be multiplied
    if (a->shape[1] != b->shape[0]) {
        return NULL;
    }

    // gets new shape and creates new tensor
    int m = a->shape[0];
    int n = b->shape[1];
    int k = b->shape[1];
    int shape[2] = {m, n};
    CBR_Tensor *out = cbr_tensor_create(2, shape, a->requires_grad || b->requires_grad);
    if (!out) return NULL;

    // loops through and works out the value of each element
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) {
                acc += a->data[i * k + p] * b->data[p * n + j];
            }
            out->data[i * n + j] = acc;
        }
    }

    return out;
}

CBR_Tensor *cbr_relu(CBR_Tensor *x) {
    if (!x) return NULL;
    CBR_Tensor *out = cbr_tensor_create(x->ndim, x->shape, x->requires_grad);
    if (!out) return NULL;

    for (int i = 0; i < x->size; i++) {
        out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    }

    return out;
}

CBR_Tensor *cbr_sum(CBR_Tensor *x) {
    if (!x) return NULL;
    int shape[1] = {1};
    CBR_Tensor *out = cbr_tensor_create(1, shape, x->requires_grad);
    if (!out) return NULL;
    float acc = 0.0f;
    for (int i = 0; i < x->size; i++) {
        acc += x->data[i];
    }
    out->data[0] = acc;
    return out;
}
