#include "cerebrics/cerebrics.h"
#include "test_utils.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    int shape[2] = {2, 3};
    float values[6] = {1, 2, 3, 4, 5, 6};
    CBR_Tensor *x = cbr_tensor_from_data(values, 2, shape, 1);

    assert(x != NULL);
    assert(cbr_num_el(x) == 6);
    assert(cbr_shape(x, 0) == 2);
    assert(cbr_shape(x, 1) == 3);
    assert(x->strides[0] == 3);
    assert(x->strides[1] == 1);
    assert(x->requires_grad == 1);
    assert(cbr_grad(x) == NULL);
    assert_close(cbr_data(x)[4], 5.0F);

    float got = 0.0F;
    assert(cbr_tensor_get(x, (int[]){1, 2}, &got) == CBR_OK);
    assert_close(got, 6.0F);

    assert(cbr_tensor_set(x, (int[]){0, 1}, 42.0F) == CBR_OK);
    assert(cbr_tensor_get(x, (int[]){0, 1}, &got) == CBR_OK);
    assert_close(got, 42.0F);

    assert(cbr_tensor_get(x, (int[]){2, 0}, &got) == CBR_ERR_OUT_OF_BOUNDS);
    assert(cbr_tensor_set(x, (int[]){0, 3}, 9.0F) == CBR_ERR_OUT_OF_BOUNDS);
    assert(cbr_tensor_get(NULL, (int[]){0, 0}, &got) == CBR_ERR_INVALID_ARGUMENT);
    assert(cbr_tensor_get(x, (int[]){0, 0}, NULL) == CBR_ERR_INVALID_ARGUMENT);

    assert(cbr_tensor_fill(x, 7.0F) == CBR_OK);
    for (int i = 0; i < cbr_num_el(x); i++) {
        assert_close(cbr_data(x)[i], 7.0F);
    }
    assert(cbr_tensor_fill(NULL, 1.0F) == CBR_ERR_INVALID_ARGUMENT);

    cbr_tensor_free(x);
    puts("test_tensor passed");
    return 0;
}
