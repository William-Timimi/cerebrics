#include "cerebrics/cerebrics.h"
#include "test_utils.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    int shape[1] = {3};
    float av[3] = {1.0f, 2.0f, 3.0f};
    float bv[3] = {10.0f, 20.0f, 30.0f};

    CBR_Tensor *a = cbr_tensor_from_data(av, 1, shape, 1);
    CBR_Tensor *b = cbr_tensor_from_data(bv, 1, shape, 1);

    assert(a != NULL);
    assert(b != NULL);

    CBR_Tensor *sum = cbr_add(a, b);
    float expected_sum[3] = {11.0f, 22.0f, 33.0f};
    assert_tensor_close(sum, expected_sum, 3);

    CBR_Tensor *diff = cbr_sub(b, a);
    float expected_diff[3] = {9.0f, 18.0f, 27.0f};
    assert_tensor_close(diff, expected_diff, 3);

    CBR_Tensor *prod = cbr_mul(a, b);
    float expected_prod[3] = {10.0f, 40.0f, 90.0f};
    assert_tensor_close(prod, expected_prod, 3);

    CBR_Tensor *scaled = cbr_scalmul(2.0f, a);
    float expected_scaled[3] = {2.0f, 4.0f, 6.0f};
    assert_tensor_close(scaled, expected_scaled, 3);

    CBR_Tensor *scalar = cbr_sum(prod);
    float expected_scalar[1] = {140.0f};
    assert_tensor_close(scalar, expected_scalar, 1);

    cbr_tensor_free(scalar);
    cbr_tensor_free(scaled);
    cbr_tensor_free(prod);
    cbr_tensor_free(diff);
    cbr_tensor_free(sum);
    cbr_tensor_free(b);
    cbr_tensor_free(a);

    puts("test_ops passed");
    return 0;
}
