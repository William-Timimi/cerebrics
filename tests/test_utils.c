#include "test_utils.h"
#include <assert.h>
#include <math.h>

void assert_close(float got, float expected) {
    assert(fabsf(got - expected) < 1e-5F);
}

void assert_tensor_close(const CBR_Tensor *got, const float *expected, int expected_size) {
    assert(got != NULL);
    assert(expected != NULL);
    assert(got->size == expected_size);

    for (int i = 0; i < expected_size; i++) {
        assert_close(got->data[i], expected[i]);
    }
}
