#include "test_utils.h"
#include <assert.h>
#include <math.h>

void assert_close(float got, float expected) {
    assert(fabsf(got - expected) < 1e-5F);
}
