#ifndef CEREBRICS_TEST_UTILS_H
#define CEREBRICS_TEST_UTILS_H

#include "cerebrics/cerebrics.h"

void assert_close(float got, float expected);
void assert_tensor_close(const CBR_Tensor *got, const float *expected, int expected_size);

#endif
