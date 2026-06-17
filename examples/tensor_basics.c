#include "cerebrics/cerebrics.h"
#include <stdio.h>

int main(void) {
    int shape[2] = {2, 3};
    CBR_Tensor *x = cbr_tensor_create(2, shape, 0);

    if (!x) {
        printf("failed to create tensor\n");
        return 1;
    }

    cbr_tensor_set(x, (int[]){0, 0}, 1.0F);
    cbr_tensor_set(x, (int[]){0, 1}, 2.0F);
    cbr_tensor_set(x, (int[]){0, 2}, 3.0F);
    cbr_tensor_set(x, (int[]){1, 0}, 4.0F);
    cbr_tensor_set(x, (int[]){1, 1}, 5.0F);
    cbr_tensor_set(x, (int[]){1, 2}, 6.0F);

    float value = 0.0F;
    CBR_Status status = cbr_tensor_get(x, (int[]){1, 2}, &value);
    if (status == CBR_OK) {
        printf("shape=(%d,%d), num_el=%d\n", cbr_shape(x, 0), cbr_shape(x, 1), cbr_num_el(x));
        printf("x[1,2]=%.1f\n", value);
    }

    status = cbr_tensor_get(x, (int[]){2, 0}, &value);
    printf("x[2,0] status=%s\n", cbr_status_string(status));

    cbr_tensor_free(x);
    return 0;
}
