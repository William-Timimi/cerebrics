#include "cerebrics/cerebrics.h"
#include <stdio.h>

static void print_tensor(const char *name, CBR_Tensor *t) {
    printf("%s: ", name);
    for (int i = 0; i < cbr_num_el(t); i++)
        printf("%.1f ", cbr_data(t)[i]);
    printf("\n");
}

int main(void) {
    int shape[2] = {2, 2};
    float a_data[4] = {1, 2, 3, 4};
    float b_data[4] = {10, 20, 30, 40};
    CBR_Tensor *a = cbr_tensor_from_data(a_data, 2, shape, 0);
    CBR_Tensor *b = cbr_tensor_from_data(b_data, 2, shape, 0);

    CBR_Tensor *sum = cbr_add(a, b);
    CBR_Tensor *sub = cbr_sub(a, b);
    CBR_Tensor *prod = cbr_mul(a, b);
    CBR_Tensor *half = cbr_scalmul(0.5f, a);

    CBR_Tensor *mm = cbr_matmul(a, b);
    CBR_Tensor *relu = cbr_relu(prod);
    CBR_Tensor *total = cbr_sum(relu);

    print_tensor("a+b", sum);
    print_tensor("a-b", sub);
    print_tensor("a*b", prod);
    print_tensor("0.5*a", half);
    print_tensor("a@b", mm);
    print_tensor("sum(relu(a*b))", total);

    cbr_tensor_free(total);
    cbr_tensor_free(relu);
    cbr_tensor_free(mm);
    cbr_tensor_free(half);
    cbr_tensor_free(prod);
    cbr_tensor_free(sub);
    cbr_tensor_free(sum);
    cbr_tensor_free(b);
    cbr_tensor_free(a);
    return 0;
}
