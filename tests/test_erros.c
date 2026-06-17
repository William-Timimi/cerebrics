#include "cerebrics/cerebrics.h"
#include <assert.h>
#include <string.h>

int main(void) {
    assert(strcmp(cbr_status_string(CBR_OK), "ok") == 0);
    assert(strcmp(cbr_status_string(CBR_ERR_INVALID_ARGUMENT), "invalid argument") == 0);
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    assert(strcmp(cbr_status_string((CBR_Status)999), "unknown error") == 0);
}
