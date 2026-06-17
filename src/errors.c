#include "cerebrics/errors.h"

const char *cbr_status_string(CBR_Status status) {
    switch (status) {
    case CBR_OK:
        return "ok";
    case CBR_ERR_INVALID_ARGUMENT:
        return "invalid argument";
    case CBR_ERR_OUT_OF_BOUNDS:
        return "out of bounds";
    case CBR_ERR_SHAPE:
        return "shape error";
    case CBR_ERR_ALLOCATION:
        return "allocation failed";
    case CBR_ERR_UNSUPPORTED:
        return "unsupported operation";
    default:
        return "unknown error";
    }
}
