#ifndef CEREBRICS_ERRORS_H
#define CEREBRICS_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CBR_OK = 0,
    CBR_ERR_INVALID_ARGUMENT = 1,
    CBR_ERR_OUT_OF_BOUNDS = 2,
    CBR_ERR_SHAPE = 3,
    CBR_ERR_ALLOCATION = 4,
    CBR_ERR_UNSUPPORTED = 5
} CBR_Status;

const char *cbr_status_string(CBR_Status status);

#ifdef __cplusplus
}
#endif
#endif
