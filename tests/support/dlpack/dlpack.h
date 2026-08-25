#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensorVersioned {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensorVersioned* self);
    uint64_t flags;
    uint32_t major;
    uint32_t minor;
} DLManagedTensorVersioned;

#ifdef __cplusplus
}
#endif

#endif
