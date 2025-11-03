#include <emmintrin.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#include <linux/limits.h>
#include <sys/stat.h>

#include "driver/device.h"
#include "log.h"
#include "memory.h"
#include "pci.h"
#include "xmmult_accel.h"
#include "xmmult_accel_type.h"


XMmult_accel *xmmult_accel_device_init(const char *pci_addr) {
    remove_driver(pci_addr);
    XMmult_accel *InstancePtr = calloc(1, sizeof(XMmult_accel));
    InstancePtr->Control_BaseAddress = (u64) pci_map_resource(pci_addr);
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;
    return InstancePtr;
}
int xmmult_accel_execute(XMmult_accel *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
    int N, int K, int M, int updateA) {
    // 1. Wait for Idle
    while (XMmult_accel_IsIdle(InstancePtr) == 0);

    // 2. Set parameters
    XMmult_accel_Set_N(InstancePtr, N);
    XMmult_accel_Set_K(InstancePtr, K);
    XMmult_accel_Set_M(InstancePtr, M);
    XMmult_accel_Set_update_A(InstancePtr, updateA);

    // 3. Set pointers
    XMmult_accel_Set_A(InstancePtr, (uintptr_t) A);
    XMmult_accel_Set_B(InstancePtr, (uintptr_t) B);
    XMmult_accel_Set_C(InstancePtr, (uintptr_t) C);

    // 4. Start the accelerator
    XMmult_accel_Start(InstancePtr);

    // 5. Wait for Done
    while (XMmult_accel_IsDone(InstancePtr) == 0);

    return 0;

}