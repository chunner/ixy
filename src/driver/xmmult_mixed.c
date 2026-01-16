#include <emmintrin.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>
#include <time.h>

#include <linux/limits.h>
#include <sys/stat.h>

#include "driver/device.h"
#include "log.h"
#include "memory.h"
#include "pci.h"
#include "xmmult_mixed.h"
#include "xmmult_mixed_hw.h"

typedef __uint128_t MEM_TYPE;

#define MAX_N 64
#define MAX_K 3072
#define MAX_M 3072
#define MAX_A 64 * 3072 * 4 // 768 KB
#define MAX_B 768 * 3072 * 4 // 9 MB
#define MAX_C 64 * 3072 * 4 // 768 KB

XMmult_mixed *xmmult_mixed_device_init(const char *pci_addr) {
    ixy_log_init("xmmult_mixed.log");
    remove_driver(pci_addr);

    char iommu_path[PATH_MAX];
    snprintf(iommu_path, sizeof(iommu_path), "/sys/bus/pci/devices/%s/iommu_group", pci_addr);
    // 使用 access(..., F_OK) 检查文件是否存在
    if (access(iommu_path, F_OK) == 0) {
        // 文件存在，说明开启了 IOMMU，可以安全调用 vfio_init
        int vfio_fd = vfio_init(pci_addr);
        if (vfio_fd != -1) {
            info("IOMMU/VFIO mode enabled. Container FD: %d", vfio_fd);
            set_vfio_container(vfio_fd);
        }
    } else {
        // 文件不存在，说明没开 IOMMU，跳过 vfio_init 以免程序崩溃
        info("No IOMMU group found for device %s. Running in Legacy (Hugepages) mode.", pci_addr);
    }

    XMmult_mixed *InstancePtr = calloc(1, sizeof(XMmult_mixed));
    InstancePtr->Control_BaseAddress = (u64) pci_map_resource(pci_addr);
    InstancePtr->dma_A = memory_allocate_dma(2 * 1024 * 1024, 1); // 2 MB
    InstancePtr->dma_B = memory_allocate_dma(2 * 1024 * 1024, 1); // 2 MB
    InstancePtr->dma_C = memory_allocate_dma(2 * 1024 * 1024, 1); // 2 MB

    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;
    XMmult_mixed_InterruptGlobalDisable(InstancePtr);
    XMmult_mixed_DisableAutoRestart(InstancePtr);
    _mm_mfence();
    return InstancePtr;
}
int xmmult_mixed_execute(XMmult_mixed *InstancePtr, const uintptr_t A, const uintptr_t B, const uintptr_t C,
    const int N, const int K, const int M, const int mode, const int updateA,
    const size_t A_size, const size_t B_size, const size_t C_size) {
    // Copy data to device memory
    memcpy((void *) InstancePtr->dma_A.virt, (void *) A, A_size);
    memcpy((void *) InstancePtr->dma_B.virt, (void *) B, B_size);

    // 1. Wait for Idle
    while (!XMmult_mixed_IsIdle(InstancePtr));
    // 2. Set parameters
    XMmult_mixed_Set_A_mem(InstancePtr, InstancePtr->dma_A.phy);
    XMmult_mixed_Set_B_mem(InstancePtr, InstancePtr->dma_B.phy);
    XMmult_mixed_Set_C_mem(InstancePtr, InstancePtr->dma_C.phy);
    XMmult_mixed_Set_N(InstancePtr, N);
    XMmult_mixed_Set_K(InstancePtr, K);
    XMmult_mixed_Set_M(InstancePtr, M);
    XMmult_mixed_Set_mode(InstancePtr, mode);
    XMmult_mixed_Set_update_A(InstancePtr, updateA);
    _mm_mfence();
    // 3. Start the accelerator
    XMmult_mixed_Start(InstancePtr);
    _mm_mfence();
    // 4. Wait for Done
    while (!XMmult_mixed_IsDone(InstancePtr));
    _mm_mfence();
    // 5. Copy result back to host memory
    memcpy((void *) C, (void *) InstancePtr->dma_C.virt, C_size);
    return 0;
}