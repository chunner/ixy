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

#define MAX_N 64    // Maximum number of rows in matrix A and output C
#define MAX_K 768   // Maximum shared dimension between matrices A and B
#define MAX_M 768   // Maximum number of columns in matrix B and output C

XMmult_accel *xmmult_accel_device_init(const char *pci_addr) {
    remove_driver(pci_addr);
    XMmult_accel *InstancePtr = calloc(1, sizeof(XMmult_accel));
    InstancePtr->Control_BaseAddress = (u64) pci_map_resource(pci_addr);
    InstancePtr->dma_A = memory_allocate_dma(MAX_N * MAX_K * sizeof(int8_t), 1);
    InstancePtr->dma_B = memory_allocate_dma(MAX_K * MAX_M * sizeof(int8_t), 1);
    InstancePtr->dma_C = memory_allocate_dma(MAX_N * MAX_M * sizeof(int32_t), 1);

    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;
    XMmult_accel_InterruptGlobalDisable(InstancePtr);
    XMmult_accel_DisableAutoRestart(InstancePtr);
    _mm_mfence();
    return InstancePtr;
}
int xmmult_accel_execute(XMmult_accel *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
    int N, int K, int M, int updateA) {
    // debug info
    printf("xmmult_accel_execute called with parameters:\n");
    printf("  A: %p\n", (void *) A);
    printf("  B: %p\n", (void *) B);
    printf("  C: %p\n", (void *) C);
    printf("  N: %d\n", N);
    printf("  K: %d\n", K);
    printf("  M: %d\n", M);
    printf("  updateA: %d\n", updateA);

    // uintptr_t A_phy = virt_to_phys(A);
    // uintptr_t B_phy = virt_to_phys(B);
    // uintptr_t C_phy = virt_to_phys(C);
    // printf("Physical Address A: %p\n", A_phy);
    // printf("Physical Address B: %p\n", B_phy);
    // printf("Physical Address C: %p\n", C_phy);

    // struct dma_memory dma_A = memory_allocate_dma(N * K * sizeof(int8_t), 1);
    // struct dma_memory dma_B = memory_allocate_dma(K * M * sizeof(int8_t), 1);
    // struct dma_memory dma_C = memory_allocate_dma(N * M * sizeof(int32_t), 0);

    // Copy input matrices to DMA buffers
    memcpy(InstancePtr->dma_A.virt, (void *) A, N * K * sizeof(int8_t));
    memcpy(InstancePtr->dma_B.virt, (void *) B, K * M * sizeof(int8_t));

    // 1. Wait for Idle
    while (XMmult_accel_IsIdle(InstancePtr) == 0);

    printf("xmmult_accel is idle, proceeding with execution.\n");
    // 2. Set parameters
    XMmult_accel_Set_N(InstancePtr, N);
    XMmult_accel_Set_K(InstancePtr, K);
    XMmult_accel_Set_M(InstancePtr, M);
    XMmult_accel_Set_update_A(InstancePtr, updateA);

    // 3. Set pointers
    XMmult_accel_Set_A(InstancePtr, (uintptr_t) InstancePtr->dma_A.phy);
    XMmult_accel_Set_B(InstancePtr, (uintptr_t) InstancePtr->dma_B.phy);
    XMmult_accel_Set_C(InstancePtr, (uintptr_t) InstancePtr->dma_C.phy);

    _mm_mfence();

    printf("Parameters and pointers set, starting accelerator.\n");
    // 4. Start the accelerator
    XMmult_accel_Start(InstancePtr);

    _mm_mfence();
    printf("Accelerator started, waiting for completion.\n");
    // 5. Wait for Done
    while (XMmult_accel_IsDone(InstancePtr) == 0);
    printf("xmmult_accel execution completed.\n");

    _mm_mfence(); 
    // 6. Copy result back to C
    memcpy((void *) C, InstancePtr->dma_C.virt, N * M * sizeof(int32_t));
    

    return 0;

}