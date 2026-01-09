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
#include "xmmult_accel.h"
#include "xmmult_accel_type.h"

#define MAX_N 64    // Maximum number of rows in matrix A and output C
#define MAX_K 768   // Maximum shared dimension between matrices A and B
#define MAX_M 768   // Maximum number of columns in matrix B and output C

static double get_time_diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

XMmult_accel *xmmult_accel_device_init(const char *pci_addr, size_t dsize_in, size_t dsize_out) {
    remove_driver(pci_addr);

    char iommu_path[PATH_MAX];
    snprintf(iommu_path, sizeof(iommu_path), "/sys/bus/pci/devices/%s/iommu_group", pci_addr);

    // 使用 access(..., F_OK) 检查文件是否存在
    if (access(iommu_path, F_OK) == 0) {
        // 文件存在，说明开启了 IOMMU，可以安全调用 vfio_init
        int vfio_fd = vfio_init(pci_addr);
        if (vfio_fd != -1) {
            printf("IOMMU/VFIO mode enabled. Container FD: %d\n", vfio_fd);
            set_vfio_container(vfio_fd);
        }
    } else {
        // 文件不存在，说明没开 IOMMU，跳过 vfio_init 以免程序崩溃
        printf("No IOMMU group found for device %s. Running in Legacy (Hugepages) mode.\n", pci_addr);
    }

    XMmult_accel *InstancePtr = calloc(1, sizeof(XMmult_accel));
    InstancePtr->Control_BaseAddress = (u64) pci_map_resource(pci_addr);
    InstancePtr->dma_A = memory_allocate_dma(MAX_N * MAX_K * sizeof(dsize_in), 1);
    InstancePtr->dma_B = memory_allocate_dma(MAX_K * MAX_M * sizeof(dsize_in), 1);
    InstancePtr->dma_C = memory_allocate_dma(MAX_N * MAX_M * sizeof(dsize_out), 1);

    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;
    XMmult_accel_InterruptGlobalDisable(InstancePtr, MMULT_FP16);
    XMmult_accel_DisableAutoRestart(InstancePtr, MMULT_FP16);

    XMmult_accel_InterruptGlobalDisable(InstancePtr, MMULT_INT8);
    XMmult_accel_DisableAutoRestart(InstancePtr, MMULT_INT8);

    _mm_mfence();
    return InstancePtr;
}
int xmmult_accel_execute(XMmult_accel *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
    int N, int K, int M, int updateA, size_t dsize_in, size_t dsize_out, uint64_t device_offset) {
    struct timespec t_start, t_memcpy_in, t_idle, t_setup, t_compute, t_memcpy_out;

    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // Copy input matrices to DMA buffers
    memcpy(InstancePtr->dma_A.virt, (void *) A, N * K * dsize_in);
    memcpy(InstancePtr->dma_B.virt, (void *) B, K * M * dsize_in);

    clock_gettime(CLOCK_MONOTONIC, &t_memcpy_in);

    // 1. Wait for Idle
    while (XMmult_accel_IsIdle(InstancePtr, device_offset) == 0);

    clock_gettime(CLOCK_MONOTONIC, &t_idle);

    // printf("xmmult_accel is idle, proceeding with execution.\n");
    // 2. Set parameters
    XMmult_accel_Set_N(InstancePtr, N, device_offset);
    XMmult_accel_Set_K(InstancePtr, K, device_offset);
    XMmult_accel_Set_M(InstancePtr, M, device_offset);
    XMmult_accel_Set_update_A(InstancePtr, updateA, device_offset);

    // 3. Set pointers
    XMmult_accel_Set_A(InstancePtr, (uintptr_t) InstancePtr->dma_A.phy, device_offset);
    XMmult_accel_Set_B(InstancePtr, (uintptr_t) InstancePtr->dma_B.phy, device_offset);
    XMmult_accel_Set_C(InstancePtr, (uintptr_t) InstancePtr->dma_C.phy, device_offset);

    _mm_mfence();

    clock_gettime(CLOCK_MONOTONIC, &t_setup);

    // printf("Parameters and pointers set, starting accelerator.\n");
    // 4. Start the accelerator
    XMmult_accel_Start(InstancePtr, device_offset);

    _mm_mfence();
    // printf("Accelerator started, waiting for completion.\n");
    // 5. Wait for Done
    while (XMmult_accel_IsDone(InstancePtr, device_offset) == 0);
    // printf("xmmult_accel execution completed.\n");

    clock_gettime(CLOCK_MONOTONIC, &t_compute);

    _mm_mfence();
    // 6. Copy result back to C
    memcpy((void *) C, InstancePtr->dma_C.virt, N * M * sizeof(dsize_out));

    clock_gettime(CLOCK_MONOTONIC, &t_memcpy_out);

    // === 计算各阶段耗时 ===
    double time_memcpy_in = get_time_diff_ms(t_start, t_memcpy_in);
    double time_wait_idle = get_time_diff_ms(t_memcpy_in, t_idle);
    double time_reg_config = get_time_diff_ms(t_idle, t_setup);
    double time_fpga_compute = get_time_diff_ms(t_setup, t_compute);
    double time_memcpy_out = get_time_diff_ms(t_compute, t_memcpy_out);
    double time_total = get_time_diff_ms(t_start, t_memcpy_out);

    // === 打印耗时统计与百分比 ===
    printf("=== C Driver Profiling (N=%d, K=%d, M=%d) ===\n", N, K, M);
    printf("  Memcpy Host->DMA: %8.3f ms (%6.2f%%)\n", time_memcpy_in, (time_memcpy_in / time_total) * 100.0);
    printf("  Wait Idle:        %8.3f ms (%6.2f%%)\n", time_wait_idle, (time_wait_idle / time_total) * 100.0);
    printf("  Reg Config:       %8.3f ms (%6.2f%%)\n", time_reg_config, (time_reg_config / time_total) * 100.0);
    printf("  FPGA Compute:     %8.3f ms (%6.2f%%)\n", time_fpga_compute, (time_fpga_compute / time_total) * 100.0);
    printf("  Memcpy DMA->Host: %8.3f ms (%6.2f%%)\n", time_memcpy_out, (time_memcpy_out / time_total) * 100.0);
    printf("  ----------------------------------------\n");
    printf("  Total C Time:     %8.3f ms (100.00%%)\n", time_total);
    printf("=============================================\n");


    return 0;

}