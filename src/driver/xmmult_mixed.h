// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XMMULT_MIXED_H
#define XMMULT_MIXED_H

#ifdef __cplusplus
extern "C" {
#endif

    /***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xmmult_mixed_hw.h"
#include "memory.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
    typedef uint8_t u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;
#else
    typedef struct {
#ifdef SDT
        char *Name;
#else
        u16 DeviceId;
#endif
        u64 Control_BaseAddress;
    } XMmult_mixed_Config;
#endif

    // typedef struct {
    //     u64 Control_BaseAddress;
    //     u32 IsReady;
    // } XMmult_mixed;
    typedef struct {
        u64 Control_BaseAddress;
        u32 IsReady;
        struct dma_memory dma_A;
        struct dma_memory dma_B;
        struct dma_memory dma_C;
    } XMmult_mixed;

    typedef u32 word_type;

    /***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XMmult_mixed_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XMmult_mixed_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XMmult_mixed_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XMmult_mixed_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
    int XMmult_mixed_Initialize(XMmult_mixed *InstancePtr, UINTPTR BaseAddress);
    XMmult_mixed_Config *XMmult_mixed_LookupConfig(UINTPTR BaseAddress);
#else
    int XMmult_mixed_Initialize(XMmult_mixed *InstancePtr, u16 DeviceId);
    XMmult_mixed_Config *XMmult_mixed_LookupConfig(u16 DeviceId);
#endif
    int XMmult_mixed_CfgInitialize(XMmult_mixed *InstancePtr, XMmult_mixed_Config *ConfigPtr);
#else
    int XMmult_mixed_Initialize(XMmult_mixed *InstancePtr, const char *InstanceName);
    int XMmult_mixed_Release(XMmult_mixed *InstancePtr);
#endif

    void XMmult_mixed_Start(XMmult_mixed *InstancePtr);
    u32 XMmult_mixed_IsDone(XMmult_mixed *InstancePtr);
    u32 XMmult_mixed_IsIdle(XMmult_mixed *InstancePtr);
    u32 XMmult_mixed_IsReady(XMmult_mixed *InstancePtr);
    void XMmult_mixed_EnableAutoRestart(XMmult_mixed *InstancePtr);
    void XMmult_mixed_DisableAutoRestart(XMmult_mixed *InstancePtr);

    void XMmult_mixed_Set_A_mem(XMmult_mixed *InstancePtr, u64 Data);
    u64 XMmult_mixed_Get_A_mem(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_B_mem(XMmult_mixed *InstancePtr, u64 Data);
    u64 XMmult_mixed_Get_B_mem(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_C_mem(XMmult_mixed *InstancePtr, u64 Data);
    u64 XMmult_mixed_Get_C_mem(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_N(XMmult_mixed *InstancePtr, u32 Data);
    u32 XMmult_mixed_Get_N(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_K(XMmult_mixed *InstancePtr, u32 Data);
    u32 XMmult_mixed_Get_K(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_M(XMmult_mixed *InstancePtr, u32 Data);
    u32 XMmult_mixed_Get_M(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_mode(XMmult_mixed *InstancePtr, u32 Data);
    u32 XMmult_mixed_Get_mode(XMmult_mixed *InstancePtr);
    void XMmult_mixed_Set_update_A(XMmult_mixed *InstancePtr, u32 Data);
    u32 XMmult_mixed_Get_update_A(XMmult_mixed *InstancePtr);

    void XMmult_mixed_InterruptGlobalEnable(XMmult_mixed *InstancePtr);
    void XMmult_mixed_InterruptGlobalDisable(XMmult_mixed *InstancePtr);
    void XMmult_mixed_InterruptEnable(XMmult_mixed *InstancePtr, u32 Mask);
    void XMmult_mixed_InterruptDisable(XMmult_mixed *InstancePtr, u32 Mask);
    void XMmult_mixed_InterruptClear(XMmult_mixed *InstancePtr, u32 Mask);
    u32 XMmult_mixed_InterruptGetEnabled(XMmult_mixed *InstancePtr);
    u32 XMmult_mixed_InterruptGetStatus(XMmult_mixed *InstancePtr);
    XMmult_mixed *xmmult_mixed_device_init(const char *pci_addr);
    int xmmult_accel_execute(XMmult_mixed *InstancePtr, const uintptr_t A, const uintptr_t B, const uintptr_t C,
        const int N, const int K, const int M, const int mode, const int updataA,
        const size_t A_size, const size_t B_size, const size_t C_size);

#ifdef __cplusplus
}
#endif

#endif
