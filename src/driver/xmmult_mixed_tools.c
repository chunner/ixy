// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xmmult_mixed.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XMmult_mixed_CfgInitialize(XMmult_mixed *InstancePtr, XMmult_mixed_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XMmult_mixed_Start(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL) & 0x80;
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XMmult_mixed_IsDone(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XMmult_mixed_IsIdle(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XMmult_mixed_IsReady(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XMmult_mixed_EnableAutoRestart(XMmult_mixed *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XMmult_mixed_DisableAutoRestart(XMmult_mixed *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_AP_CTRL, 0);
}

void XMmult_mixed_Set_A_mem(XMmult_mixed *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_A_MEM_DATA, (u32)(Data));
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_A_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_mixed_Get_A_mem(XMmult_mixed *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_A_MEM_DATA);
    Data += (u64)XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_A_MEM_DATA + 4) << 32;
    return Data;
}

void XMmult_mixed_Set_B_mem(XMmult_mixed *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_B_MEM_DATA, (u32)(Data));
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_B_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_mixed_Get_B_mem(XMmult_mixed *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_B_MEM_DATA);
    Data += (u64)XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_B_MEM_DATA + 4) << 32;
    return Data;
}

void XMmult_mixed_Set_C_mem(XMmult_mixed *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_C_MEM_DATA, (u32)(Data));
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_C_MEM_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_mixed_Get_C_mem(XMmult_mixed *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_C_MEM_DATA);
    Data += (u64)XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_C_MEM_DATA + 4) << 32;
    return Data;
}

void XMmult_mixed_Set_N(XMmult_mixed *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_N_DATA, Data);
}

u32 XMmult_mixed_Get_N(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_N_DATA);
    return Data;
}

void XMmult_mixed_Set_K(XMmult_mixed *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_K_DATA, Data);
}

u32 XMmult_mixed_Get_K(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_K_DATA);
    return Data;
}

void XMmult_mixed_Set_M(XMmult_mixed *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_M_DATA, Data);
}

u32 XMmult_mixed_Get_M(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_M_DATA);
    return Data;
}

void XMmult_mixed_Set_mode(XMmult_mixed *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_MODE_DATA, Data);
}

u32 XMmult_mixed_Get_mode(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_MODE_DATA);
    return Data;
}

void XMmult_mixed_Set_update_A(XMmult_mixed *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_UPDATE_A_DATA, Data);
}

u32 XMmult_mixed_Get_update_A(XMmult_mixed *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_UPDATE_A_DATA);
    return Data;
}

void XMmult_mixed_InterruptGlobalEnable(XMmult_mixed *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_GIE, 1);
}

void XMmult_mixed_InterruptGlobalDisable(XMmult_mixed *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_GIE, 0);
}

void XMmult_mixed_InterruptEnable(XMmult_mixed *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_IER);
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_IER, Register | Mask);
}

void XMmult_mixed_InterruptDisable(XMmult_mixed *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_IER);
    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_IER, Register & (~Mask));
}

void XMmult_mixed_InterruptClear(XMmult_mixed *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_mixed_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_ISR, Mask);
}

u32 XMmult_mixed_InterruptGetEnabled(XMmult_mixed *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_IER);
}

u32 XMmult_mixed_InterruptGetStatus(XMmult_mixed *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMmult_mixed_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_MIXED_CONTROL_ADDR_ISR);
}

