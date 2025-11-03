// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xmmult_accel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XMmult_accel_CfgInitialize(XMmult_accel *InstancePtr, XMmult_accel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XMmult_accel_Start(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL) & 0x80; // bit 7  - auto_restart (Read/Write)
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL, Data | 0x01); // bit 0  - ap_start (Read/Write/COH)
}

u32 XMmult_accel_IsDone(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XMmult_accel_IsIdle(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XMmult_accel_IsReady(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XMmult_accel_EnableAutoRestart(XMmult_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XMmult_accel_DisableAutoRestart(XMmult_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XMmult_accel_Set_A(XMmult_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_A_DATA, (u32)(Data));
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_A_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_accel_Get_A(XMmult_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_A_DATA);
    Data += (u64)XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_A_DATA + 4) << 32;
    return Data;
}

void XMmult_accel_Set_B(XMmult_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_B_DATA, (u32)(Data));
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_B_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_accel_Get_B(XMmult_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_B_DATA);
    Data += (u64)XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_B_DATA + 4) << 32;
    return Data;
}

void XMmult_accel_Set_C(XMmult_accel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_C_DATA, (u32)(Data));
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_C_DATA + 4, (u32)(Data >> 32));
}

u64 XMmult_accel_Get_C(XMmult_accel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_C_DATA);
    Data += (u64)XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_C_DATA + 4) << 32;
    return Data;
}

void XMmult_accel_Set_N(XMmult_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_N_DATA, Data);
}

u32 XMmult_accel_Get_N(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_N_DATA);
    return Data;
}

void XMmult_accel_Set_K(XMmult_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_K_DATA, Data);
}

u32 XMmult_accel_Get_K(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_K_DATA);
    return Data;
}

void XMmult_accel_Set_M(XMmult_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_M_DATA, Data);
}

u32 XMmult_accel_Get_M(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_M_DATA);
    return Data;
}

void XMmult_accel_Set_update_A(XMmult_accel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_UPDATE_A_DATA, Data);
}

u32 XMmult_accel_Get_update_A(XMmult_accel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_UPDATE_A_DATA);
    return Data;
}

void XMmult_accel_InterruptGlobalEnable(XMmult_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_GIE, 1);
}

void XMmult_accel_InterruptGlobalDisable(XMmult_accel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_GIE, 0);
}

void XMmult_accel_InterruptEnable(XMmult_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_IER);
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_IER, Register | Mask);
}

void XMmult_accel_InterruptDisable(XMmult_accel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_IER);
    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XMmult_accel_InterruptClear(XMmult_accel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XMmult_accel_WriteReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_ISR, Mask);
}

u32 XMmult_accel_InterruptGetEnabled(XMmult_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_IER);
}

u32 XMmult_accel_InterruptGetStatus(XMmult_accel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XMmult_accel_ReadReg(InstancePtr->Control_BaseAddress, XMMULT_ACCEL_CONTROL_ADDR_ISR);
}


/*===================================================================================*/

