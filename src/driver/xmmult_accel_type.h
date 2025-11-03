// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)
// Tool Version Limit: 2024.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of A
//        bit 31~0 - A[31:0] (Read/Write)
// 0x14 : Data signal of A
//        bit 31~0 - A[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of B
//        bit 31~0 - B[31:0] (Read/Write)
// 0x20 : Data signal of B
//        bit 31~0 - B[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of C
//        bit 31~0 - C[31:0] (Read/Write)
// 0x2c : Data signal of C
//        bit 31~0 - C[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of N
//        bit 31~0 - N[31:0] (Read/Write)
// 0x38 : reserved
// 0x3c : Data signal of K
//        bit 31~0 - K[31:0] (Read/Write)
// 0x40 : reserved
// 0x44 : Data signal of M
//        bit 31~0 - M[31:0] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of update_A
//        bit 31~0 - update_A[31:0] (Read/Write)
// 0x50 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XMMULT_ACCEL_CONTROL_ADDR_AP_CTRL       0x00
#define XMMULT_ACCEL_CONTROL_ADDR_GIE           0x04
#define XMMULT_ACCEL_CONTROL_ADDR_IER           0x08
#define XMMULT_ACCEL_CONTROL_ADDR_ISR           0x0c
#define XMMULT_ACCEL_CONTROL_ADDR_A_DATA        0x10
#define XMMULT_ACCEL_CONTROL_BITS_A_DATA        64
#define XMMULT_ACCEL_CONTROL_ADDR_B_DATA        0x1c
#define XMMULT_ACCEL_CONTROL_BITS_B_DATA        64
#define XMMULT_ACCEL_CONTROL_ADDR_C_DATA        0x28
#define XMMULT_ACCEL_CONTROL_BITS_C_DATA        64
#define XMMULT_ACCEL_CONTROL_ADDR_N_DATA        0x34
#define XMMULT_ACCEL_CONTROL_BITS_N_DATA        32
#define XMMULT_ACCEL_CONTROL_ADDR_K_DATA        0x3c
#define XMMULT_ACCEL_CONTROL_BITS_K_DATA        32
#define XMMULT_ACCEL_CONTROL_ADDR_M_DATA        0x44
#define XMMULT_ACCEL_CONTROL_BITS_M_DATA        32
#define XMMULT_ACCEL_CONTROL_ADDR_UPDATE_A_DATA 0x4c
#define XMMULT_ACCEL_CONTROL_BITS_UPDATE_A_DATA 32


