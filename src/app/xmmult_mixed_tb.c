#include <stdio.h>
#include <log.h>
#include "memory.h"
#include "half.h"

#include "driver/xmmult_mixed.h"

const char *pci_addr = "0000:00:04.0";
#define N 6
#define K 8
#define M 10
#define MODE_INT8 0
#define MODE_INT4 1
#define MODE_FP16 2
#define MODE_FP32 3

XMmult_mixed *InstancePtr = NULL;

int test_mmult_mixed_int8() {
    int8_t A[N][K], B[K][M];
    int32_t C_hw[N][M], C_sw[N][M];
    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = (rand() % 256) - 128; // Random int8_t value
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = (rand() % 256) - 128; // Random int8_t value
        }
    }
    // Perform matrix multiplication in software
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C_sw[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C_sw[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    // Perform matrix multiplication in hardware
    xmmult_mixed_execute(InstancePtr, (uintptr_t *) A, (uintptr_t *) B, (uintptr_t *) C_hw, N, K, M,
        MODE_INT8, // mode
        1, // updataA
        N * K * sizeof(int8_t), // sizeA
        K * M * sizeof(int8_t), // sizeB
        N * M * sizeof(int32_t)  // sizeC
        );
    // print results (A, B, C)
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4d ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%4d ", B[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Software):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8d ", C_sw[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Hardware):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8d ", C_hw[i][j]);
        }
        printf("\n");
    }
    // Compare results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (C_sw[i][j] != C_hw[i][j]) {
                printf("Mismatch at C[%d][%d]: SW=%d, HW=%d", i, j, C_sw[i][j], C_hw[i][j]);
                return -1;
            }
        }
    }

    printf("===============INT8 matrix multiplication test passed.\n");
    return 0;
}
#define bus_t __int128_t
static int IN_PER_BUS = 32; // number of int4_t per bus_t
static inline void store_int4_to_bus(bus_t *bus_ptr, int linear_index, int8_t vsinged) {
    // clamp to int4
    if (vsinged > 7) vsinged = 7;
    if (vsinged < -8) vsinged = -8;

    int bus_idx = linear_index / IN_PER_BUS;
    int pos_in_bus = linear_index % IN_PER_BUS;
    int shift = pos_in_bus * 4;
    bus_t bus = bus_ptr[bus_idx];
    bus &= ~((bus_t)0xF << shift); // Clear the 4 bits at position
    bus |= ((bus_t) (vsinged & 0xF) << shift); // Set the new value
    bus_ptr[bus_idx] = bus;
}
int test_mmult_mixed_int4() {
    int8_t A_ref[N][K], B_ref[K][M];
    int32_t C_hw[N][M], C_sw[N][M];
    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A_ref[i][j] = (rand() % 16) - 8; // Random int4_t value
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B_ref[i][j] = (rand() % 16) - 8; // Random int4_t value
        }
    }
    // Pack int4_t matrices into int8_t matrices
    bus_t A_bus[(N * K + IN_PER_BUS - 1) / IN_PER_BUS];
    bus_t B_bus[(K * M + IN_PER_BUS - 1) / IN_PER_BUS];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            store_int4_to_bus(A_bus, i * K + j, A_ref[i][j]);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            store_int4_to_bus(B_bus, i * M + j, B_ref[i][j]);
        }
    }
    // Perform matrix multiplication in software
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C_sw[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C_sw[i][j] += A_ref[i][k] * B_ref[k][j];
            }
        }
    }
    // Perform matrix multiplication in hardware
    xmmult_mixed_execute(InstancePtr, (uintptr_t *) A_bus, (uintptr_t *) B_bus, (uintptr_t *) C_hw, N, K, M,
        MODE_INT4, // mode
        1, // updataA
        N * K / 2, // sizeA
        K * M / 2, // sizeB
        N * M * sizeof(int32_t)  // sizeC
    );
    // print results (A, B, C)
    printf("Matrix A (int4):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%4d ", A_ref[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B (int4):\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%4d ", B_ref[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Software):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8d ", C_sw[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Hardware):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8d ", C_hw[i][j]);
        }
        printf("\n");
    }

    // Compare results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (C_sw[i][j] != C_hw[i][j]) {
                printf("Mismatch at C[%d][%d]: SW=%d, HW=%d", i, j, C_sw[i][j], C_hw[i][j]);
                return -1;
            }
        }
    }
    printf("===============INT4 matrix multiplication test passed.\n");
    return 0;
}

int test_mmult_mixed_fp16() {
    half A[N][K], B[K][M];
    float C_hw[N][M], C_sw[N][M];
    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = float_to_half(((float)rand() / RAND_MAX) * 2.0f - 1.0f); // Random half value
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = float_to_half(((float)rand() / RAND_MAX) * 2.0f - 1.0f); // Random half value
        }
    }
    // Perform matrix multiplication in software
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C_sw[i][j] = 0.0f;
            for (int k = 0; k < K; k++) {
                C_sw[i][j] += half_to_float(A[i][k]) * half_to_float(B[k][j]);
            }
        }
    }
    // Perform matrix multiplication in hardware
    xmmult_mixed_execute(InstancePtr, (uintptr_t *) A, (uintptr_t *) B, (uintptr_t *) C_hw, N, K, M,
        MODE_FP16, // mode
        1, // updataA
        N * K * sizeof(half), // sizeA
        K * M * sizeof(half), // sizeB
        N * M * sizeof(float)  // sizeC
    );
    // Compare results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float diff = C_sw[i][j] - C_hw[i][j];
            if (diff < -0.01f || diff > 0.01f) {
                printf("Mismatch at C[%d][%d]: SW=%f, HW=%f\n", i, j, C_sw[i][j], C_hw[i][j]);
                return -1;
            }
        }
    }
    printf("===============FP16 matrix multiplication test passed.\n");
    return 0;
}
int test_mmult_mixed_fp32() {
    float A[N][K], B[K][M], C_hw[N][M], C_sw[N][M];
    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float value
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float value
        }
    }
    // Perform matrix multiplication in software
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C_sw[i][j] = 0.0f;
            for (int k = 0; k < K; k++) {
                C_sw[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    // Perform matrix multiplication in hardware
    xmmult_mixed_execute(InstancePtr, (uintptr_t *) A, (uintptr_t *) B, (uintptr_t *) C_hw, N, K, M,
        MODE_FP32, // mode
        1, // updataA
        N * K * sizeof(float), // sizeA
        K * M * sizeof(float), // sizeB
        N * M * sizeof(float)  // sizeC
    );
    // print results (A, B, C)
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%8.4f ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8.4f ", B[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Software):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%10.4f ", C_sw[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C (Hardware):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%10.4f ", C_hw[i][j]);
        }
        printf("\n");
    }
    // Compare results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float diff = C_sw[i][j] - C_hw[i][j];
            if (diff < -0.0001f || diff > 0.0001f) {
                printf("Mismatch at C[%d][%d]: SW=%f, HW=%f\n", i, j, C_sw[i][j], C_hw[i][j]);
                return -1;
            }
        }
    }
    printf("===============FP32 matrix multiplication test passed.\n");
}
int main() {
    InstancePtr = xmmult_mixed_device_init(pci_addr);
    test_mmult_mixed_int8();
    test_mmult_mixed_int4();
    test_mmult_mixed_fp16();
    test_mmult_mixed_fp32();
}