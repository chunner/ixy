#include <stdio.h>
#include <log.h>
#include "memory.h"
#include "half.h"

#include "driver/xmmult_accel.h"

const char *pci_addr = "0000:00:04.0";
#define N 6
#define K 8
#define M 10

int main() {
    int errors = 0;
    errors += test_mmult_accel_int8();
    errors += test_mmult_accel_half();
    if (errors == 0) {
        printf("All tests passed successfully.\n");
    } else {
        printf("Some tests failed with a total of %d errors.\n", errors);
    }
    return errors;
}

int test_mmult_accel_int8() {
    XMmult_accel *InstancePtr = xmmult_accel_device_init(pci_addr, sizeof(int8_t), sizeof(int32_t));
    int8_t A[N][K], B[K][M];
    int32_t C[N][M];
    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = rand() % 10;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = rand() % 10;
        }
    }
    // Perform matrix multiplication using the accelerator
    xmmult_accel_execute(InstancePtr, (uintptr_t) A, (uintptr_t) B, (uintptr_t) C, N, K, M, 1, sizeof(int8_t), sizeof(int32_t), 0x1000);
    // Check results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int32_t expected = 0;
            for (int k = 0; k < K; k++) {
                expected += A[i][k] * B[k][j];
            }
            if (C[i][j] != expected) {
                errors++;
                if (errors < 10) {
                    printf("Error at C[%d][%d]: expected %d, got %d\n", i, j, expected, C[i][j]);
                }
            }
        }
    }
    // print results (A, B, C)
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%5d ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%5d ", B[i][j]);
        }
        printf("\n");
    }
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%5d ", C[i][j]);
        }
        printf("\n");
    }

    if (errors == 0) {
        printf("Matrix multiplication(INT_8) successful, no errors found.\n");
    } else {
        printf("Matrix multiplication completed with %d errors.\n", errors);
    }
    printf("==============Test INT8 done=================\n");
    return errors;
}

int test_mmult_accel_half() {
    XMmult_accel *InstancePtr = xmmult_accel_device_init(pci_addr, sizeof(half), sizeof(float));
    half A[N][K], B[K][M];
    float C[N][M];
    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = float_to_half((float) (rand() % 100) / 10.0f);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = float_to_half((float) (rand() % 100) / 10.0f);
        }
    }
    // Perform matrix multiplication using the accelerator
    xmmult_accel_execute(InstancePtr, (uintptr_t) A, (uintptr_t) B, (uintptr_t) C, N, K, M, 1, sizeof(half), sizeof(float), 0x0000);
    // Check results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += half_to_float(A[i][k]) * half_to_float(B[k][j]);
            }
            if (fabs(C[i][j] - expected) > 0.1f) {
                errors++;
                if (errors < 10) {
                    printf("Error at C[%d][%d]: expected %f, got %f\n", i, j, expected, C[i][j]);
                }
            }
        }
    }
    // print results (A, B, C)
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%6.2f ", half_to_float(A[i][j]));
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%6.2f ", half_to_float(B[i][j]));
        }
        printf("\n");
    }
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%6.2f ", C[i][j]);
        }
        printf("\n");
    }
    if (errors == 0) {
        printf("Matrix multiplication(HALF) successful, no errors found.\n");
    } else {
        printf("Matrix multiplication completed with %d errors.\n", errors);
    }
    printf("==============Test HALF done=================\n");
    return errors;
}