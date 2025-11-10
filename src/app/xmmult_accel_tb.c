#include <stdio.h>
#include <log.h>
#include "memory.h"

#include "driver/xmmult_accel.h"
#define DTYPE_IN int8_t
#define DTYPE_OUT int32_t

int main(int argc, char *argv[]) {
    XMmult_accel *InstancePtr = xmmult_accel_device_init(argv[1]);
    int N = 6, K = 8, M = 10;
    DTYPE_IN A[N][K], B[K][M];
    DTYPE_OUT C[N][M];
    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A[i][j] = i + j;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = i - j;
        }
    }
    // debug info
    debug("debug info from testbench:\n");
    debug("Pointer A: %p\n", A);
    debug("Pointer B: %p\n", B);
    debug("Pointer C: %p\n", C);
    // Perform matrix multiplication using the accelerator


    xmmult_accel_execute(InstancePtr, (uintptr_t) A, (uintptr_t) B, (uintptr_t) C, N, K, M, 1);
    // Check results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int expected = 0;
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
    if (errors == 0) {
        printf("Matrix multiplication successful, no errors found.\n");
    } else {
        printf("Matrix multiplication completed with %d errors.\n", errors);
    }
    // printf A, B, C beautifully
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
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%6d ", C[i][j]);
        }
        printf("\n");
    }
}