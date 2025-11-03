#include <stdio.h>

#include "driver/xmmult_accel.h"
int main(int argc, char *argv[]) {
    XMmult_accel *InstancePtr = xmmult_accel_device_init(argv[1]);
    int N = 6, K = 8, M = 10;
    int A[N][K], B[K][M], C[N][M];
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
}