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
    errors += test_mmult_accel_int4();
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
#define bus_t uint64_t
static int IN_PER_WORD = 16;
static inline void store_int4_packed(bus_t* dst_words, int linear_index, int8_t v_signed)
{
    // clamp to int4
    if (v_signed < -8) v_signed = -8;
    if (v_signed >  7) v_signed =  7;

    int word_idx = linear_index / IN_PER_WORD;
    int nib_idx  = linear_index % IN_PER_WORD;
    int shift    = nib_idx * 4;

    bus_t w = dst_words[word_idx];
    w &= ~((bus_t)0xFULL << shift);
    w |=  ((bus_t)((uint8_t)v_signed & 0xFu) << shift);
    dst_words[word_idx] = w;
}

int test_mmult_accel_int4() {
    XMmult_accel *InstancePtr = xmmult_accel_device_init(pci_addr, 0.5, sizeof(int32_t));
    int8_t A_ref[N][K], B_ref[K][M];
    int32_t C[N][M];
    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            A_ref[i][j] = rand() % 16 - 8; // int4 range [-8, 7]
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            B_ref[i][j] = rand() % 16 - 8; // int4 range [-8, 7]
        }
    }
    // Pack matrices A and B into int4 format
    bus_t A_packed[(N * K + IN_PER_WORD - 1) / IN_PER_WORD];
    bus_t B_packed[(K * M + IN_PER_WORD - 1) / IN_PER_WORD];
    // Initialize packed arrays to zero
    for (int i = 0; i < (N * K + IN_PER_WORD - 1) / IN_PER_WORD; i++) {
        A_packed[i] = 0;
    }
    for (int i = 0; i < (K * M + IN_PER_WORD - 1) / IN_PER_WORD; i++) {
        B_packed[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            store_int4_packed(A_packed, i * K + j, A_ref[i][j]);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            store_int4_packed(B_packed, i * M + j, B_ref[i][j]);
        }
    }
    // Perform matrix multiplication using the accelerator
    xmmult_accel_execute(InstancePtr, (uintptr_t) A_packed, (uintptr_t) B_packed, (uintptr_t) C, N, K, M, 1, 0.5, sizeof(int32_t), 0x2000);
    // Check results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int32_t expected = 0;
            for (int k = 0; k < K; k++) {
                expected += A_ref[i][k] * B_ref[k][j];
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
    printf("Matrix A (int4):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%3d ", A_ref[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B (int4):\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            printf("%3d ", B_ref[i][j]);
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
        printf("Matrix multiplication(INT_4) successful, no errors found.\n");
    } else {
        printf("Matrix multiplication completed with %d errors.\n", errors);
    }
    printf("==============Test INT4 done=================\n");
    return errors;
}