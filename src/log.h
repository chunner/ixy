#ifndef IXY_LOG_H
#define IXY_LOG_H

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

// 供外部初始化日志文件
FILE* ixy_log_fp(void);
int ixy_log_init(const char* path);
void ixy_log_close(void);

#ifndef NDEBUG
#define debug(fmt, ...) do {\
    fprintf(ixy_log_fp(), "[DEBUG] %s:%d %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__);\
} while(0)
#else
#define debug(fmt, ...) do {} while(0)
#undef assert
#define assert(expr) (void) (expr)
#endif

#define info(fmt, ...) do {\
    fprintf(ixy_log_fp(), "[INFO ] %s:%d %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__);\
} while(0)

#define warn(fmt, ...) do {\
    fprintf(ixy_log_fp(), "[WARN ] %s:%d %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__);\
} while(0)

#define error(fmt, ...) do {\
    fprintf(ixy_log_fp(), "[ERROR] %s:%d %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__);\
    abort();\
} while(0)

#define check_err(expr, op) ({\
    int64_t result = (int64_t) (expr);\
    if ((int64_t) result == -1LL) {\
        int err = errno;\
        char buf[512];\
        strerror_r(err, buf, sizeof(buf));\
        fprintf(ixy_log_fp(), "[ERROR] %s:%d %s(): Failed to %s: %s\n", __FILE__, __LINE__, __func__, op, buf);\
        exit(err);\
    }\
    result;\
})

static void hexdump(void* void_ptr, size_t len) {
    uint8_t* ptr = (uint8_t*) void_ptr;
    char ascii[17];
    for (uint32_t i = 0; i < len; i += 16) {
        fprintf(ixy_log_fp(), "%06x: ", i);
        int j = 0;
        for (; j < 16 && i + j < len; j++) {
            fprintf(ixy_log_fp(), "%02x", ptr[i + j]);
            if (j % 2) {
                fprintf(ixy_log_fp(), " ");
            }
            ascii[j] = isprint(ptr[i + j]) ? ptr[i + j] : '.';
        }
        ascii[j] = '\0';
        if (j < 16) {
            for (; j < 16; j++) {
                fprintf(ixy_log_fp(), "  ");
                if (j % 2) {
                    fprintf(ixy_log_fp(), " ");
                }
            }
        }
        fprintf(ixy_log_fp(), "  %s\n", ascii);
    }
}

#endif //IXY_LOG_H
