#include "log.h"

static FILE* g_ixy_log_fp = NULL;

FILE* ixy_log_fp(void) {
    return g_ixy_log_fp ? g_ixy_log_fp : stderr;
}

int ixy_log_init(const char* path) {
    if (!path || !*path) return -1;

    FILE* fp = fopen(path, "a"); // append
    if (!fp) return -1;

    // 行缓冲：每行立刻写入文件，便于崩溃时保留日志
    setvbuf(fp, NULL, _IOLBF, 0);

    g_ixy_log_fp = fp;
    return 0;
}

void ixy_log_close(void) {
    if (g_ixy_log_fp) {
        fclose(g_ixy_log_fp);
        g_ixy_log_fp = NULL;
    }
}