#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
    int data_mover(const char *pci_addr, uint64_t src_addr, uint64_t dst_addr, uint32_t size);
}

int say_hello() {
    printf("Hello from C++!\n");
    return 42; // Just a placeholder function
}
PYBIND11_MODULE(cdma_mover, m) {
    m.def("say_hello", &say_hello, "Say hello");
    m.def("data_mover", &data_mover, "Move data using CDMA");
}
