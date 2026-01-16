#include <pybind11/pybind11.h>
#include <driver/xmmult_accel.h>
#include <driver/xmmult_mixed.h>
namespace py = pybind11;

extern "C" {
    XMmult_accel *xmmult_accel_device_init(const char *pci_addr, float dsize_in, float dsize_out);
    int xmmult_accel_execute(XMmult_accel *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
        int N, int K, int M, int updateA, float dsize_in, float dsize_out, uint64_t device_offset);
    XMmult_mixed *xmmult_mixed_device_init(const char *pci_addr);
    int xmmult_mixed_execute(XMmult_mixed *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
        const int N, const int K, const int M, const int mode, const int updateA,
        const size_t A_size, const size_t B_size, const size_t C_size);
}

int say_hello() {
    printf("Hello from C++!\n");
    return 42; // Just a placeholder function
}
PYBIND11_MODULE(accel_ip, m) {
    py::class_<XMmult_accel>(m, "XMmult_accel");
    py::class_<XMmult_mixed>(m, "XMmult_mixed");

    m.def("say_hello", &say_hello, "Say hello");
    m.def("xmmult_accel_device_init", &xmmult_accel_device_init,
        "Initialize XMmult_accel device",
        py::return_value_policy::reference);
    m.def("xmmult_accel_execute", &xmmult_accel_execute, "Execute XMmult_accel operation");
    m.def("xmmult_mixed_device_init", &xmmult_mixed_device_init,
        "Initialize XMmult_mixed device",
        py::return_value_policy::reference);
    m.def("xmmult_mixed_execute", &xmmult_mixed_execute, "Execute XMmult_mixed operation");
}
