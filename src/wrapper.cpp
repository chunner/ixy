#include <pybind11/pybind11.h>
#include <driver/xmmult_accel.h>
namespace py = pybind11;

extern "C" {
    int execute(const char *pci_addr, uint64_t src_addr, uint64_t dst_addr, uint32_t size);
    XMmult_accel *xmmult_accel_device_init(const char *pci_addr);
    int xmmult_accel_execute(XMmult_accel *InstancePtr, const uintptr_t A, const uintptr_t B, uintptr_t C,
        int N, int K, int M, int updateA);
}

int say_hello() {
    printf("Hello from C++!\n");
    return 42; // Just a placeholder function
}
PYBIND11_MODULE(accel_ip, m) {
    py::class_<XMmult_accel>(m, "XMmult_accel");

    m.def("say_hello", &say_hello, "Say hello");
    m.def("execute", &execute, "execute operator");
    m.def("xmmult_accel_device_init", &xmmult_accel_device_init,
        "Initialize XMmult_accel device",
        py::return_value_policy::reference);
    m.def("xmmult_accel_execute", &xmmult_accel_execute, "Execute XMmult_accel operation");
}
