import numpy as np
import accel_ip
import os

pci_addr = os.environ.get('PCI_ADDR')

N, K, M = 64, 768, 768 
A = np.random.randint(-10, 10, size=(N, K)).astype(np.int8, order='C')
B = np.random.randint(-10, 10, size=(K, M)).astype(np.int8, order='C')

print("Mat A：")
print(A)
print("\nMat B：")
print(B)

C1=np.dot(A.astype(np.int32),B.astype(np.int32))
print("\nMat C (by numpy):")
print(C1)



accel = accel_ip.xmmult_accel_device_init(pci_addr)
C2= np.zeros((A.shape[0], B.shape[1]), dtype=np.int32, order='C')
accel_ip.xmmult_accel_execute(
    accel, 
    A.ctypes.data, 
    B.ctypes.data, 
    C2.ctypes.data,
    A.shape[0], 
    A.shape[1], 
    B.shape[1], 
    1)
print("\nMat C (by accel_ip):")
print(C2)

# import os
# import struct

# def virt_to_phys(addr):
#     page_size = os.sysconf("SC_PAGE_SIZE")
#     pagemap_entry_size = 8
#     vpn = addr // page_size

#     with open("/proc/self/pagemap", "rb") as f:
#         f.seek(vpn * pagemap_entry_size)
#         entry = f.read(pagemap_entry_size)
#         val = struct.unpack("Q", entry)[0]

#         if (val >> 63) == 0:
#             raise RuntimeError("page not present")

#         pfn = val & ((1 << 55) - 1)
#         return (pfn * page_size) + (addr % page_size)


# virt_addr = A.ctypes.data
# phys_addr = virt_to_phys(virt_addr)

# print("virtual:", hex(virt_addr))
# print("physical:", hex(phys_addr))
