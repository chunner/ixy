import numpy as np
import accel_ip

pci_addr = "0000:00:04.0"


A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int8)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]], dtype=np.int8)
              
print("Mat A：")
print(A)
print("\nMat B：")
print(B)

C1=np.dot(A.astype(np.int32),B.astype(np.int32))
print("\nMat C (by numpy):")
print(C1)



accel = accel_ip.xmmult_accel_device_init(pci_addr)
C2= np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)
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
