import cdma_mover
import numpy as np
cdma_mover.say_hello()

a = np.arange(0, 10, 1, dtype=np.int8)
b = np.zeros(10, dtype=np.int8)
print(f"Before move: a = {a}, b = {b}")
cdma_mover.data_mover("0000:00:04.0",a.ctypes.data, b.ctypes.data, 10 * 1) # 10 elements * 4 bytes each (int32)
print(f"After move: a = {a}, b = {b}")