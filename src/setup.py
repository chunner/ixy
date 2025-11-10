from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'accel_ip',
        sources=['wrapper.cpp', 
                 'app/ixy-pktgen.c', 
                 'driver/cdma.c', 
                 'driver/device.c',
                 'driver/xmmult_accel.c',
                 'driver/xmmult_accel_tools.c',
                 'interrupts.c',
                 'memory.c',
                 'pci.c',
                 'libixy-vfio.c',
                 'stats.c'],
        include_dirs=['include', 
                      pybind11.get_include(), 
                      '.'],
        language='c++',
        define_macros=[('__linux__', '1')],
    )
]

setup(
    name='accel_ip',
    version='1.0',
    ext_modules=ext_modules,
)
