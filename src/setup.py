from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'cdma_mover',
        sources=['wrapper.cpp', 
                 'app/ixy-pktgen.c', 
                 'driver/cdma.c', 
                 'driver/device.c',
                 'interrupts.c',
                 'memory.c',
                 'pci.c',
                 'libixy-vfio.c',
                 'stats.c'],
        include_dirs=['include', 
                      pybind11.get_include(), 
                      '.'],
        language='c++',
    )
]

setup(
    name='cdma_mover',
    version='1.0',
    ext_modules=ext_modules,
)
