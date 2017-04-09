from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name = 'obliquity_inference',
    packages = ['obliquity_inference'],
    ext_modules=[Extension('obliquity_inference._cosi_pdf',['./cython/_cosi_pdf.pyx'],include_dirs=[numpy.get_include()])],
    cmdclass = {'build_ext': build_ext}
)
