from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
import numpy

setup(
    name = 'obliquity_inference',
    version = "0.0.1",
    #package_dir = { 'obliquity_inference':  'obliquity_inference',
    #                 'obliquity_inference.plotting': 'obliquity_inference/plotting'},
    packages = ['obliquity_inference','obliquity_inference.plotting'],
    #packages = ['obliquity_inference'],
    #packages = find_packages(),
    ext_modules=[Extension('_cosi_pdf',['./cython/_cosi_pdf.pyx'],include_dirs=[numpy.get_include()])],
    cmdclass = {'build_ext': build_ext}
    package_data={'obliquity_inference':['data/*.txt','data/*.csv']},
    include_package_data=True

)
