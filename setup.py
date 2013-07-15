from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("crfsuite", ["src/crfsuite.pyx",
                                          'src/iwa.c'],
                             libraries = ["crfsuite"],
                             library_dirs = ["/usr/local/lib"],
                             include_dirs = ["include", numpy.get_include()])]
)
