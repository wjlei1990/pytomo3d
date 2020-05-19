from __future__ import print_function
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as test_command


class PyTest(test_command):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="pytomo3d",
    version="0.2.0",
    license='GNU Lesser General Public License, version 3 (LGPLv3)',
    description="Python toolkits for seismic tomograpy",
    author="Wenjie Lei",
    author_email="lei@princeton.edu",
    url="https://github.com/wjlei1990/pytomo3d",
    packages=find_packages(),
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology", "tomography", "adjoint", "signal", "inversion", "window"
    ],
    install_requires=[
        "numpy", "obspy==1.0.3", "flake8>=3.0", "pytest", "nose",
        "future>=0.14.1", "pyflex", "pyadjoint", "geographiclib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
