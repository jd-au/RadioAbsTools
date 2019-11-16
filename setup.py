#! /usr/bin/env python
"""
Setup for RadioAbsTools
"""
import os
import sys
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    """Read a file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
    """Get the version number of RadioAbsTools"""
    import RadioAbsTools
    return RadioAbsTools.__version__


reqs = ['numpy>=1.12',
        'scipy>=0.16',
        'astropy>=2.0',
        'matplotlib>=2.2.3']

setup(
    name="RadioAbsTools",
    version=get_version(),
    author="James Dempsey",
    author_email="James.Dempsey@anu.edu.au",
    description="Tools for analysing radio astronomy absorption spectra",
    #url="https://github.com/jd-au/",
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    packages=['RadioAbsTools'],
    install_requires=reqs,
    #scripts=[],
    #data_files=[('RadioAbsTools', [os.path.join(data_dir, 'MOC.fits')]) ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'nose']
)