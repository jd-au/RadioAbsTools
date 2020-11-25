.. image:: https://github.com/jd-au/RadioAbsTools/workflows/Python%20package/badge.svg
   :target: https://github.com/jd-au/RadioAbsTools/actions
   :alt: Python package status
   
============================================
Analysing Radio Astronomy Absorption Spectra
============================================

RadioAbsTools is a suite of tools for working with radio astronomy absorption spectra.
It is written with neutral hydrogen (HI) spectra in mind but should be adaptable to other lines.

Installation and Requirements
-----------------------------

RadioAbsTools works with Python 3.6 or later and requires Astropy v3 or later.

For local use it can be installed by downloading the source and then:

.. code:: bash

    cd radioabstools
    pip install -e .

RadioAbsTools requires numpy, scipy and astropy.

Usage
-----

cube_tools.py
.............

`cube_tools` contains methods for extracting spectra from FITS cubes.


spectrum_tools.py
.................

`spectrum_tools` contains methods for dealing with absorption spectra. 

*Example Usage:*

.. code:: python

    from RadioAbsTools import spectrum_tools
    import numpy as np

    velocities = np.arange(10)*1000
    fluxes = np.ones((10))*5
    fluxes[7:8]=2
    fluxes[2]=5.1
    spec_mean, spec_sd = spectrum_tools.get_mean_continuum(velocities, fluxes, 1000, 5000)
    sigma_opacity = np.ones(velocities.shape)* spec_sd
    opacity = fluxes/spec_mean
    spectrum_tools.plot_absorption_spectrum(velocities, opacity, 'spectrum.png', 
        'Sample Spectrum', 1000, 5000, sigma_opacity)
    rating, opacity_range, max_s_max_n = spectrum_tools.rate_spectrum(opacity, spec_sd)

Testing
-------

This module uses pytest for unit testing. To run the tests, use

.. code:: bash

    pytest





