# Tests for the spectrum_tools.py class.

# Author James Dempsey
# Date 16 Nov 2019

from __future__ import print_function, division

import matplotlib
matplotlib.use('agg')
import numpy as np

from RadioAbsTools import spectrum_tools


def test_calc_rating():
    assert spectrum_tools.calc_rating(1.0, 5, 0.05) == 'A'
    assert spectrum_tools.calc_rating(1.6, 5, 0.05) == 'B'
    assert spectrum_tools.calc_rating(1.0, 2.9, 0.05) == 'B'
    assert spectrum_tools.calc_rating(1.0, 5, 0.4) == 'B'
    assert spectrum_tools.calc_rating(1.6, 2.8, 0.05) == 'C'
    assert spectrum_tools.calc_rating(1.0, 2.9, 0.4) == 'C'
    assert spectrum_tools.calc_rating(1.8, 2.95, 0.4) == 'D'


def test_get_mean_continuum():
    velocity = np.arange(10)
    flux = np.ones((10))*5
    print (velocity, flux)
    mean, std = spectrum_tools.get_mean_continuum(velocity, flux, 0, 5)
    assert mean == 5
    assert std == 0

    # Check values above range are not included
    flux[5:] = 5000
    mean, std = spectrum_tools.get_mean_continuum(velocity, flux, 0, 5)
    assert mean == 5
    assert std == 0

    # Check values below range are not included
    flux[0] = 5000
    mean, std = spectrum_tools.get_mean_continuum(velocity, flux, 0, 5)
    assert mean == 5
    assert std == 0

    # Check values within range are included
    flux[1] = 45
    mean, std = spectrum_tools.get_mean_continuum(velocity, flux, 0, 5)
    assert mean == 15
    assert std == np.std(np.array([45,5,5,5])/15)
    mean, std = spectrum_tools.get_mean_continuum(velocity, flux, 1, 5)
    assert mean == 5
    assert std == 0

