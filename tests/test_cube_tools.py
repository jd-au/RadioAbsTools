# Tests for the spectrum_tools.py class.

# Author James Dempsey
# Date 16 Nov 2020

from __future__ import print_function, division

import math

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from RadioAbsTools import cube_tools

def _pixel_in_ellipse(ellipse_centre, x, y, a, b, pa, dec_offset=0*u.deg, expected=True):
    point  = SkyCoord((19-x)*u.arcsec, y*u.arcsec+dec_offset)

    result =  cube_tools.point_in_ellipse(ellipse_centre, point, a, b, pa)
    if expected is not None: # and expected != result:
        print ("Details for point x:{} y:{} at point {} arcsec".format(x, y, point.to_string(style='decimal', unit=u.arcsec)))
        cube_tools.point_in_ellipse(ellipse_centre, point, a, b, pa, verbose=True)
    return result

def _plot_grid(ellipse_centre, a, b, pa, dec_offset=0*u.deg):
    grid = np.zeros((20,20))
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if _pixel_in_ellipse(ellipse_centre, x, y, a, b, pa, dec_offset=dec_offset, expected=None):
                grid[y,x] = 1
    #grid[12,5] = 2
    print (grid)


def test_point_in_ellipse_vertical():
    # Define an ellipse centred at cell 10,10 in a 20x20 arcsec grid at low declination
    ellipse_centre = SkyCoord(9*u.arcsec, 10*u.arcsec)
    a = 6 # arcsec
    b = 3 # arcsec
    pa = 0*u.deg.to(u.rad) # rad

    _plot_grid(ellipse_centre, a, b, pa)

    assert _pixel_in_ellipse(ellipse_centre, 10, 10, a, b, pa), "Centre should be in ellipse"
    assert _pixel_in_ellipse(ellipse_centre, 10, 4, a, b, pa), "Top of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 10, 16, a, b, pa), "Bottom of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 9, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 11, 10, a, b, pa), "Right of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 7, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 13, 10, a, b, pa), "Right of ellipse should be inside"
    assert not _pixel_in_ellipse(ellipse_centre, 6, 10, a, b, pa, expected=False), "Off left of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 14, 10, a, b, pa, expected=False), "Off right of ellipse should be outside"


def test_point_in_ellipse_horizontal():
    # Define an ellipse centred at cell 10,10 in a 20x20 arcsec grid at low declination
    ellipse_centre = SkyCoord(9*u.arcsec, 10*u.arcsec)
    a = 6 # arcsec
    b = 3 # arcsec
    pa = 90*u.deg.to(u.rad) # rad

    _plot_grid(ellipse_centre, a, b, pa)

    assert _pixel_in_ellipse(ellipse_centre, 10, 10, a, b, pa), "Centre should be in ellipse"
    assert _pixel_in_ellipse(ellipse_centre, 10, 7, a, b, pa), "Top of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 10, 13, a, b, pa), "Bottom of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 5, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 15, 10, a, b, pa), "Right of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 4, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 16, 10, a, b, pa), "Right of ellipse should be inside"
    assert not _pixel_in_ellipse(ellipse_centre, 3, 10, a, b, pa, expected=False), "Off left of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 17, 10, a, b, pa, expected=False), "Off right of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 6, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 14, a, b, pa, expected=False), "Below ellipse should be outside"


def test_point_in_ellipse_diagonal():
    # Define an ellipse centred at cell 10,10 in a 20x20 arcsec grid at low declination
    ellipse_centre = SkyCoord(9*u.arcsec, 10*u.arcsec)
    a = 6 # arcsec
    b = 3 # arcsec
    pa = 45*u.deg.to(u.rad) # rad 

    _plot_grid(ellipse_centre, a, b, pa)

    assert _pixel_in_ellipse(ellipse_centre, 10, 10, a, b, pa), "Centre should be in ellipse"
    assert _pixel_in_ellipse(ellipse_centre, 6, 6, a, b, pa), "Top-left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 14, 14, a, b, pa), "Bottom-right of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 7, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 13, 10, a, b, pa), "Right of ellipse should be inside"
    assert not _pixel_in_ellipse(ellipse_centre, 3, 10, a, b, pa, expected=False), "Top-right mirror of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 17, 10, a, b, pa, expected=False), "Bottom-left mirror of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 6, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 14, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 9, 5, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 11, 15, a, b, pa, expected=False), "Below ellipse should be outside"


def test_point_in_ellipse_acute():
    # Define an ellipse centred at cell 10,10 in a 20x20 arcsec grid at low declination
    ellipse_centre = SkyCoord(9*u.arcsec, 10*u.arcsec)
    a = 6 # arcsec
    b = 3 # arcsec
    pa = 20*u.deg.to(u.rad) # rad 

    _plot_grid(ellipse_centre, a, b, pa)

    assert _pixel_in_ellipse(ellipse_centre, 10, 10, a, b, pa), "Centre should be in ellipse"
    assert _pixel_in_ellipse(ellipse_centre, 8, 5, a, b, pa), "Top-left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 13, 14, a, b, pa), "Bottom-right of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 7, 10, a, b, pa), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 13, 10, a, b, pa), "Right of ellipse should be inside"
    assert not _pixel_in_ellipse(ellipse_centre, 3, 10, a, b, pa, expected=False), "Top-right mirror of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 17, 10, a, b, pa, expected=False), "Bottom-left mirror of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 4, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 10, 16, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 9, 4, a, b, pa, expected=False), "Above ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 11, 16, a, b, pa, expected=False), "Below ellipse should be outside"


def test_point_in_ellipse_vertical_high_dec():
    # Define an ellipse centred at cell 10,10 in a 20x20 arcsec grid at low declination
    dec_offset = -70*u.deg
    ellipse_centre = SkyCoord(9*u.arcsec, 10*u.arcsec+dec_offset)
    a = 6 # arcsec
    b = 3 # arcsec
    pa = 0*u.deg.to(u.rad) # rad

    _plot_grid(ellipse_centre, a, b, pa, dec_offset=dec_offset)

    # Note our test grid doesn't include the distortion due to declination so we expect the ellipse to wider instead 
    assert _pixel_in_ellipse(ellipse_centre, 10, 10, a, b, pa, dec_offset=dec_offset), "Centre should be in ellipse"
    assert _pixel_in_ellipse(ellipse_centre, 10, 4, a, b, pa, dec_offset=dec_offset), "Top of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 10, 16, a, b, pa, dec_offset=dec_offset), "Bottom of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 2, 10, a, b, pa, dec_offset=dec_offset), "Left of ellipse should be inside"
    assert _pixel_in_ellipse(ellipse_centre, 18, 10, a, b, pa, dec_offset=dec_offset), "Right of ellipse should be inside"
    assert not _pixel_in_ellipse(ellipse_centre, 1, 10, a, b, pa, dec_offset=dec_offset, expected=False), "Off left of ellipse should be outside"
    assert not _pixel_in_ellipse(ellipse_centre, 19, 10, a, b, pa, dec_offset=dec_offset, expected=False), "Off right of ellipse should be outside"


def test_get_weighting_array():
    # array is order z, y, x
    data = np.zeros((40,10,10))
    # outer ring
    data[:,3:7,3:7] = 1
    data[:,4:6,4:6] = 2
    data_slice = data[0,:,:]
    print (data_slice)

    velocities = np.linspace(60, 99, num=40)*1000
    weights = cube_tools.get_weighting_array(data, velocities, 60*(u.km/u.s).to(u.m/u.s), 70*(u.km/u.s).to(u.m/u.s))
    print (weights)

    assert data_slice.shape == weights.shape
    assert round(np.sum(weights),3) == 1
    assert np.allclose(weights[data_slice == 0], 0)
    assert np.allclose(weights[data_slice == 2], 0.142857)
    assert np.allclose(weights[data_slice == 1], 0.035714)

    # Check that data outside the continuum range are ignored
    data[10:,:,:] = 1000
    weights = cube_tools.get_weighting_array(data, velocities, 60*(u.km/u.s).to(u.m/u.s), 70*(u.km/u.s).to(u.m/u.s))
    assert data_slice.shape == weights.shape
    assert round(np.sum(weights),3) == 1
    assert np.allclose(weights[data_slice == 0], 0)
    assert np.allclose(weights[data_slice == 2], 0.142857)
    assert np.allclose(weights[data_slice == 1], 0.035714)

    # Check that missing planes are ignored
    data[5,:,:] =  0
    weights = cube_tools.get_weighting_array(data, velocities, 60*(u.km/u.s).to(u.m/u.s), 70*(u.km/u.s).to(u.m/u.s))
    assert data_slice.shape == weights.shape
    assert round(np.sum(weights),3) == 1
    assert np.allclose(weights[data_slice == 0], 0)
    assert np.allclose(weights[data_slice == 2], 0.142857)
    assert np.allclose(weights[data_slice == 1], 0.035714)

    data[6,:,:] =  np.nan
    weights = cube_tools.get_weighting_array(data, velocities, 60*(u.km/u.s).to(u.m/u.s), 70*(u.km/u.s).to(u.m/u.s))
    assert data_slice.shape == weights.shape
    assert round(np.sum(weights),3) == 1
    assert np.allclose(weights[data_slice == 0], 0)
    assert np.allclose(weights[data_slice == 2], 0.142857)
    assert np.allclose(weights[data_slice == 1], 0.035714)

    # Check that all continuum planes are used
    data[7,3:7,3:7] =  3
    weights = cube_tools.get_weighting_array(data, velocities, 60*(u.km/u.s).to(u.m/u.s), 70*(u.km/u.s).to(u.m/u.s))
    assert data_slice.shape == weights.shape
    assert round(np.sum(weights),3) == 1
    assert np.allclose(weights[data_slice == 0], 0)
    assert np.allclose(weights[data_slice == 2], 0.117036)
    assert np.allclose(weights[data_slice == 1], 0.044321)
