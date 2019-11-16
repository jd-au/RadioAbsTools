# Tests for the spectrum_tools.py class.

# Author James Dempsey
# Date 16 Nov 2019

import matplotlib
matplotlib.use('agg')

from RadioAbsTools import spectrum_tools


def test_calc_rating():
    assert spectrum_tools.calc_rating(1.0, 5, 0.05) == 'A'
    assert spectrum_tools.calc_rating(1.6, 5, 0.05) == 'B'
    assert spectrum_tools.calc_rating(1.0, 2.9, 0.05) == 'B'
    assert spectrum_tools.calc_rating(1.0, 5, 0.4) == 'B'
    assert spectrum_tools.calc_rating(1.6, 2.8, 0.05) == 'C'
    assert spectrum_tools.calc_rating(1.0, 2.9, 0.4) == 'C'
    assert spectrum_tools.calc_rating(1.8, 2.95, 0.4) == 'D'

