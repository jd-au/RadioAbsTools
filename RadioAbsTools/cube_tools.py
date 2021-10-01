import math
import os

from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits, votable
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import numpy.core.records as rec

_allowed_weights = ['square', 'linear', 'none']


class IslandRange(object):
    def __init__(self, isle_id):
        self.isle_id = isle_id


def read_sources(filename, min_sn=10, min_flux=0.02):
    print ("Extracting sources from " + filename)
    sources = []

    if not os.path.exists(filename):
        print ("Warning: File %s does not exist, skipping source read." % \
               filename)
        return sources

    src_votable = votable.parse(filename, pedantic=False)
    results = src_votable.get_first_table().array
    for row in results:
        id = str(row['island']) + "-" + str(row['source'])
        ra = row['ra']
        dec = row['dec']
        rms = row['local_rms']
        flux = row['peak_flux']
        sn = flux / rms
        print ("Found source %s at %.4f, %.4f with flux %.4f and rms of %.4f "
               "giving S/N of %.4f" % (id, ra, dec, flux, rms, sn))
        if sn > min_sn and flux > min_flux:
            src = dict(zip(results.dtype.names,row))
            src['id'] = id
            src['sn'] = sn
            #sources.append([ra, dec, id, flux, row['island']])
            sources.append(src)
        else:
            print ("Ignoring source at %.4f, %.4f due to low S/N of %.4f or "
                   "flux of %.4f" % (ra, dec, sn, flux))

    return sources


def read_islands(filename):
    print ("Extracting islands from " + filename)
    islands = {}

    if not os.path.exists(filename):
        print ("Warning: File %s does not exist, skipping island read." % \
               filename)
        return {}

    isle_votable = votable.parse(filename, pedantic=False)
    results = isle_votable.get_first_table().array
    for row in results:
        islands[row['island']] = row
    return islands


def calc_island_ranges(islands, pixel_size):
    island_ranges = []
    for island in islands.values():
        ir = IslandRange(island['island'])
        ra = island['ra']
        dec = island['dec']
        ra_width = abs(island['x_width'] * pixel_size[0])
        dec_width = abs(island['y_width'] * pixel_size[1])
        ir.min_ra = ra - (ra_width/2)
        ir.max_ra = ra + (ra_width/2)
        ir.min_dec = dec - (dec_width/2)
        ir.max_dec = dec + (dec_width/2)
        #print("Island %d goes from %f to %f (%d*%f)/ %f to %f (%d*%f)" % (
        #    island['island'], ir.min_ra, ir.max_ra, island['x_width'], pixel_size[0], ir.min_dec, ir.max_dec,
        #    island['y_width'], pixel_size[1]))
        island_ranges.append(ir)
    return island_ranges


def find_edges(fluxes, num_edge_chan):
    """
    Seek from the edges to find where the data starts for this set of fluxes.
    This accounts for an optional number of channels in the data which have no
    data recorded.
    :param fluxes: The array of fluxes to be checked.
    :param num_edge_chan: The number of edge channels with data to be skipped
    :return: The index of the first and last cell to have data.
    """

    l_edge = 0
    r_edge = len(fluxes)-1

    while l_edge < len(fluxes)-1 and fluxes[l_edge] == 0:
        l_edge += 1

    while r_edge > 0 and fluxes[r_edge] == 0:
        r_edge -= 1

    return l_edge + num_edge_chan, r_edge - num_edge_chan


def get_weighting_array(data, velocities, continuum_start_vel, continuum_end_vel, weighting='square'):
    """
    Calculate the mean of the continuum values. This is based on precalculated regions where there is no gas expected.
    :param data: A cubelet to be analysed, should be a 3D array of flux values.
    :param velocities: A numpy array of velocity values in m/s
    :param continuum_start_vel: The lower bound of the continuum velocity range (in m/s)
    :param continuum_end_vel: The upper bound of the continuum velocity range (in m/s)
    :param weighting: The weighting scheme to use, one of square, linear, none
    :return: A 2D array of weighting values for the
    """

    if (continuum_start_vel > np.max(velocities)) or (continuum_end_vel < np.min(velocities)):
        raise Exception("Continuum range {} to {} is outside of the data velocity range {} to {}".format(
            continuum_start_vel, continuum_end_vel, np.min(velocities), np.max(velocities)))

    continuum_range = np.where(continuum_start_vel < velocities)
    if len(continuum_range) == 0:
        return np.zeros(data.shape[1:2])

    if weighting not in _allowed_weights:
        raise Exception("Weighting must by one of ", ', '.join(_allowed_weights))

    bin_start = continuum_range[0][0]
    continuum_range = np.where(velocities < continuum_end_vel)
    bin_end = continuum_range[0][-1]

    # print("Using bins %d to %d (velocity range %d to %d) out of %d" % (
    #    bin_start, bin_end, continuum_start_vel, continuum_end_vel, len(velocities)))
    # print(data.shape)
    continuum_sample = np.array(data[bin_start:bin_end, :, :])
    continuum_sample[continuum_sample<0]=0

    # print ("...gave sample of", continuum_sample)
    mean_cont = np.nanmean(continuum_sample, axis=0)
    mean_cont[mean_cont<0]=0
    if weighting == 'square':
        mean_sq = mean_cont ** 2
        sum_sq = np.nansum(mean_sq)
        weights = mean_sq / sum_sq
    elif weighting == 'linear':
        weights = mean_cont / np.nansum(mean_cont)
    else:
        # No weights, just trim to ellipse
        weights = mean_cont
        weights[weights>0]=1
        weights = weights/np.sum(weights)
    # print("Got weighting of {} from {} and {}".format(weighting, mean_sq, sum_sq))
    return weights


def point_in_ellipse(origin, point, a, b, pa_rad, verbose=False):
    """
    Identify if the point is inside the ellipse.

    :param origin A SkyCoord defining the centre of the ellipse.
    :param point A SkyCoord defining the point to be checked.
    :param a The semi-major axis in arcsec of the ellipse
    :param b The semi-minor axis in arcsec of the ellipse
    :param pa_rad The position angle of the ellipse. This is the angle of the major axis measured in radians East of 
                   North (or CCW from the y axis).
    """
    # Convert point to be in plane of the ellipse, accounting for distortions at high declinations
    p_ra_dist = (point.icrs.ra.degree - origin.icrs.ra.degree)* math.cos(origin.icrs.dec.rad)
    p_dec_dist = point.icrs.dec.degree - origin.icrs.dec.degree

    # Calculate the angle and radius of the test opoint relative to the centre of the ellipse
    # Note that we reverse the ra direction to reflect the CCW direction
    radius = math.sqrt(p_ra_dist**2 + p_dec_dist**2)
    diff_angle = (math.pi/2 + pa_rad) if p_dec_dist == 0 else math.atan(p_ra_dist / p_dec_dist) - pa_rad

    # Obtain the point position in terms of the ellipse major and minor axes
    minor = radius * math.sin(diff_angle)
    major = radius * math.cos(diff_angle)
    if verbose:
        print ('point relative to ellipse centre angle:{} deg radius:{:.4f}" maj:{:.2f}" min:{:.2f}"'.format(math.degrees(diff_angle), radius*3600, 
                major*3600, minor*3600))
    
    a_deg = a / 3600.0
    b_deg = b / 3600.0

    # Calc distance from origin relative to a and b
    dist = math.sqrt((major / a_deg) ** 2 + (minor / b_deg) ** 2)
    if verbose:
        print("Point %s is %f from ellipse %f, %f, %f at %s." % (point, dist, a, b, math.degrees(pa_rad), origin))
    return round(dist,3) <= 1.0


def get_integrated_spectrum(image, w, src, velocities, continuum_start_vel, continuum_end_vel, radius=None, 
                            plot_weight_path=None, weighting='square'):
    """
    Calculate the integrated spectrum of the component.
    :param image: The image's data array
    :param w: The image's world coordinate system definition
    :param src: The details of the component being processed, must have ra, dec, a, b, pa and comp_name keys
    :param velocities: A numpy array of velocity values in m/s
    :param continuum_start_vel: The lower bound of the continuum velocity range (in m/s)
    :param continuum_end_vel: The upper bound of the continuum velocity range (in m/s)
    :param radius: The radius of the box around the source centre where data will be checked for membership of the 
        source ellipse. Default is to use the semi-major axis of the source.
    :param plot_weight_path: The path to which diagnostic plots are output. Default is not to output plots.
    :param weighting: The weighting scheme to use, one of square, linear, none
    :return: An array of average flux/pixel across the component at each velocity step
    """

    if weighting not in _allowed_weights:
        raise Exception("Weighting must by one of ", ', '.join(_allowed_weights))

    if plot_weight_path:
        print ("getting spectrum for source " + str(src))
    has_stokes = len(image.shape) > 3
    pix = w.wcs_world2pix(src['ra'], src['dec'], 0, 0, 1) if has_stokes else w.wcs_world2pix(src['ra'], src['dec'], 0, 1)
    x_coord = int(np.round(pix[0])) - 1  # 266
    y_coord = int(np.round(pix[1])) - 1  # 197
    if not radius:
        radius = math.ceil(src['a'])
    #print("Translated %.4f, %.4f to %d, %d" % (
    #    src['ra'], src['dec'], x_coord, y_coord))
    #print (w)
    y_min = y_coord - radius
    y_max = y_coord + radius
    x_min = x_coord - radius
    x_max = x_coord + radius
    data = np.copy(image[0, :, y_min:y_max+1, x_min:x_max+1]) if has_stokes else np.copy(image[:, y_min:y_max+1, x_min:x_max+1])
    if plot_weight_path:
        # non wcs plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        ax.imshow(np.nansum(data, axis=0), origin='lower')
        plt.title(src['comp_name'])
        fname = plot_weight_path + '/'+ src['comp_name'] + '_data.png'
        print ('Plotting data to ' + fname) 
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

        # wcs plot
        plt.subplot(projection=w.celestial)
        if has_stokes:
            plt.imshow(image[0,10,:,:], origin='lower')
        else:
            plt.imshow(image[10,:,:], origin='lower')
        plt.grid(color='white', ls='solid')
        fname = plot_weight_path + '/'+ src['comp_name'] + '_image_wcs.png'
        print ('Plotting data to ' + fname) 
        plt.savefig(fname, bbox_inches='tight')
        plt.close()


    origin = SkyCoord(src['ra'], src['dec'], frame='icrs', unit="deg")
    pa_rad = math.radians(src['pa'])
    total_pixels = (y_max-y_min +1) * (x_max-x_min +1)
    outside_pixels = 0
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            eq_pos = w.wcs_pix2world(x, y, 0, 0, 0) if has_stokes else w.wcs_pix2world(x, y, 0, 0)
            point = SkyCoord(eq_pos[0], eq_pos[1], frame='icrs', unit="deg")
            in_ellipse = point_in_ellipse(origin, point, src['a'], src['b'], pa_rad)
            if not in_ellipse:
                data[:, y-y_min, x-x_min] = 0
                outside_pixels += 1
            #print (point.ra, point.dec, x, y, in_ellipse)

    # print("Found {} pixels out of {} inside the component {} at {} {}".format(total_pixels - outside_pixels, total_pixels,
    #                                                                   src['comp_name'],
    #                                                                   point.galactic.l.degree,
    #                                                                   point.galactic.b.degree))
    weights = get_weighting_array(data, velocities, continuum_start_vel, continuum_end_vel, weighting=weighting)
    integrated = np.nansum(data * weights, axis=(1, 2))
    inside_pixels = total_pixels - outside_pixels
    if inside_pixels <= 0:
        print ("Error: No data for component!")
    else:
        integrated /= inside_pixels

    if plot_weight_path:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        pos = ax.imshow(weights, origin='lower')
        fig.colorbar(pos, ax=ax)
        plt.title(src['comp_name'])
        fname = plot_weight_path + '/'+ src['comp_name'] + '_weights.png'
        print ('Plotting weights to ' + fname) 
        print ('Ellipse ra={} dec={} pa={:.03f} deg {:.03f}pi rad'.format(src['ra'], src['dec'], src['pa'], pa_rad/math.pi))
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    return integrated


def extract_spectra(fits_filename, src_filename, isle_filename, continuum_start_vel, continuum_end_vel, num_edge_chan = 10):
    #num_edge_chan = 10
    #fits_filename = "{0}/1420/magmo-{1}_1420_sl_restor.fits".format(daydirname,
    #                                                                field)
    #src_filename = "{0}/{1}_src_comp.vot".format(daydirname, field)
    #isle_filename = "{0}/{1}_src_isle.vot".format(daydirname, field)

    spectra = dict()
    source_ids = dict()
    if not os.path.exists(fits_filename):
        print ("Warning: File %s does not exist, skipping extraction." % \
              fits_filename)
        return spectra, source_ids, []

    sources = read_sources(src_filename)
    islands = read_islands(isle_filename)
    hdulist = fits.open(fits_filename)
    image = hdulist[0].data
    header = hdulist[0].header
    w = WCS(header)
    index = np.arange(header['NAXIS3'])
    beam_maj = header['BMAJ'] * 60 * 60
    beam_min = header['BMIN'] * 60 * 60
    beam_area = math.radians(header['BMAJ']) * math.radians(header['BMIN'])
    # print ("Beam was %f x %f arcsec giving area of %f radians^2." % (beam_maj, beam_min, beam_area))
    ranges = calc_island_ranges(islands, (header['CDELT1'], header['CDELT2']))
    velocities = w.wcs_pix2world(10,10,index[:],0,0)[2]
    for src in sources:
        c = SkyCoord(src['ra'], src['dec'], frame='icrs', unit="deg")

        img_slice = get_integrated_spectrum(image, w, src, velocities, continuum_start_vel, continuum_end_vel)

        l_edge, r_edge = find_edges(img_slice, num_edge_chan)
        # print("Using data range %d - %d out of %d channels." % (
        #    l_edge, r_edge, len(img_slice)))

        # plotSpectrum(np.arange(slice.size), slice)
        spectrum_array = rec.fromarrays(
            [np.arange(img_slice.size)[l_edge:r_edge],
             velocities[l_edge:r_edge],
             img_slice[l_edge:r_edge]],
            names='plane,velocity,flux')
        spectra[c.galactic.l] = spectrum_array

        # isle = islands.get(src['island'], None)
        src_map = {'id': src['id'], 'flux': src['peak_flux'], 'pos': c, 'beam_area': beam_area}
        src_map['a'] = src['a']
        src_map['b'] = src['b']
        src_map['pa'] = src['pa']
        # print (src_map)
        source_ids[c.galactic.l] = src_map
    del image
    del header
    hdulist.close()

    return spectra, source_ids, ranges
