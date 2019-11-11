import math
import os

from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits, votable
from astropy.wcs import WCS
import numpy as np
import numpy.core.records as rec


class IslandRange(object):
    def __init__(self, isle_id):
        self.isle_id = isle_id


def read_sources(filename):
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
        if sn > 10 and flux > 0.02:
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

    while fluxes[l_edge] == 0 and l_edge < len(fluxes):
        l_edge += 1

    while fluxes[r_edge] == 0 and r_edge > 0:
        r_edge -= 1

    return l_edge + num_edge_chan, r_edge - num_edge_chan


def get_weighting_array(data, velocities, continuum_start_vel, continuum_end_vel):
    """
    Calculate the mean of the continuum values. This is based on precalculated regions where there is no gas expected.
    :param data: A cubelet to be analysed, should be a 3D array of flux values.
    :param velocities: A numpy array of velocity values in m/s
    :param continuum_start_vel: The lower bound of the continuum velocity range (in m/s)
    :param continuum_end_vel: The upper bound of the continuum velocity range (in m/s)
    :return: A 2D array of weighting values for the
    """

    continuum_range = np.where(continuum_start_vel < velocities)
    if len(continuum_range) == 0:
        return np.zeros(data.shape[1:2])

    bin_start = continuum_range[0][0]
    continuum_range = np.where(velocities < continuum_end_vel)
    bin_end = continuum_range[0][-1]

    # print("Using bins %d to %d (velocity range %d to %d) out of %d" % (
    #    bin_start, bin_end, continuum_start_vel, continuum_end_vel, len(velocities)))
    # print(data.shape)
    continuum_sample = data[bin_start:bin_end, :, :]
    # print ("...gave sample of", continuum_sample)
    mean_cont = np.mean(continuum_sample, axis=0)
    mean_sq = mean_cont ** 2
    sum_sq = np.sum(mean_sq)
    weighting = mean_sq / sum_sq
    # print("Got weighting of {} from {} and {}".format(weighting, mean_sq, sum_sq))
    return weighting


def point_in_ellipse(origin, point, a, b, pa_rad):
    # Convert point to be in plane of the ellipse
    p_ra_dist = point.icrs.ra.degree - origin.icrs.ra.degree
    p_dec_dist = point.icrs.dec.degree - origin.icrs.dec.degree
    x = p_ra_dist * math.cos(pa_rad) + p_dec_dist * math.sin(pa_rad)
    y = - p_ra_dist * math.sin(pa_rad) + p_dec_dist * math.cos(pa_rad)

    a_deg = a / 3600
    b_deg = a / 3600

    # Calc distance from origin relative to a/b
    dist = math.sqrt((x / a_deg) ** 2 + (y / b_deg) ** 2)
    # print("Point %s is %f from ellipse %f, %f, %f at %s." % (point, dist, a, b, math.degrees(pa_rad), origin))
    return dist <= 1.0


def get_integrated_spectrum(image, w, src, velocities, continuum_start_vel, continuum_end_vel):
    """
    Calculate the integrated spectrum of the component.
    :param image: The image's data array
    :param w: The image's world coordinate system definition
    :param src: The details of the component being processed
    :param velocities: A numpy array of velocity values in m/s
    :param continuum_start_vel: The lower bound of the continuum velocity range (in m/s)
    :param continuum_end_vel: The upper bound of the continuum velocity range (in m/s)
    :return: An array of average flux/pixel across the component at each velocity step
    """
    pix = w.wcs_world2pix(src['ra'], src['dec'], 0, 0, 1)
    x_coord = int(np.round(pix[0])) - 1  # 266
    y_coord = int(np.round(pix[1])) - 1  # 197
    #print("Translated %.4f, %.4f to %d, %d" % (
    #    src['ra'], src['dec'], x_coord, y_coord))
    radius = 2
    y_min = y_coord - radius
    y_max = y_coord + radius
    x_min = x_coord - radius
    x_max = x_coord + radius
    data = np.copy(image[0, :, y_min:y_max+1, x_min:x_max+1])

    origin = SkyCoord(src['ra'], src['dec'], frame='icrs', unit="deg")
    pa_rad = math.radians(src['pa'])
    total_pixels = (y_max-y_min +1) * (x_max-x_min +1)
    outside_pixels = 0
    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            eq_pos = w.wcs_pix2world(i+1, j+1, 0, 0, 1)
            point = SkyCoord(eq_pos[0], eq_pos[1], frame='icrs', unit="deg")
            if not point_in_ellipse(origin, point, src['a'], src['b'], pa_rad):
                data[:, i-x_min, j-y_min] = 0
                outside_pixels += 1
    #print("Found {} pixels out of {} inside the component {} at {} {}".format(total_pixels - outside_pixels, total_pixels,
    #                                                                   src['id'],
    #                                                                   point.galactic.l.degree,
    #                                                                   point.galactic.b.degree))
    weighting = get_weighting_array(data, velocities, continuum_start_vel, continuum_end_vel)
    integrated = np.sum(data * weighting, axis=(1, 2))
    inside_pixels = total_pixels - outside_pixels
    if inside_pixels <= 0:
        print ("Error: No data for component!")
    else:
        integrated /= inside_pixels

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
