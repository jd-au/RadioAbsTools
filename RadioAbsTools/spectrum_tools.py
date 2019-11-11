import matplotlib.pyplot as plt
import numpy as np


def get_mean_continuum(spectrum, continuum_start_vel, continuum_end_vel):
    """
    Calculate the mean of the continuum values. This is based on precalculated regions where there is no gas expected.
    :param spectrum: The spectrum to be analysed, should be a numpy array of
        plane, velocity and flux values.
    :param continuum_start_vel: The lower bound of the continuum velocity range (in m/s)
    :param continuum_end_vel: The upper bound of the continuum velocity range (in m/s)
    :return: A pair of float which is the mean continuum flux and the standard deviation of the flux.
    """

    continuum_range = np.where(continuum_start_vel < spectrum.velocity)
    if len(continuum_range) ==0:
        return None, None, continuum_start_vel, continuum_end_vel

    bin_start = continuum_range[0][0]
    continuum_range = np.where(spectrum.velocity < continuum_end_vel)
    bin_end = continuum_range[0][-1]

    print("Using bins %d to %d (velocity range %d to %d) out of %d" % (
        bin_start, bin_end, continuum_start_vel, continuum_end_vel, len(spectrum.velocity)))
    continuum_sample = spectrum.flux[bin_start:bin_end]
    # print ("...gave sample of", continuum_sample)
    mean_cont = np.mean(continuum_sample)
    sd_cont = np.std(continuum_sample/mean_cont)
    return mean_cont, sd_cont


def plot_absorption_spectrum(velocity, optical_depth, filename, title, con_start_vel, con_end_vel, sigma_tau, range=None):
    """
    Output a plot of opacity vs LSR velocity to a specified file.

    :param velocity: The velocity data (in m/s)
    :param optical_depth: The opacity values for each velocity step
    :param filename: The file the plot should be written to. Should be
         an .eps or .pdf file.
    :param title: The title for the plot
    :param con_start_vel: The minimum velocity that the continuum was measured at.
    :param con_end_vel: The maximum velocity that the continuum was measured at.
    """
    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(velocity/1000, optical_depth, lw=1)
    if range:
        plt.xlim(range)

    if len(sigma_tau) > 0:
        tau_max = 1 + sigma_tau
        tau_min = 1 - sigma_tau
        plt.fill_between(velocity/1000, tau_min, tau_max, facecolor='lightgray', color='lightgray')

    plt.axhline(1, color='r')
    plt.axvline(con_start_vel/1000, color='g', linestyle='dashed')
    plt.axvline(con_end_vel/1000, color='g', linestyle='dashed')

    plt.xlabel(r'Velocity relative to LSR (km/s)')
    plt.ylabel(r'$e^{(-\tau)}$')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    #plt.show()
    plt.close()
    return


def calc_rating(opacity_range, max_s_max_n, continuum_sd):
    rating_codes = 'ABCDEF'
    rating = 0

    if opacity_range > 1.5:
        rating += 1
    if max_s_max_n < 3:
        rating += 1
    if continuum_sd*3 > 1:
        rating += 1

    return rating_codes[rating]


def rate_spectrum(opacity, continuum_sd):
    min_opacity = np.min(opacity)
    max_opacity = np.max(opacity)

    opacity_range = max_opacity - min_opacity
    max_s_max_n = (1 - min_opacity) / (max_opacity - 1)

    rating = calc_rating(opacity_range, max_s_max_n, continuum_sd)
    return rating, opacity_range, max_s_max_n
