"""
This module contains functions to perform aperture
photometry for staring mode observations, including tools
for IRAF-style photometry (aperture photometry with
non-native background/error methods).

Authors
-------
    Varun Bajaj, December 2017
    Clare Shanahan, December 2019
    Mariarosa Marinelli, April 2023

Notes
-----
Refactored in spring 2023 to accommodate updated
version of `photutils` and merge functionality from the
`wfc3-photometry` repository. Major changes:
     1. `SourceCatalog` is now employed instead of the
        deprecated `source_properties` function.
     2. A pixel-wise threshold image is created using
        `photutils` function `detect_threshold` and a
        sigma level instead of a float threshold to use
        for the entire image.
     3. Sources are detected using said custom threshold
        image and an integer number of connected pixels
        instead of a 3x3 Gaussian 2D kernel.
     4. Added helper function to enable multiple attempts
        at fitting Gaussian distribution to data with
        different parameters.
     5. Added functionality for calculating IRAF-style
        photometry. Note that at present, the background
        computations for `iraf_style_photometry()` will
        fully include a pixel that has ANY overlap with the
        background aperture (the annulus). This is to
        simplify the computation of the median, as a
        weighted median is nontrivial, and slower. See
        additional notes below.

IRAF-style photometry
---------------------
The function, `iraf_style_photometry()`, serves to ease the
computation of photometric magnitudes and errors using
`photutils` by replicating DAOPHOT's photometry and error
methods.

The formula for DAOPHOT's error is:

    err = sqrt (Poisson_noise / epadu
                + area * stdev**2
                + area**2 * stdev**2 / nsky)

This in turn gives a magnitude error:

    mag_err = 1.0857 * err / flux

The variables are as follows.
    - epadu : electrons per ADU (gain)
    - area : the photometric aperture area
    - stdev is the uncertainty in the sky measurement
    - nsky is the sky annulus area.

To get the uncertainty in the sky we implement a custom
background tool, `compute_phot_error()`, which also enables
computation of the mean and median of the sky as well (more
robust statistics). All the stats are sigma-clipped, and
are calculated by `make_aperture_stats_tbl()`.

Functions
---------
_fit_gaussian(data, bins, hist_ranges)
    Attempts to fit data with different ranges.
calc_sky_annulus(data, x, y, r_in, r_out, sky_method, sigma_clip)
    Calculate background in a circular annulus.
circular_aperture_photometry(data, xc, yc, aperture_radii)
    Perform photometry with circular aperture(s).
compute_iraf_style_error(flux_var, bg_phot, bg_method, ap_area, epadu)
    Compute flux error using DAOPHOT-style computation.
detect_sources_segmap(data, nsigma, npixels, show)
    Runs image segmentation to detect sources in `data`.
iraf_style_photometry(phot_aps, bg_aps, data, error_array, bg_method, epadu)
    Computes photometry with `photutils` apertures and IRAF formulae.
show_source_detection_plot(data, coo_tab)
    Plot detected sources, if any.
"""


import matplotlib.pyplot as plt
import numpy as np
import warnings

from matplotlib.colors import LogNorm
from astropy.table import Table
try:
    from ginga.util import zscale
except:
    from ginga.AutoCuts import ZScale
from photutils.segmentation import (detect_sources,
                                    detect_threshold,
                                    SourceCatalog)
from photutils.aperture import (aperture_photometry,
                                CircularAnnulus,
                                CircularAperture)
#from background import make_aperture_stats_tbl, calc_1d_gauss_background
from wfc3_phot_tools.staring_mode.background import make_aperture_stats_tbl, calc_1d_gauss_background

def _fit_gaussian(data, bins, hist_ranges):
    """Attempts to fit data with different ranges.

    Helper function takes a list of tuples, representing
    different value ranges with which to attempt to fit a
    Gaussian, and tries to fit a Gaussian. If the data is
    successfully fit, the function returns the A, mu, and
    sigma returned by `calc_1d_gauss_background()` and
    breaks (so that you don't continue to attempt to fit
    the data once a solution has been found). If the data
    is unable to be fit, this function will return three
    values of -999. This was written to eliminate a
    series of nested try/except loops.

    Parameters
    ----------
    data : array-like
    bins : int
    hist_ranges : list of tuples

    Returns
    -------
    A : float or int
    mu : float or int
    sigma : float or int
    """
    for hist_range in hist_ranges:
        A, mu, sigma = calc_1d_gauss_background(data, bins=bins,
                                                hist_range=hist_range)
        if A != None:
            return A, mu, sigma
            break

    if x == None:
        return -999, -999, -999


def calc_sky_annulus(data, x, y, r_in, r_out, sky_method='median',
                     sigma_clip=True):
    """Calculate background in a circular annulus.

    Calculates the background level in a circular annulus
    centered at `x, y` with an inner radius of `r_in`
    pixels and an outer radius of `r_out` pixels.

    Sky level and standard deviation in annulus are
    returned. Options for `sky_method` are `mean`,
    `median`, or `mode`. Uses a sigma clip with sigma =
    `n_sigma_clip` to compute stats. If n_sigma_clip is set
    to 0, sigma clipping won't be done.

    Notes
    -----
    Refactored Gaussian fit section to utilize new helper
    function `_fit_gaussian()`, which eliminates the need
    for several nested try/except loops.

    To-Do:
    ------
        - Move to `background.py`

    Parameters
    ----------
    data : array
        Science data array.
    x, y : float
        Pixel position for annulus center.
    r_in, r_out : int
        Inner, outer circular annulus radii
    sky_method : str
        'mean', 'median', 'mode', or 'gaussian'
    n_sigma_clip : int
        Used for sigma clipping. 0 to turn off sigma
        clipping.

    Returns
    -------
    (back, backstd) : tuple
        Sky level in annulus, std of sky.

    """
    sky_apertures = CircularAnnulus((x, y), r_in, r_out)

    if sky_method in ['mean', 'median', 'mode']:
        sky_tbl = make_aperture_stats_tbl(data, sky_apertures, sigma_clip=sigma_clip)
        return(sky_tbl['aperture_'+sky_method].item(),
               sky_tbl['aperture_std'].item())

    elif sky_method == 'gaussian':
        print('fitting gaussian to data for background.')

        hist_ranges = [(-10, 10), (-20, 10), (-10, 20), (-5, 5)]
        A, mu, sigma = _fit_gaussian(data, bins, hist_ranges)

        if A == -999:
            print("Couldn't fit Gaussian.")

        return (mu, sigma)


def circular_aperture_photometry(data, xc, yc, aperture_radii):
    """Perform photometry with circular aperture(s).

    Aperture photometry with circular apertures on single
    source in `data` located at (xc, yc).

    Returns a table of aperture sums for every aperture
    size in `aperture_radii`.

    Parameters
    ----------
    data : array-like
        Image array.
    xc, yc : floats
        x, y location of source in `data`.
    aperture_radii : float or list of floats
        Desired aperture radii, in pixels.

    Returns
    -------
    phot_table : `astropy.table.Table`
        Table with columns for x&y source position, and sum
        in every circular aperture in `aperture_radii`.
    """
    if type(aperture_radii) == float:
        aperture_radii = [aperture_radii]

    aps = [CircularAperture((xc, yc), r=rad) for rad in aperture_radii]
    phot_table = Table(aperture_photometry(data, aps))
    phot_table.remove_column('id')

    table_order = []
    for i, rad in enumerate(aperture_radii):
        rad = str(rad)
        phot_table.rename_column('aperture_sum_{}'.format(str(i)),
                                 'countrate_{}'.format(rad))
        table_order.append('countrate_{}'.format(rad))

    table_order = ['xcenter', 'ycenter'] + table_order

    phot_table = phot_table[table_order]

    return phot_table


def compute_iraf_style_error(flux_var, bg_phot, bg_method, ap_area, epadu=1.0):
    """Compute flux error using DAOPHOT-style computation.

    Parameters
    ----------
    flux_var :
    bg_phot :
    bg_method :
    ap_area :
    epadu : float

    Returns
    -------
    flux_error : float
    """
    bg_variance_terms = (ap_area * bg_phot['aperture_std'] ** 2. ) \
                        * (1. + ap_area/bg_phot['aperture_area'])
    variance = flux_var / epadu + bg_variance_terms
    flux_error = variance ** .5

    return flux_error


def detect_sources_segmap(data, nsigma, npixels, show=False):
    """Runs image segmentation to detect sources in `data`.

    This function utilizes two `photutils` functions to
    detect sources in staring mode data:
        `detect_threshold(data, nsigma)`
            This function uses simple sigma-clipped
            statistics to compute a scalar BG and noise
            estimate, which is used as a detection
            threshold for `detect_sources()`.
        `detect_sources(data, threshold, npixels)`
            This function detects sources above a threshold
            of specific pixel-wise data values, which are
            connected by a specified number of pixels.

    Parameters
    ----------
    data : array-like
        Array of image data.
    nsigma : int or float
        The number of standard deviations per pixel above
        the background for which to consider a pixel as
        possibly being part of a source.
    npixels : int
        The minimum number of connected pixels, each
        greater than the corresponding pixel-wise value in
        `threshold`, that an object must have to be
        detected. Must be a positive integer.
    show : bool
        Show a plot of detected source(s). Default is
        `False`.

    Returns
    -------
    coo_tab : `astropy.table.Table` or NoneType
        If one or more sources are detected, returns the
        corresponding source properties table generated
        from the `photutils` `SourceCatalog()` class. If no
        sources are detected, returns `None`.
    """

    threshold = detect_threshold(data, nsigma=nsigma)
    segm = detect_sources(data, threshold=threshold, npixels=npixels)

    if segm:
        coo_tab = SourceCatalog(data, segm).to_table()
        if show:
            show_source_detection_plot(data, coo_tab)

    if not segm:
        if show:
            show_source_detection_plot(data, None)
        coo_tab = None

    return coo_tab



def iraf_style_photometry(phot_aps, bg_aps, data,
                          error_array=None, bg_method='mode', epadu=1.0):
    """Use IRAF formulae & `photutils` apertures for photometry.

    Note
    ----
        Derivation/explanation for constant factor 1.0857
        used for calculating magnitude error can be found
        in Newberry's 'Signal-to Noise Considerations for
        Sky-Subtracted CCD Data' (1991PASP..103..122N). It
        is a correction term between the error in flux and
        in magnitude, and originates from approximating
        2.5 / ln(10).

    Parameters
    ----------
    phot_aps : `photutils.PixelAperture`
        The `phoutils` aperture object with which to
        compute the photometry (i.e. the object returned
        via `CircularAperture`).
    bg_aps : `photutils.PixelAperture`
        The `phoutils` aperture object in which to measure
        the background (i.e. the object returned via
        `CircularAnnulus`).
    data : array
        Array of image data.
    error_array: array, optional
        The array of pixelwise error of the data. If None,
        the Poisson noise term in the error computation
        will just be the square root of the flux/`epadu`.
        If not `None`, the 'aperture_sum_err' column output
        by `aperture_photometry` (divided by `epadu`) will
        be used as the Poisson noise term.
    bg_method: str , optional
        The statistic used to calculate the background;
        'mean', 'median', or 'mode'. All measurements are
        sigma-clipped. Note that as per DAOPHOT:
            `mode` = 3 * median - 2 * mean
    epadu: float, optional
        Gain in electrons per adu (only use if image units
        aren't e-).

    Returns
    -------
    final_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, flux,
        flux_error, mag, and mag_err measurements for each
        source.
    """
    if bg_method not in ['mean', 'median', 'mode']:
        raise ValueError('Invalid background method, choose either \
                          mean, median, or mode')

    phot = aperture_photometry(data, phot_aps, error=error_array)
    bg_phot = make_aperture_stats_tbl(data, bg_aps, sigma_clip=True)

    if callable(phot_aps.area):
        ap_area = phot_aps.area()
    else:
        ap_area = phot_aps.area
    bg_method_name = 'aperture_{}'.format(bg_method)

    flux = phot['aperture_sum'] - bg_phot[bg_method_name] * ap_area

    # To calculate error, need variance of sources for Poisson noise term.
    # This means photometric error needs to be squared (if error array existed).
    if error_array is not None:
        flux_error = compute_iraf_style_error(phot['aperture_sum_err']**2.0,
                                              bg_phot, bg_method, ap_area,
                                              epadu)

    # If `error_array` does not exist, error = bg-subtracted flux ** .5.
    else:
        flux_error = compute_iraf_style_error(flux**0.5, bg_phot, bg_method,
                                              ap_area, epadu)

    mag = -2.5 * np.log10(flux)
    mag_err = 1.0857 * flux_error / flux

    # Make the final table.
    X, Y = phot_apertures.positions.T
    stacked = np.stack([X, Y, flux, flux_error, mag, mag_err, ap_area], axis=1)
    names = ['X', 'Y', 'flux', 'flux_error', 'mag', 'mag_error', 'phot_ap_area']

    final_tbl = Table(data=stacked, names=names)

    return final_tbl


def show_source_detection_plot(data, coo_tab):
    """Plot detected sources, if any.

    Displays a plot of `data` (z-scaled), with sources in
    `coo_tab` overplotted.

    Parameters
    ----------
    data : array
        Image array.
    coo_tab : `astropy.table.Table`
        Table with source positions as columns 'xcentroid'
        and 'ycentroid'.

    Outputs
    -------
        Window with plot.
    """
    try:
        z1, z2 = zscale.zscale(data)
    except:
        z1, z2 = Zscale(data)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(data, origin='lower', cmap='Greys_r', vmin=z1, vmax=z2)
    if coo_tab:
        ax.scatter(coo_tab['xcentroid'], coo_tab['ycentroid'],
                    edgecolors='r', facecolors='none', s=200, zorder=10)
        if len(coo_tab) == 1:
            ax.set_title(f'1 Source Detected')
        else:
            ax.set_title(f'{len(coo_tab)} Sources Detected')
    else:
        ax.set_title('No Sources Detected')
    plt.show()
