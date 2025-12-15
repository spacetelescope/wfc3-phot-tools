"""
This module contains functions for computing background
statistics within `photutils` apertures.

Authors
-------
    Varun Bajaj, 2017-2018
    Clare Shanahan, December 2019
    Mariarosa Marinelli, 2022-2023

Use
---
    This script is intended to be imported:

        from wfc3_phot_tools.staring_mode import background

    or:

        from wfc3_phot_tools.staring_mode.background \
            import make_aperture_stats_tbl

Functions
---------
_gauss(x, *p)
    Helper function to create a Gaussian distribution.
calc_1d_gauss_background(data, bins, hist_range)
    Fits 1D Gaussian to data and returns the coefficients.
calc_aperture_stats(data, mask, sigma_clip)
    Computes stats in an aperture/annulus.
make_aperture_stats_tbl(data, apertures, method, sigma_clip)
    Computes statistics in `photutils` aperture(s).
"""
import numpy as np
from astropy.table import Table
from photutils.aperture.mask import ApertureMask
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import sigmaclip


def _gauss(x, *p):
    """Helper function to create a Gaussian distribution.
    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def calc_1d_gauss_background(data, bins=100, hist_range=(-10, 10)):
    """Fits 1D Gaussian to data and returns the coefficients.

    Parameters
    -----------
    data : array-like
        Science array.
    bins : int
        Number of bins for histogram fit.
    hist_range : tuple of ints
        Range for fit (min, max).
    Returns
    -------
    coeff : tuple
        A, mu, sigma of gaussian fit.
    """
    try:
        data = data.flatten()
        h, b = np.histogram(data, range=hist_range, bins=bins)
        bd = b[1]-b[0]
        bdist = int(1.1/bd)
        locs = find_peaks(h, distance=bdist)[0]
        p0 = [1., 0., 1.]
        centers = .5*b[:-1] + .5*b[1:]
        coeff, var_matrix = curve_fit(_gauss, centers[locs], h[locs], p0=p0)
        # Get the fitted curve - necessary?
        hist_fit = _gauss(centers, *coeff)

    except RuntimeError:
        coeff = None, None, None

    return coeff


def calc_aperture_stats(data, mask, sigma_clip):
    """Computes stats in an aperture/annulus.

    Helper function to calculate the statistics for pixels
    falling within some Photutils aperture mask on some
    array of data.

    Parameters
    ----------
    data : array-like
        Input image data.
    mask : array-like
        Mask from `photutils` apertures.
    sigma_clip : Boolean
        Flag to activate sigma clipping of background pixels

    Returns
    -------
    stats : list
        List of the mean, median, mode, standard deviation,
        and actual area of non-NaN pixels in the background
        aperture/annulus.
    """
    cutout = mask.cutout(data, fill_value=np.nan)

    if cutout is None:
        mean = median = mode = std = actual_area = -9999.

    else:
        values = cutout
        values = values[~np.isnan(values)]

        if sigma_clip:
            values, clow, chigh = sigmaclip(values, low=3, high=3)

        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)

        mode = 3 * median - 2 * mean
        actual_area = (~np.isnan(values)).sum()

    stats = [mean, median, mode, std, actual_area]

    return stats


def make_aperture_stats_tbl(data, apertures, method='exact', sigma_clip=True):
    """Computes statistics in `photutils` aperture(s).

    Compute statistics for custom local background methods.
    This is primarily intended for estimating backgrounds
    via annulus apertures. The intent is that this falls easily
    into other code to provide background measurements.

    Parameters
    ----------
    data : array
        The data for the image to be measured.
    apertures : photutils PixelAperture object (or subclass)
        The phoutils aperture object in which to measure
        the background stats (i.e. the object returned via
        `CircularAperture`, `CircularAnnulus`, or
        `RectangularAperture`).
    method: str
        The method by which to handle the pixel overlap.
        Defaults to computing the exact area.
        NOTE: Currently, this will actually fully include a
        pixel where the aperture has ANY overlap, as a
        median is also being performed.  If the method is
        set to 'center' the pixels will only be included if
        the pixel's center falls within the aperture.
    sigma_clip: bool
        Flag to activate sigma clipping of background pixels

    Returns
    -------
    aperture_stats_tbl : astropy.table.Table
        An astropy Table with the columns 'x', 'y', 'mean',
        'median', 'mode', 'std', 'aperture_nonnan_area',
        and a row for each of the sets of coordinates
        specified in `apertures.positions`.
    """
    # Get the masks that will be used to identify our desired pixels.
    masks = apertures.to_mask(method=method)
    if isinstance(masks, ApertureMask): # this fixes different return types
        masks = [masks]

    # Compute the stats of pixels within the masks.
    aperture_stats_rows = [calc_aperture_stats(data, mask, sigma_clip)
                           for mask in masks]

    aperture_stats_tbl = Table(rows=aperture_stats_rows,
                               names=('aperture_mean', 'aperture_median',
                                      'aperture_mode', 'aperture_std',
                                      'aperture_nonnan_area'))

    # Hopefully this will maintain backwards compatibility
    if apertures.positions.shape == (2,):
        aperture_stats_tbl['x'] = [apertures.positions[0]]
        aperture_stats_tbl['y'] = [apertures.positions[1]]

    else:
        aperture_stats_tbl['x'] = [position[0] for position in apertures.positions]
        aperture_stats_tbl['y'] = [position[1] for position in apertures.positions]


    return aperture_stats_tbl
