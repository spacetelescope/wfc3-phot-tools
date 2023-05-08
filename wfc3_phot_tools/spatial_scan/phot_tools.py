"""
    This module contains functions to perform source
    detection and aperture photometry on spatial scan data.

    Authors
    -------
        Mariarosa Marinelli, June 2022; May 2023
        Clare Shanahan, Dec 2019

    Use
    ---
        This script is intended to be imported:

            from wfc3_phot_tools.spatial_scan import phot_tools

"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.stats import sigmaclip
from photutils import (detect_sources, detect_threshold, segmentation,
                       RectangularAperture, aperture_photometry)

def _check_filepath(filepath):
    """
    Helper function for saving source detection images
    generated by `detect_sources_scan()`.

    Parameter
    ---------
    filepath : str or NoneType

    Returns
    -------
    filepath : str
    """
    cwd = os.getcwd()
    default_filename = 'image.jpg'
    default_filepath = os.path.join(cwd, default_filename)

    if (filepath == None) or (filepath == ''):
        message = 'No file name or path specified.\n'
        filepath = default_filepath

    else:
        # if the filepath is a directory (does not have an extension)
        if os.path.splitext(filepath)[-1] == '':
            # if the directory exists:
            if os.path.isdir(filepath):
                message = 'No file name specified.\n'
                filepath = os.path.join(filepath, default_filename)
            # if the directory does not exist:
            else:
                message = 'No file name specified, and specified '\
                          f'directory is invalid: {filepath}\n'
                filepath = default_filepath

        #if the filepath contains an extension
        else:
            dirname = os.path.dirname(filepath)
            # if there isn't a directory in the filepath
            if dirname == '':     # if no directory specified, only file name
                message = f'No directory specified.\n'
                filepath = os.path.join(cwd, filepath)
            # if there is a directory in the filepath
            else:
                # if the specified directory exists
                if os.path.isdir(dirname):
                    message = ''
                # if the specified directory does not exist
                else:
                    message = f'Specified directory is invalid: {dirname}\n'
                    filename = os.path.basename(filepath)
                    filepath = os.path.join(cwd, filename)

    ext = filepath.split('.')[-1]
    # if the extension is not a valid image extension
    if ext not in ['jpeg', 'jpg', 'pdf', 'png', 'svg']:
        message = f'{message}Invalid image extension: {ext}\n'
        filepath = filepath.replace(ext, 'jpg')

    print(f'{message}Saving source detection image to {filepath}')
    return filepath


def detect_sources_scan(data, snr_threshold=3.0,
                        sigma_kernel=3,
                        size_kernel=(3, 3),
                        n_pixels=250,
                        show=False,
                        save=False,
                        title='',
                        filepath=None):
    """Detects sources in a spatial scan.

    Uses image segmentation to detect sources in spatially
    scanned images.

    A pixel-wise threshold image used to detect sources is
    generated based on the data and the snr_threshold
    provided. Data is then convolved with a 2D Gaussian
    kernel, of width sigma_kernel (default 3.0) and x, y
    size given by size_kernel (default 3 pixels x 3 pixels)
    to smooth out some of the background noise.

    A segmentation map of the convolved image is generated
    using the threshold image and npixels, the lower limit
    on the number of connected pixels that represent a true
    source (default is 1000., since scans cover a larger
    area of the detector).

    Optionally, a plot showing the detected source(s) can
    be shown.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D array of floats representing the scan image.
    snr_threshold : int or float
        For creation of the threshold image, the signal-to-
        noise ratio per pixel above the background used for
        which to consider a pixel as possibly being part of
        a source. The background is calculated for the
        entire image using sigma-clipped statistics.
    sigma_kernel : float or int
        Width of 2D gaussian kernel, in pixels.
    size_kernel : tuple
        x, y size in pixels of kernel.
    n_pixels : int
        The (positive) integer number of connected pixels,
        each greater than the threshold, that an object
        must have to be detected.
    save : bool
        If True, the segmentation map and data image will
        be saved.
    show : bool
        If True, the image will be displayed with all of
        the identified sources marked.
    title : str
        Optional string to add to the image title.
    filepath : None or str
        Optional string to define filepath and filename
        where image should be saved. If `None` and
        `save=True`, will save image to current working
        directory. If no filename is specified, image will
        be saved as `image.jpg`. If parent directory does
        not exist or is not specified, image will be saved
        to current working directory.

    Returns
    -------
    props_tbl : `astropy.table.table.Qtable` or NoneType
        If sources are detected, returns an Astropy QTable
        containing properties of detected sources. If no
        sources are detected (if the segmentation map is
        None), returns None.
    """
    # make threshold image
    threshold = detect_threshold(data, nsigma=snr_threshold)

    # construct gaussian kernel to smooth image
    # don't think this is actually needed
    sigma = sigma_kernel * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=size_kernel[0],
                              y_size=size_kernel[1])
    kernel.normalize()

    # pass in data, convolution kernel to make segmentation map
    segm = detect_sources(data, threshold, npixels=n_pixels)

    if segm == None:
        props_tbl = None

    else:
        props = segmentation.SourceCatalog(data, segm)
        props_tbl = props.to_table()

        if show or save:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
            ax[0].imshow(segm.data, origin='lower')
            ax[0].set_title('segmentation map')

            ax[1].scatter(props_tbl['xcentroid'],
                          props_tbl['ycentroid'],
                          marker='x', s=100, c='r')
            z1, z2 = (-12.10630989074707, 32.53888328838081)
            ax[1].imshow(data, origin='lower', cmap='Greys_r', vmin=z1, vmax=z2)
            ax[1].set_title('data')

            title_str = f'{title}{len(props_tbl)} detected source'
            if len(props_tbl) > 1:
                title_str = title_str+'s'
            plt.suptitle(title_str)
            plt.tight_layout()

            if save:
                filepath = _check_filepath(filepath)
                plt.savefig(filepath, dpi=250)

            if show:
                plt.show()

            plt.close()

    return props_tbl

def isolate_sky_rind(data, x_pos, y_pos,
                     source_mask_len, source_mask_width,
                     n_pix):
    """Masks out data not in the sky background rind.

    Parameters
    ----------
    data : array-like
    x_pos : int or float
    y_pos : int or float
    source_mask_len : int
    source_mask_width : int
    n_pix : int

    Returns
    -------
    rind_data : array-like
        Input array, with data not in the sky background
        masked with `np.nan`s.
    """
    rind_data = copy.deepcopy(data)

    ap_y = source_mask_len/2.
    ap_x = source_mask_width/2.

    # mask inside inner boundary:
    rind_data[int(y_pos-ap_y):int(y_pos+ap_y),
              int(x_pos-ap_x):int(x_pos+ap_x)] = np.nan

    # mask outside of 30 pixel rind:

    # mask left and bottom:
    x_l = int(x_pos-ap_x-n_pix)
    y_b = int(y_pos-ap_y-n_pix)

    rind_data[:, :x_l] = np.nan
    rind_data[:y_b, :] = np.nan

    # mask right and top:
    x_r = int(x_pos+ap_x+n_pix)
    y_t = int(y_pos+ap_y+n_pix)

    rind_data[:, x_r:] = np.nan
    rind_data[y_t:, :] = np.nan

    return rind_data


def calc_sky(data, x_pos, y_pos,
             source_mask_len, source_mask_width,
             n_pix=30, method='median'):
    """Calculates sky flux level in background rind.

    Calculates sky level in a rectangular annulus or "rind"
    around a scanned source. The `source_mask` parameters
    determine the inner boundary of the rind, within which
    all pixels will be masked. `n_pix` provides the width
    of the sky background rind, such that the dimensions of
    the rind's outer boundary (outside of which all pixels
    are masked) are (2*n_pix + source_mask_width,
    2*n_pix + source_mask_len), centered over the same
    reference point as the photometric aperture (should be
    the midpoint of the scan). Background is thus computed
    from the pixels between the inner and outer boundaries.
    By default, the background is calculated as a sigma-
    clipped median; as an alternative, a sigma-clipped mean
    can be returned.

    Parameters
    ----------
    data : 'numpy.ndarray'
        2D array of floats representing the scan image.
    x_positions : float
        X position of source.
    y_positions : float
        Y position of source.
    source_mask_len : int
        Length of rectangle (along y axis) used to mask
        source. Should generally be `sky_ap_dim[1]`.
    source_mask_width : int
        Width of rectangle (along x axis) used to mask
        source. Should generally be `sky_ap_dim[0]`.
    n_pix : int
        Number of pixels around source masking rectangle
        that define the rind region used to measure the
        background. Default is 30.
    method : str
        'Median' or 'Mean', sigma clipped.

    Returns
    -------
    back : float
        Either the median or mean of the pixels in the
        sky background aperture, in units of counts per
        second.
    back_rms : float
        The standard deviation of the pixel background
        level inside the sky aperture, in units of counts
        per second.
    """
    rind_data = isolate_sky_rind(data, x_pos, y_pos,
                                 source_mask_len, source_mask_width,
                                 n_pix)

    flat_dat = rind_data.flatten()
    # Only use data where value is not nan:
    flat_masked_dat = flat_dat[~np.isnan(flat_dat)]

    clipped, low, upp = sigmaclip(flat_masked_dat)
    backrms = np.std(clipped)
    if method == 'median':
        back = np.median(clipped)
    if method == 'mean':
        back = np.mean(clipped)

    return back, backrms


def aperture_photometry_scan(data, x_pos, y_pos, ap_width, ap_length,
                             theta=0.0, show=False, plt_title=None):
    """Calculates photometry in rectangular aperture.

    Performs aperture photometry on source located at (`x_pos`,
    `y_pos`), using a rectangular aperture of dimensions
    specified by (`ap_length`, `ap_width`).

    Note
    ----
    Aperture sums are NOT sky-subtracted by default!

    Parameters
    ----------
    data : 'numpy.ndarray'
        2D array of floats representing the scan image.
    x_pos : float
        X position of source.
    y_pos : float
        Y position of source
    ap_width : int
        Width (along x axis) of photometric aperture.
    ap_length : int
        Length (along y axis) of photometric aperture.
    theta : float
        Angle of orientation (from x-axis) for aperture,
        in radians. Increases counter-clockwise.
    show : bool, optional
        If true, plot showing aperture(s) on source will
        pop up. Defaults to F.
    plt_title : str or None, optional
        Only used if `show` is True. Title for plot.
        Defaults to None.

    Returns
    -------
    phot_tab : `astropy.table`
        Table containing photometric sum.
    """
    copy_data = copy.deepcopy(data)

    rect_ap = RectangularAperture((x_pos, y_pos), w=ap_width,
                                  h=ap_length, theta=theta)

    phot_table = aperture_photometry(copy_data, rect_ap,
                                     method='exact')
    if show:
        mask = rect_ap.to_mask(method='center')
        data_cutout = mask.cutout(data)
        plt.title(plt_title)
        z1, z2 = (-12.10630989074707, 32.53888328838081)
        plt.imshow(data, origin='lower', vmin=z1, vmax=z2)
        rect_ap.plot(color='white', lw=2)
        plt.show()
        plt.close()

    return phot_table
