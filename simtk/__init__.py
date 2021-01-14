"""sim: structured illumination microscopy"""

__version__ = "0.1.0"

__all__ = [
        'load_image',
        'normalize_psf',
        'center_psf',
        'generate_otf',
        ]

import itk
import numpy as np
import scipy.optimize

def load_image(filename, nphases=3, norientations=1, spacing=None):
    """Load SIM acquisition data from an image file.

    Read the filename containing multiple information components, reformat the
    data into a multi-component vector image, and optionally override the voxel
    spacing information.

    Parameters
    ----------

    filename: str
        Path to the SIM acquisition data, often a TIFF file.
    nphases: int
        Number of phases acquired in every plane.
    norientations: int
        Number of orientations acquired in every plane.
    spacing: array of float's
        Override image spacing metadata with the provided [dx, dy, dz] spacing.

    Returns
    -------

    itk.VectorImage
        3D, multi-component SIM acquition image. Each component corresponds to an
        specific phase and orientation information component.
    """

    image = itk.imread(filename)
    image.DisconnectPipeline()
    data_view = itk.array_view_from_image(image)

    size = itk.size(image)
    components = nphases*norientations

    result_shape = list(data_view.shape)
    result_shape[0] = int(result_shape[0] / components)
    result_shape = result_shape + [components,]
    result = np.empty(result_shape, dtype=np.float32)
    for component in range(components):
        result[:,:,:,component] = data_view[component::components,:,:]

    result = itk.PyBuffer[itk.VectorImage[itk.F, 3]].GetImageViewFromArray(result, is_vector=True)
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(image.GetSpacing())
    result.SetDirection(image.GetDirection())
    if spacing is not None:
        result.SetSpacing(spacing)

    return result

def normalize_psf(psf):
    """Remove DC fourier component, normalize information component to sum to
    unity.

    Parameters
    ----------

    psf: itk.VectorImage
        SIM point spread function (PSF) acquisition data, e.g. from simtk.load_image.

    Returns
    -------

    itk.VectorImage
        Normalized PSF.
    """

    psf_arr = itk.array_view_from_image(psf)
    result = np.copy(psf_arr)
    for component in range(psf_arr.shape[-1]):
        mean = np.mean(psf_arr[..., component])
        result[..., component] = result[..., component] - mean
        result[result < 0.0] = 0.0
        sum_ = np.sum(result[..., component])
        result[..., component] = result[..., component] / sum_

    result_image = itk.PyBuffer[itk.VectorImage[itk.F, 3]].GetImageViewFromArray(result, is_vector=True)
    result_image.SetOrigin(psf.GetOrigin())
    result_image.SetSpacing(psf.GetSpacing())
    result_image.SetDirection(psf.GetDirection())

    return result_image

def center_psf(psf):
    """Center the PSF in the image. The data is resampled around the center and
    the origin of the output is set to the center.

    Parameters
    ----------

    psf: itk.VectorImage
        SIM point spread function (PSF) acquisition data, e.g. from
        simtk.normalize_psf.

    Returns
    -------

    itk.VectorImage
        Centered PSF.
    """

    psf_arr = itk.array_view_from_image(psf)
    widefield_psf = np.zeros(psf_arr.shape[:-1], dtype=np.float32)
    for component in range(psf_arr.shape[-1]):
        widefield_psf[...] += psf_arr[..., component]
    widefield_psf /= np.sum(widefield_psf)

    (nz, ny, nx) = widefield_psf.shape
    (zz, yy, xx) = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij')
    peak_image_flat = np.ravel(widefield_psf)
    peak_data = np.stack((peak_image_flat,
                          np.ravel(zz),
                          np.ravel(yy),
                          np.ravel(xx)),
                         axis=1)

    max_value = np.max(peak_image_flat)
    max_index = np.argmax(peak_image_flat)
    (cz, cy, cx) = np.unravel_index(max_index, widefield_psf.shape)

    initial_params = np.zeros((7,), dtype=np.float64)
    initial_params[0] = max_value # max value in the window
    initial_params[1] = cz # z pixel index of the peak
    initial_params[2] = cy # y pixel index of the peak
    initial_params[3] = cx # x pixel index of the peak
    initial_params[4] = 1.5 # z sigma
    initial_params[5] = 1.5 # y sigma
    initial_params[6] = 1.5 # x sigma

    def gaussian_fit_objective(params, peak_data):
        actual = peak_data[:,0]
        expected = params[0] * np.exp(-((peak_data[:,1] - params[1])**2 / (2*params[4]**2) + \
                                        (peak_data[:,2] - params[2])**2 / (2*params[5]**2) + \
                                        (peak_data[:,3] - params[3])**2 / (2*params[6]**2)))
        err = actual - expected
        result = np.sum(err**2)
        return result

    params_opt = scipy.optimize.fmin(func=gaussian_fit_objective,
                                     x0=initial_params,
                                     args=(peak_data,),
                                     maxiter=10000,
                                     maxfun=10000,
                                     disp=False,
                                     xtol=1e-6,
                                     ftol=1e-6)
    (cz_opt, cy_opt, cx_opt) = params_opt[1:4]

    result = np.copy(psf_arr)
    transform = itk.TranslationTransform[itk.D, 3].New()
    transform_params = transform.GetParameters()
    transform_params[0] = float(cx) - cx_opt
    transform_params[1] = float(cy) - cy_opt
    transform_params[2] = float(cz) - cz_opt
    transform.SetParameters(transform_params)
    interpolator = itk.WindowedSincInterpolateImageFunction[itk.Image[itk.F,3],
            3, itk.HammingWindowFunction[3]].New()
    for component in range(psf_arr.shape[-1]):
        component_arr = psf_arr[..., component]
        component_image = itk.image_view_from_array(component_arr)
        resampled = itk.resample_image_filter(component_image,
                                              use_reference_image=True,
                                              reference_image=component_image,
                                              transform=transform,
                                              interpolator=interpolator,
                                              )
        resampled_arr = itk.array_view_from_image(resampled)
        result[..., component] = resampled_arr

    result_image = itk.PyBuffer[itk.VectorImage[itk.F, 3]].GetImageViewFromArray(result, is_vector=True)
    origin = list(itk.origin(psf))
    spacing = itk.spacing(psf)
    origin[0] -= cx * spacing[0]
    origin[1] -= cy * spacing[1]
    origin[2] -= cz * spacing[2]
    result_image.SetOrigin(origin)
    result_image.SetSpacing(spacing)
    result_image.SetDirection(psf.GetDirection())

    return result_image

def generate_otf(psf, nphases=3, norientations=1):
    """Generate the OTF from the PSF.
    the origin of the output is set to the center.

    Parameters
    ----------

    psf: itk.VectorImage
        SIM point spread function (PSF) acquisition data, e.g. from
        simtk.center_psf.
    nphases: int
        Number of phases acquired in every plane.
    norientations: int
        Number of orientations acquired in every plane.

    Returns
    -------

    ndarray with dimensions [w, u, v, phase, orientation]
    """

    pass
