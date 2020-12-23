"""sim: structured illumination microscopy"""

__version__ = "0.1.0"

__all__ = [
        'load_image',
        ]

import itk
import numpy as np

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
