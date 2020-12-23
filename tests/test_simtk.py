from pathlib import Path
import os
import sys

import itk
import numpy as np

script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / '..' / 'data').resolve()
if not (data_dir / 'input' / 'Cell_3phase.tif').exists():
    print('Testing data not available.')
    sys.exit(1)
package_dir = (script_dir / '..').resolve()
sys.path.insert(0, str(package_dir))

test_psf_file = str(data_dir / 'input' / 'PSF_3phase.tif')

from simtk import load_image, normalize_psf

def test_package():
    import simtk
    assert hasattr(simtk, '__version__')

def test_load_image(benchmark):
    image = benchmark(load_image, test_psf_file, spacing=[0.11, 0.11, 0.1])

    baseline = str(data_dir / 'baseline' / 'load_image.nrrd')
    baseline_image = itk.imread(baseline)

    image_arr = itk.array_view_from_image(image)
    baseline_arr = itk.array_view_from_image(baseline_image)
    assert np.array_equal(image_arr, baseline_arr)

    image_spacing = np.asarray(itk.spacing(image))
    baseline_spacing = np.asarray(itk.spacing(baseline_image))
    assert np.array_equal(image_spacing, baseline_spacing)

def test_normalize_psf(benchmark):
    psf_filename = str(data_dir / 'baseline' / 'load_image.nrrd')
    psf = itk.imread(psf_filename)
    normalized = benchmark(normalize_psf, psf)

    normalized_arr = itk.array_view_from_image(normalized)
    for component in range(normalized_arr.shape[-1]):
        mean = np.mean(normalized_arr[...,component])
        assert np.isclose(float(mean), 0.0, atol=1e-4)
        sum_ = np.sum(normalized_arr[...,component])
        assert np.isclose(float(sum_), 1.0, atol=1e-2)

    baseline = str(data_dir / 'baseline' / 'normalize_psf.nrrd')
    baseline_image = itk.imread(baseline)

    baseline_arr = itk.array_view_from_image(baseline_image)
    assert np.array_equal(normalized_arr, baseline_arr)

    normalized_spacing = np.asarray(itk.spacing(normalized))
    baseline_spacing = np.asarray(itk.spacing(baseline_image))
    assert np.array_equal(normalized_spacing, baseline_spacing)
