from pathlib import Path
import os
import sys

script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / 'data').resolve()
if not (data_dir / 'Cell_3phase.tif').exists():
    print('Testing data not available.')
    sys.exit(1)

test_psf_file = str(data_dir / 'PSF_3phase.tif')

import simtk

def test_package():
    import simtk
    assert hasattr(simtk, '__version__')

def test_load_image():
    image = simtk.load_image(test_psf_file, spacing=[0.11, 0.11, 0.1])
