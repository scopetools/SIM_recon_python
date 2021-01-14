from pathlib import Path
import os
import sys

import itk
import numpy as np

script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / ".." / "data").resolve()
if not (data_dir / "input" / "Cell_3phase.tif").exists():
    print("Testing data not available.")
    sys.exit(1)
package_dir = (script_dir / "..").resolve()
sys.path.insert(0, str(package_dir))

test_psf_file = str(data_dir / "input" / "PSF_3phase.tif")

import simtk.pydra


def test_import_simtk_pydra():
    assert "simtk" in simtk.pydra.__doc__
    assert "pydra" in simtk.pydra.__doc__
