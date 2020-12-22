from pathlib import Path
import os
import sys

script_dir = Path(__file__).resolve().parent
data_dir = (script_dir / 'data').resolve()
print(data_dir)
if not (data_dir / 'Cell_3phase.tif').exists():
    print('Testing data not available.')
    sys.exit(1)


def test_package():
    import simtk
    assert hasattr(simtk, '__version__')
