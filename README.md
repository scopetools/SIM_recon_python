# simtk: structured illumination microscopy image reconstruction

![Build and test](https://github.com/sim-reconstruction/simtk/workflows/Build%20and%20test/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sim-reconstruction/simtk/blob/master/LICENSE)

Open source structured illumination microscopy image reconstruction in an
easy-to-use, performant, and extensible package.

Installation
------------

```
pip install simtk
```

Hacking
-------

Contributions are welcome and appreciated.

To set up a development build::

```
git clone https://github.com/sim-reconstruction/simtk
cd simtk

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests.py -v --benchmark-autosave --cov=simtk
```
