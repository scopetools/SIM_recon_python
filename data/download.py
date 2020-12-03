#!/usr/bin/env python3

from urllib.request import urlretrieve
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

# Data (file_name, sha512_hash)
data = [
    (
        "Cell_3phase.tif",
        "645cb8970de3370fb15835107843710e799f72f58012003db35b056160ad4de995622e12727d0b739a8afd1586c3faa178f3b71c39bd4dbc9308e8ef5ebb8e11",
    ),
    (
        "Cell_5phase_small.tif",
        "dd39269640efc251686f6e80bc42e3438e6f308274ca3d0b99e8da02f5976ad2cbd10cb594d2811631d0273413711096388b3a609eb41df2b04a1beeca4d91d1",
    ),
    (
        "PSF_3phase.tif",
        "80cc81d668bdcc6847599bb7eab477dd169024093b07ab743154d2e6df86471d955fba1234c9d728d28e2000be72f1f19677dad8f0e710f9c26fee1c3c0d2a3d",
    ),
    (
        "PSF_5phase.tif",
        "131f0621562a3832d270683c84d414acc20aacd7c1dae494dceef0cb156c170b6836f81544b80877978839a9eb82a882d3b5754040f6c098babd13be02f2d79e",
    ),
]

for file_name, sha512_hash in data:
    file_path = os.path.join(script_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}")
        url = f"https://data.kitware.com/api/v1/file/hashsum/sha512/{sha512_hash}/download"
        urlretrieve(url, file_path)
