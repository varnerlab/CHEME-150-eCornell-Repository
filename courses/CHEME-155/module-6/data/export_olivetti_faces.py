"""
Export the Olivetti Faces dataset to CSV files for use in Julia.

Downloads the .mat file from figshare (the same source scikit-learn uses),
parses it with only numpy (no scipy), and writes CSV files.

Dataset: 400 grayscale images (40 subjects, 10 images each), each 64x64 pixels.
Pixel values are floats in [0, 1].

Outputs:
    olivetti_faces.csv   - 400 rows x 4096 columns (each row is a flattened 64x64 image)
    olivetti_targets.csv - 400 rows x 1 column (subject ID, 0-39)

Usage:
    python3 export_olivetti_faces.py

Note: If your numpy/scipy is broken, run this in a fresh venv:
    python3 -m venv /tmp/olivetti_env
    /tmp/olivetti_env/bin/pip install numpy scipy
    /tmp/olivetti_env/bin/python export_olivetti_faces.py
"""
import urllib.request
import io
import numpy as np
from scipy.io import loadmat

url = "https://ndownloader.figshare.com/files/5976027"
print("Downloading Olivetti faces dataset...")
response = urllib.request.urlopen(url)
mat = loadmat(io.BytesIO(response.read()))

faces = mat["faces"].T  # transpose to get 400 x 4096
faces = faces / 255.0   # normalize to [0, 1]

# each subject has 10 consecutive images
targets = np.repeat(np.arange(40), 10)

np.savetxt("olivetti_faces.csv", faces, delimiter=",", fmt="%.6f")
np.savetxt("olivetti_targets.csv", targets, delimiter=",", fmt="%d")

print(f"Exported {faces.shape[0]} images (each 64x64 = {faces.shape[1]} pixels)")
print(f"  olivetti_faces.csv   : {faces.shape}")
print(f"  olivetti_targets.csv : {targets.shape}")
