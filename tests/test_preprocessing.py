"""
Tests for the preprocessing module.

These tests ensure that images are loaded correctly and basic preprocessing
operations (grayscale conversion, Gaussian blur, binary thresholding and
inversion) behave as expected.
"""
import numpy as np
import cv2
import os
from signature_detector.modules import preprocessing


def test_preprocess_image_basic(tmp_path):
    """Create a simple synthetic image and verify preprocessing output."""
    # Create a simple black image with a white rectangle (simulating a letter)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 40), (150, 60), (255, 255, 255), thickness=-1)
    # Save to a temporary file so load_image can read it
    tmp_file = tmp_path / 'test_img.png'
    cv2.imwrite(str(tmp_file), img)
    # Load and preprocess using our functions
    loaded = preprocessing.load_image(str(tmp_file))
    processed = preprocessing.preprocess_image(loaded)
    # Check that the processed image is 2D (grayscale) and binary
    assert processed.ndim == 2
    # Values should be 0 or 255 only
    unique_vals = np.unique(processed)
    assert set(unique_vals.tolist()).issubset({0, 255})
    # The white rectangle should remain white (255) in the processed image
    assert processed[50, 100] == 255
