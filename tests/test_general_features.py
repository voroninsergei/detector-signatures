"""
Tests for the general_features module.

These tests validate the computation of letter sizes, spacing, slant and
connectivity on simple synthetic binary images.
"""
import numpy as np
import cv2
from signature_detector.modules import general_features


def create_binary_image_with_letters() -> np.ndarray:
    """
Create a simple binary image with two rectangular 'letters' separated by a
fixed gap. Letters are white on a black background. Returns the inverted
binary image (white letters = 255).
    """
    img = np.zeros((50, 100), dtype=np.uint8)
    # two rectangles of width 10 and height 20
    cv2.rectangle(img, (10, 15), (20, 35), 255, thickness=-1)
    cv2.rectangle(img, (40, 15), (50, 35), 255, thickness=-1)
    return img


def test_compute_letter_sizes():
    img = create_binary_image_with_letters()
    avg, std = general_features.compute_letter_sizes(img)
    # Both letters have height 20 pixels, so average should be 20 and std close to 0
    assert abs(avg - 20) < 0.1
    assert std < 0.1


def test_compute_spacing():
    img = create_binary_image_with_letters()
    avg, std = general_features.compute_spacing(img)
    # Gap between letters horizontally is 20 pixels minus widths (should be approx 10)
    assert abs(avg - 10) < 1.0
    # There is only one gap so std should be 0
    assert std < 1e-3


def test_compute_connectivity():
    img = create_binary_image_with_letters()
    # connectivity should be 0, as there are no connected letters
    conn = general_features.compute_connectivity(img)
    assert conn == 0.0
