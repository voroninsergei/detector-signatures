"""
Tests for the comparative module.

These tests verify that the similarity metric returns high values for
similar images and lower values for dissimilar ones.  Synthetic images
are used to isolate the behaviour of the comparison function.
"""
import numpy as np
import cv2
import os
from signature_detector.modules import comparative, preprocessing


def create_simple_image(path: str, shift: int = 0):
    """Create a synthetic image with a single white line shifted horizontally."""
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.line(img, (10 + shift, 25), (90 + shift, 25), (255, 255, 255), 2)
    cv2.imwrite(path, img)


def test_compare_identical(tmp_path):
    """Two identical images should yield high similarity (~1)."""
    file_a = tmp_path / 'a.png'
    file_b = tmp_path / 'b.png'
    create_simple_image(str(file_a), shift=0)
    create_simple_image(str(file_b), shift=0)
    sim, details = comparative.compare_images(str(file_a), str(file_b))
    assert sim >= 0.75


def test_compare_different(tmp_path):
    """Images with different line positions should yield lower similarity."""
    file_a = tmp_path / 'a.png'
    file_b = tmp_path / 'b.png'
    create_simple_image(str(file_a), shift=0)
    create_simple_image(str(file_b), shift=10)
    sim, details = comparative.compare_images(str(file_a), str(file_b))
    assert sim < 0.75
