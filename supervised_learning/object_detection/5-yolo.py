#!/usr/bin/env python3
"""Defines the Yolo class for object detection."""
import cv2
import numpy as np
import tensorflow as tf


class Yolo:
    """Contains methods for initializing and processing YOLO model data."""

    # ... __init__, process_outputs, filter_boxes, non_max_suppression, load_images ...

    def preprocess_images(self, images):
        """Resizes and rescales images for input into the YOLO model."""
        input_h = self.model.input_shape[1]
        input_w = self.model.input_shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized_img = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized_img / 255.0)

        return np.array(pimages), np.array(image_shapes)
