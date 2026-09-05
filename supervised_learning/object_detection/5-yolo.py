#!/usr/bin/env python3
"""Defines the Yolo class for object detection."""
import cv2
import numpy as np
import tensorflow as tf


class Yolo:
    """Contains methods for initializing and processing YOLO model data."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo class instance."""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    # ... здесь должны быть остальные методы: process_outputs, filter_boxes,
    # non_max_suppression, load_images ...

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
