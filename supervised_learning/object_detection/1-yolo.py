#!/usr/bin/env python3
"""
Contains the Yolo class for object detection processing
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class for performing object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path: path to the Keras model
            classes_path: path to file containing class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs for a single image.

        Args:
            outputs: list of numpy.ndarrays containing predictions from
                     Darknet for a single image. Shape of each element:
                     (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing original image size
                        [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs):
                - boxes: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 4) containing
                  (x1, y1, x2, y2) relative to original image
                - box_confidences: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 1) containing
                  box confidences
                - box_class_probs: list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, classes) containing
                  class probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]
        input_height = int(self.model.input.shape[1])
        input_width = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract tx, ty, tw, th predictions
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Apply sigmoid to center offsets
            sig_tx = 1 / (1 + np.exp(-t_x))
            sig_ty = 1 / (1 + np.exp(-t_y))

            # Create grid indices (cy, cx)
            cy, cx = np.indices((grid_height, grid_width))
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Calculate center coordinates (bx, by) scaled to original image
            b_x = (sig_tx + cx) / grid_width * image_width
            b_y = (sig_ty + cy) / grid_height * image_height

            # Extract anchor box dimensions (pw, ph)
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Calculate box width and height (bw, bh) scaled to original image
            b_w = (pw * np.exp(t_w)) / input_width * image_width
            b_h = (ph * np.exp(t_h)) / input_height * image_height

            # Convert (bx, by, bw, bh) to corner coordinates (x1, y1, x2, y2)
            x1 = b_x - b_w / 2
            y1 = b_y - b_h / 2
            x2 = b_x + b_w / 2
            y2 = b_y + b_h / 2

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Extract and activate box confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
    