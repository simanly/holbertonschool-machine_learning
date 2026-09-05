#!/usr/bin/env python3
"""
Contains the Yolo class for object detection
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class for object detection using YOLO v3
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the model outputs to produce boundary boxes,
        box confidences, and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size[0], image_size[1]
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            sig_tx = 1 / (1 + np.exp(-t_x))
            sig_ty = 1 / (1 + np.exp(-t_y))

            cy, cx = np.indices((grid_h, grid_w))
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            b_x = (sig_tx + cx) / grid_w
            b_y = (sig_ty + cy) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            b_w = (pw * np.exp(t_w)) / input_w
            b_h = (ph * np.exp(t_h)) / input_h

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_conf)

            box_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_prob)

        return boxes, box_confidences, box_class_probs
    