#!/usr/bin/env python3
"""
Contains the Yolo class for object detection
"""
import cv2
import glob
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes based on objectness score and class threshold
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered bounding boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for c in unique_classes:
            idxs = np.where(box_classes == c)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            keep = np.array(keep)
            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), c))
            predicted_box_scores.append(cls_scores[keep])

        if len(box_predictions) > 0:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0)
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0)
        else:
            box_predictions = np.array([])
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder path
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths
