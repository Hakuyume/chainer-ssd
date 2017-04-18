import itertools
import numpy as np

from lib import matrix_iou


class MultiBoxEncoder(object):

    def __init__(self, model):
        self.variance = model.variance

        default_boxes = list()
        for k in range(len(model.grids)):
            for v, u in itertools.product(range(model.grids[k]), repeat=2):
                cx = (u + 0.5) * model.steps[k]
                cy = (v + 0.5) * model.steps[k]

                s = model.sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(model.sizes[k] * model.sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = model.sizes[k]
                for ar in model.aspect_ratios[k]:
                    default_boxes.append(
                        (cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    default_boxes.append(
                        (cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self.default_boxes = np.array(default_boxes)

    def encode(self, boxes, labels, threshold=0.5):
        if len(boxes) == 0:
            return (
                np.empty(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        iou = matrix_iou(
            np.hstack((
                self.default_boxes[:, :2] - self.default_boxes[:, 2:] / 2,
                self.default_boxes[:, :2] + self.default_boxes[:, 2:] / 2)),
            boxes)
        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        boxes = boxes[gt_idx]
        labels = labels[gt_idx]

        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_bboxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + labels
        conf[iou < threshold] = 0

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc, conf, nms_threshold, conf_threshold):
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        conf = np.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        scores = conf[:, 1:]

        all_boxes = list()
        all_labels = list()
        all_scores = list()

        for label in range(scores.shape[1]):
            mask = scores[:, label] >= conf_threshold
            label_boxes = boxes[mask]
            label_scores = scores[mask, label]

            selection = np.zeros(len(label_scores), dtype=bool)
            for i in label_scores.argsort()[::-1]:
                iou = matrix_iou(
                    label_boxes[np.newaxis, i],
                    label_boxes[selection])
                if (iou > nms_threshold).any():
                    continue
                selection[i] = True

                all_boxes.append(label_boxes[i])
                all_labels.append(label)
                all_scores.append(label_scores[i])

        return np.stack(all_boxes), np.stack(all_labels), np.stack(all_scores)
