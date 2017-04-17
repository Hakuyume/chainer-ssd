import itertools
import numpy as np

from lib import matrix_iou


class MultiBoxEncoder(object):

    def __init__(self, model, steps, sizes, variance):
        assert len(model.grids) == len(model.aspect_ratios)
        assert len(model.grids) == len(steps)
        assert len(model.grids) + 1 == len(sizes)

        self.variance = variance

        default_boxes = list()
        for k in range(len(model.grids)):
            for v, u in itertools.product(range(model.grids[k]), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                s = sizes[k]
                default_boxes.append((cx, cy, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                default_boxes.append((cx, cy, s, s))

                s = sizes[k]
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
        conf = conf[:, 1:]

        for label in range(conf.shape[1]):
            selection = np.zeros(conf.shape[0], dtype=bool)
            for i in conf[:, label].argsort()[::-1]:
                if conf[i, label] < conf_threshold:
                    break
                iou = matrix_iou(
                    boxes[np.newaxis, i],
                    boxes[selection])
                if (iou > nms_threshold).any():
                    continue
                selection[i] = True
                yield boxes[i], label, conf[i, label]
