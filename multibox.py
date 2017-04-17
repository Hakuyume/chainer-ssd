import itertools
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import rect


class MultiBox(chainer.Chain):

    def mine_hard_negative(self, n, loss_conf, t_conf):
        loss_conf = chainer.cuda.to_cpu(loss_conf.data).copy()
        t_conf = chainer.cuda.to_cpu(t_conf.data)

        loss_conf = loss_conf.reshape((n, -1))
        t_conf = t_conf.reshape((n, -1))

        loss_conf[t_conf > 0] = 0
        rank = (-loss_conf).argsort(axis=1).argsort(axis=1)
        hard_neg = rank < ((t_conf > 0).sum(axis=1) * 3)[:, np.newaxis]

        return self.xp.array(hard_neg).flatten()

    def loss(self, x_loc, x_conf, t_loc, t_conf):
        xp = self.xp
        pos = (t_conf.data > 0).flatten()
        if xp.logical_not(pos).all():
            return 0, 0

        x_loc = F.reshape(x_loc, (-1, 4))
        t_loc = F.reshape(t_loc, (-1, 4))
        loss_loc = F.huber_loss(x_loc, t_loc, 1)
        loss_loc = F.where(pos, loss_loc, xp.zeros_like(loss_loc.data))
        loss_loc = F.sum(loss_loc) / pos.sum()

        n = t_conf.shape[0]
        x_conf = F.reshape(x_conf, (-1, self.n_classes + 1))
        t_conf = F.flatten(t_conf)
        loss_conf = F.logsumexp(x_conf, axis=1) - F.select_item(x_conf, t_conf)
        hard_neg = self.mine_hard_negative(n, loss_conf, t_conf)
        loss_conf = F.where(
            xp.logical_or(pos, hard_neg),
            loss_conf, xp.zeros_like(loss_conf.data))
        loss_conf = F.sum(loss_conf) / pos.sum()

        return loss_loc, loss_conf


class MultiBoxEncoder(object):

    def __init__(self, grids, steps, sizes, aspect_ratios, variance):
        self.aspect_ratios = aspect_ratios
        self.variance = variance

        boxes = list()
        for k in range(len(grids)):
            for v, u in itertools.product(range(grids[k]), repeat=2):
                cx = (u + 0.5) * steps[k]
                cy = (v + 0.5) * steps[k]

                s = sizes[k]
                boxes.append((cx, cy, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                boxes.append((cx, cy, s, s))

                s = sizes[k]
                for ar in self.aspect_ratios[k]:
                    boxes.append((cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    boxes.append((cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self.default_boxes = np.array(boxes)

    def encode(self, boxes, classes, threshold=0.5):
        if len(boxes) == 0:
            return (
                np.empty(self.default_boxes.shape, dtype=np.float32),
                np.zeros(self.default_boxes.shape[:1], dtype=np.int32))

        iou = rect.matrix_iou(
            np.hstack((
                self.default_boxes[:, :2] - self.default_boxes[:, 2:] / 2,
                self.default_boxes[:, :2] + self.default_boxes[:, 2:] / 2)),
            boxes)
        gt_idx = iou.argmax(axis=1)
        iou = iou.max(axis=1)
        boxes = boxes[gt_idx]
        classes = classes[gt_idx]

        loc = np.hstack((
            ((boxes[:, :2] + boxes[:, 2:]) / 2 - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log((boxes[:, 2:] - boxes[:, :2]) / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + classes
        conf[iou < threshold] = 0

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc, conf):
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        conf = np.exp(conf)
        conf /= conf.sum(axis=1, keepdims=True)
        conf = conf[:, 1:]

        return boxes, conf

    def non_maximum_suppression(
            self, boxes, conf, nms_threshold, conf_threshold):
        for cls in range(conf.shape[1]):
            selected = np.zeros((conf.shape[0],), dtype=bool)
            for i in conf[:, cls].argsort()[::-1]:
                if conf[i, cls] < conf_threshold:
                    break
                box = rect.Rect.LTRB(*boxes[i])
                iou = rect.matrix_iou(
                    boxes[np.newaxis, i],
                    boxes[selected])
                if (iou >= nms_threshold).any():
                    continue
                selected[i] = True
                yield box, cls, conf[i, cls]
