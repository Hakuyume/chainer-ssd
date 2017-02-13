import itertools
import numpy as np

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F


class MultiBox(chainer.Chain):

    def __init__(self, n_class, n_anchors):
        super().__init__(
            loc=chainer.ChainList(),
            conf=chainer.ChainList(),
        )

        self.n_class = n_class

        for n in n_anchors:
            self.loc.add_link(L.Convolution2D(
                None, n * 4, 3, stride=1, pad=1))
            self.conf.add_link(L.Convolution2D(
                None, n * (self.n_class + 1), 3, stride=1, pad=1))

    def __call__(self, xs):
        hs_loc = list()
        hs_conf = list()

        for i, x in enumerate(xs):
            h_loc = self.loc[i](x)
            h_loc = F.transpose(h_loc, (0, 2, 3, 1))
            h_loc = F.reshape(h_loc, (h_loc.shape[0], -1, 4))
            hs_loc.append(h_loc)

            h_conf = self.conf[i](x)
            h_conf = F.transpose(h_conf, (0, 2, 3, 1))
            h_conf = F.reshape(h_conf, (h_conf.shape[0], -1, self.n_class + 1))
            hs_conf.append(h_conf)

        hs_loc = F.concat(hs_loc, axis=1)
        hs_conf = F.concat(hs_conf, axis=1)

        return hs_loc, hs_conf

    def loss(self, x_loc, x_conf, t_loc, t_conf):
        xp = cuda.get_array_module(x_loc.data)

        loss_loc = F.reshape(
            F.huber_loss(
                F.reshape(x_loc, (-1, 4)),
                F.reshape(t_loc, (-1, 4)),
                1),
            t_conf.shape)
        loss_loc = F.where(
            t_conf.data > 0,
            loss_loc,
            xp.zeros_like(loss_loc.data))

        loss_conf = F.logsumexp(x_conf, axis=2) - F.reshape(
            F.select_item(
                F.reshape(x_conf, (-1, self.n_class + 1)),
                F.flatten(t_conf)),
            t_conf.shape)

        loss = F.sum(
            F.sum(loss_loc + loss_conf, axis=1) *
            xp.where(
                (t_conf.data > 0).any(axis=1),
                1 / (t_conf.data > 0).sum(axis=1),
                xp.zeros(t_conf.shape[:1])).astype(np.float32))

        return loss / t_conf.shape[0]


class MultiBoxEncoder:

    def __init__(self, n_scale, aspect_ratios, variance, grids):
        self.aspect_ratios = aspect_ratios
        self.variance = variance

        size = 300
        min_ratio, max_ratio = 20, 90
        step = int((max_ratio - min_ratio) / (n_scale - 2))
        min_sizes = [size * 10 // 100]
        max_sizes = [size * 20 // 100]
        for ratio in range(min_ratio, max_ratio + 1, step):
            min_sizes.append(size * ratio // 100)
            max_sizes.append(size * (ratio + step) // 100)

        boxes = list()
        for k in range(n_scale):
            for v, u in itertools.product(range(grids[k]), repeat=2):
                cx = (u + 0.5) / grids[k]
                cy = (v + 0.5) / grids[k]

                s = min_sizes[k] / size
                boxes.append((cx, cy, s, s))

                s = np.sqrt(min_sizes[k] * max_sizes[k]) / size
                boxes.append((cx, cy, s, s))

                s = min_sizes[k] / size
                for ar in self.aspect_ratios[k]:
                    boxes.append((cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    boxes.append((cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self.default_boxes = np.array(boxes)

    @property
    def n_anchors(self):
        return tuple((len(ar) + 1) * 2 for ar in self.aspect_ratios)

    def encode(self, boxes, classes, threshold=0.5):
        lt = np.maximum(
            (self.default_boxes[:, :2] -
             self.default_boxes[:, 2:] / 2)[:, np.newaxis],
            boxes[:, :2] - boxes[:, 2:] / 2)
        rb = np.minimum(
            (self.default_boxes[:, :2] +
             self.default_boxes[:, 2:] / 2)[:, np.newaxis],
            boxes[:, :2] + boxes[:, 2:] / 2)

        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_defaultboxes = np.prod(self.default_boxes[:, 2:], axis=1)
        area_boxes = np.prod(boxes[:, 2:], axis=1)
        iou = area_i / (area_defaultboxes[:, np.newaxis] + area_boxes - area_i)

        gt_idx = iou.argmax(axis=1)

        loc = np.hstack((
            (boxes[gt_idx][:, :2] - self.default_boxes[:, :2]) /
            (self.variance[0] * self.default_boxes[:, 2:]),
            np.log(boxes[gt_idx][:, 2:] / self.default_boxes[:, 2:]) /
            self.variance[1]))

        conf = 1 + classes[gt_idx]
        conf[(iou < threshold).all(axis=1)] = 0

        return loc.astype(np.float32), conf.astype(np.int32)

    def decode(self, loc, conf):
        boxes = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        conf = np.exp(conf)
        conf /= conf.sum(axis=1)[:, np.newaxis]

        return boxes, conf
