import itertools
import numpy as np

import chainer
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

    def decode(self, loc, conf):
        loc = np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
        conf = np.exp(conf)
        conf /= conf.sum(axis=1)[:, np.newaxis]

        return loc, conf
