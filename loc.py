import itertools
import numpy as np


class LocEncoder:

    def __init__(self, size, n_scale, aspect_ratios, variance, grids):
        self.variance = variance

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
                for ar in aspect_ratios[k]:
                    boxes.append((cx, cy, s * np.sqrt(ar), s / np.sqrt(ar)))
                    boxes.append((cx, cy, s / np.sqrt(ar), s * np.sqrt(ar)))
        self.default_boxes = np.array(boxes)

    def decode(self, loc):
        return np.hstack((
            self.default_boxes[:, :2] +
            loc[:, :2] * self.variance[0] * self.default_boxes[:, 2:],
            self.default_boxes[:, 2:] * np.exp(loc[:, 2:] * self.variance[1])))
