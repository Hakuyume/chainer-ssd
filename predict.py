#! /usr/bin/env python3

import argparse
import cv2
import itertools
import math
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

from ssd import SSD300

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    size = 300
    grids = (38, 19, 10, 5, 3, 1)
    steps = (8, 16, 32, 64, 100, 300)
    n_scale = 6
    min_ratio, max_ratio = 20, 90
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    variance = (0.1, 0.2)

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

            s = math.sqrt(min_sizes[k] * max_sizes[k]) / size
            boxes.append((cx, cy, s, s))

            s = min_sizes[k] / size
            for ar in aspect_ratios[k]:
                boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
    boxes = np.array(boxes)

    model = SSD300(
        n_class=20,
        n_anchors=((len(ar) + 1) * 2 for ar in aspect_ratios))
    serializers.load_npz(args.model, model)

    src = cv2.imread(args.image, cv2.IMREAD_COLOR)
    src = cv2.resize(src, (size, size))
    x = src.astype(np.float32)
    x -= (103.939, 116.779, 123.68)
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis]

    loc, conf = model(x)
    loc, = loc.data
    conf, = conf.data

    conf = np.exp(conf)
    conf /= conf.sum(axis=1)[:, np.newaxis]
    conf = conf[:, 1:]

    img = src.copy()
    for i in conf.max(axis=1).argsort()[:-4:-1]:
        print(conf[i].argmax(), conf[i].max())

        box = boxes[i] * 300
        box[:2] *= 1 + loc[i][:2] * variance[0]
        box[2:] *= np.exp(loc[i][2:] * variance[1])
        cv2.rectangle(
            img,
            tuple((box[:2] - box[2:] / 2).astype(int)),
            tuple((box[:2] + box[2:] / 2).astype(int)),
            (0, 0, 255),
            3)

    while True:
        cv2.imshow('result', img)
        if cv2.waitKey() == ord('q'):
            break
