#! /usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plot
import numpy as np

import chainer
from chainer import serializers

from lib import MultiBoxEncoder
from lib import SSD300
from lib import VOCDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(20)
    serializers.load_npz(args.model, model)

    multibox_encoder = MultiBoxEncoder(model)

    src = cv2.imread(args.image, cv2.IMREAD_COLOR)

    x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
    x -= model.mean
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis]

    loc, conf = model(x)
    loc = chainer.cuda.to_cpu(loc.data)
    conf = chainer.cuda.to_cpu(conf.data)
    results = multibox_encoder.decode(loc[0], conf[0], 0.45, 0.01)

    figure = plot.figure()
    ax = figure.add_subplot(111)
    ax.imshow(src[:, :, ::-1])

    for box, label, score in results:
        box[:2] *= src.shape[1::-1]
        box[2:] *= src.shape[1::-1]
        box = box.astype(int)

        print(label + 1, score, *box)

        if score > 0.6:
            ax.add_patch(plot.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor='red', linewidth=3))
            ax.text(
                box[0], box[1], VOCDataset.labels[label],
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    plot.show()
